#!/usr/bin/env python3
"""
fine_tuning/validate_and_compare_v2.py
======================================
LegalMind AI — Complete Post-Training Validation & Comparative Benchmark Suite

Evaluates:
- Base Qwen3.6-35B-A3B + RAG Pipeline
- Fine-Tuned Standard LoRA Adapter (lora-legal-v2) + RAG Pipeline

Mandated Test Categories:
1. Article 14
2. Article 21
3. Fundamental Rights
4. Constitution overview
5. Indian Penal Code/BNS distinction
6. Criminal procedure
7. Contract law
8. Case-law explanation
9. Historical-law question
10. Unsupported/non-legal question
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(project_root / "fine_tuning"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate_v2")

from validate_adapter import validate_adapter_directory, inspect_safetensors_file

import importlib.util
spec_rag = importlib.util.spec_from_file_location(
    "rag_module", str(project_root / "scripts" / "05_rag.py")
)
rag_module = importlib.util.module_from_spec(spec_rag)
sys.modules["rag_module"] = rag_module
spec_rag.loader.exec_module(rag_module)

MODEL_PATH = rag_module.MODEL_PATH
SYSTEM_PROMPT = rag_module.SYSTEM_PROMPT
build_rag_prompt = rag_module.build_rag_prompt
strip_thinking = rag_module.strip_thinking
_parse_structured_answer = rag_module._parse_structured_answer
LegalMindModel = rag_module.LegalMindModel

from retriever import HybridRetriever, format_citations
from constitutional_retrieval import get_constitutional_retriever, is_constitutional_query, is_golden_triangle_query
from citation_verifier import extract_cited_provisions_from_text, verify_citations_in_response

TEST_QUESTIONS = [
    {
        "category": "Article 14",
        "query": "What does Article 14 of the Constitution of India guarantee and what is the arbitrariness test under E.P. Royappa?",
        "expected_legal_terms": ["article 14", "equality", "arbitrariness", "royappa", "reasonable classification"],
        "is_legal": True,
    },
    {
        "category": "Article 21",
        "query": "What does Article 21 of the Constitution of India guarantee regarding right to life and personal liberty?",
        "expected_legal_terms": ["article 21", "life", "personal liberty", "procedure established by law", "maneka gandhi"],
        "is_legal": True,
    },
    {
        "category": "Fundamental Rights",
        "query": "What is the scope of Fundamental Rights under Part III of the Constitution and can they be amended?",
        "expected_legal_terms": ["part iii", "fundamental rights", "article 368", "basic structure", "kesavananda"],
        "is_legal": True,
    },
    {
        "category": "Constitution overview",
        "query": "Provide an overview of the Constitution of India, its preamble, basic structure, and schedule arrangement.",
        "expected_legal_terms": ["preamble", "basic structure", "schedules", "constitution of india"],
        "is_legal": True,
    },
    {
        "category": "Indian Penal Code/BNS distinction",
        "query": "Explain the transition from Section 302 IPC to Section 103 BNS for the offence of murder, including changes in statutory definition and punishments.",
        "expected_legal_terms": ["section 302", "section 103", "bns", "bharatiya nyaya sanhita", "murder"],
        "is_legal": True,
    },
    {
        "category": "Criminal procedure",
        "query": "What are the essential statutory conditions and procedural rules for granting anticipatory bail under Section 438 CrPC and its corresponding BNSS provision?",
        "expected_legal_terms": ["section 438", "crpc", "bnss", "anticipatory bail", "section 482"],
        "is_legal": True,
    },
    {
        "category": "Contract law",
        "query": "What constitutes a valid contract under Section 10 and lawful consideration under Section 23 of the Indian Contract Act, 1872?",
        "expected_legal_terms": ["section 10", "section 23", "consideration", "contract act", "void"],
        "is_legal": True,
    },
    {
        "category": "Case-law explanation",
        "query": "Explain the landmark ruling and ratio decidendi of Kesavananda Bharati v. State of Kerala (1973).",
        "expected_legal_terms": ["kesavananda bharati", "basic structure", "ratio decidendi", "article 368"],
        "is_legal": True,
    },
    {
        "category": "Historical-law question",
        "query": "Explain the repeal of the Companies Act, 1956 by the Companies Act, 2013 and the corporate doctrine of ultra vires.",
        "expected_legal_terms": ["companies act, 1956", "companies act, 2013", "repeal", "ultra vires"],
        "is_legal": True,
    },
    {
        "category": "Unsupported/non-legal question",
        "query": "Explain the quantum mechanical superposition of states and Schrödinger's wave equation in Hilbert space.",
        "expected_legal_terms": ["insufficient", "not legal", "general knowledge", "not covered"],
        "is_legal": False,
    },
]


def check_repetitions(text: str, n: int = 4) -> float:
    """Computes repeated n-gram ratio to verify absence of degeneration loops."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    unique = len(set(ngrams))
    total = len(ngrams)
    return round(1.0 - (unique / total), 4)


def evaluate_response_quality(
    query: str,
    answer_text: str,
    retrieved_chunks: List[Dict[str, Any]],
    expected_terms: List[str],
    is_legal: bool,
    latency: float,
) -> Dict[str, Any]:
    ans_lower = answer_text.lower()
    
    # 1. Legal terminology score
    term_hits = sum(1 for t in expected_terms if t in ans_lower)
    terminology_score = round(term_hits / max(len(expected_terms), 1), 3)

    # 2. Extract citations
    extracted_citations = extract_cited_provisions_from_text(answer_text)
    
    # 3. Citation & Hallucination verification
    verif = verify_citations_in_response(answer_text, retrieved_chunks, query=query)
    
    # 4. Source grounding
    grounding_score = 1.0 - verif.hallucination_score
    if not is_legal:
        # For non-legal query, honesty in declaring lack of evidence is desired
        is_honest_abstention = any(w in ans_lower for w in ["insufficient", "not covered", "general knowledge", "non-legal", "cannot find"])
        grounding_score = 1.0 if is_honest_abstention else 0.5

    # 5. Logical order (Heading inspection)
    has_answer = "## answer" in ans_lower or "1. direct answer" in ans_lower or "###" in ans_lower
    has_provisions = "provision" in ans_lower or "statut" in ans_lower
    has_reasoning = "reasoning" in ans_lower or "analysis" in ans_lower or "doctrine" in ans_lower
    logical_order_score = 1.0 if (has_answer and (has_provisions or has_reasoning)) else 0.7

    # 6. Repetition check
    rep_score = check_repetitions(answer_text, n=4)

    # 7. Unsupported claim rate
    unsupported_rate = round(verif.unverified_count / max(verif.total_citations_found, 1), 3) if verif.total_citations_found > 0 else 0.0

    return {
        "character_count": len(answer_text),
        "latency_s": round(latency, 2),
        "terminology_score": terminology_score,
        "logical_order_score": logical_order_score,
        "citations_count": len(extracted_citations),
        "citations_extracted": extracted_citations[:5],
        "verified_citations": verif.verified_count,
        "unverified_citations": verif.unverified_count,
        "hallucination_rate": round(verif.hallucination_score, 3),
        "unsupported_claim_rate": unsupported_rate,
        "grounding_score": round(grounding_score, 3),
        "confidence_label": verif.confidence_label,
        "repetition_rate": rep_score,
        "has_nans": "nan" in ans_lower or math.isnan(latency),
        "sample": answer_text[:280] + ("..." if len(answer_text) > 280 else ""),
    }


def execute_full_validation_suite(adapter_dir: str, output_json: str):
    import torch
    from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
    from peft import PeftModel

    adapter_path = Path(adapter_dir).resolve()
    log.info("=" * 80)
    log.info("LEGALMINDAI — COMPLETE POST-TRAINING VALIDATION & COMPARATIVE BENCHMARK")
    log.info(f"Target Adapter: {adapter_path}")
    log.info("=" * 80)

    # Step 1: Pre-inference safetensors audit
    is_valid, report = validate_adapter_directory(adapter_path)
    log.info(f"Safetensors Audit: Status={report.get('status')}, Valid={is_valid}, NaNs={report.get('nan_tensors_count')}, Infs={report.get('inf_tensors_count')}")
    if not is_valid:
        raise ValueError(f"CRITICAL: Adapter failed safety audit: {report}")

    # Step 2: Initialize HybridRetriever on GPU 4
    log.info("Initializing HybridRetriever on GPU 4...")
    retriever = HybridRetriever(
        chunks_dir="data/chunks/legislation/",
        vector_db_dir="vector_db/legislation/",
        embed_device="cuda:4",
        rerank_device="cuda:4",
    )
    const_retriever = get_constitutional_retriever(base_retriever=retriever)
    log.info("Retrievers successfully loaded.")

    # Step 3: Load Base Model on GPU 1 (Native BF16, Zero Quantization)
    llm_gpu = 1
    log.info(f"Loading Base Qwen3.6-35B-A3B in native BF16 on GPU {llm_gpu}...")
    t0_load = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    base_model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": llm_gpu},
        local_files_only=True,
    )
    base_model.eval()
    base_load_time = round(time.time() - t0_load, 2)
    log.info(f"Base model loaded in {base_load_time}s.")

    # Step 4: Attach LoRA Adapter via PEFT
    log.info(f"Attaching Standard LoRA adapter ({adapter_path.name}) with PEFT...")
    t0_lora = time.time()
    lora_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    lora_model.eval()
    lora_load_time = round(time.time() - t0_lora, 2)
    log.info(f"LoRA adapter attached in {lora_load_time}s.")

    # In-memory post-loading tensor health check
    log.info("Auditing loaded adapter tensors in VRAM for NaN/Inf...")
    nan_params = []
    inf_params = []
    total_lora_params = 0
    for name, param in lora_model.named_parameters():
        if "lora_" in name:
            total_lora_params += param.numel()
            if torch.isnan(param.data).any().item():
                nan_params.append(name)
            if torch.isinf(param.data).any().item():
                inf_params.append(name)

    log.info(f"In-Memory Tensor Audit: {total_lora_params:,} LoRA parameters verified. NaNs={len(nan_params)}, Infs={len(inf_params)}")
    if nan_params or inf_params:
        raise FloatingPointError(f"CRITICAL: In-memory tensors contain NaNs: {nan_params}")

    # Generate function for RAG prompts
    def generate_rag_response(model_to_use, query_text, chunks, citations_list):
        rag_prompt = build_rag_prompt(query_text, chunks, citations_list)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rag_prompt},
        ]
        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            )
        except TypeError:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
        model_device = getattr(model_to_use, "device", None)
        if model_device is None:
            model_device = next(model_to_use.parameters()).device
        inputs = {k: v.to(model_device) if hasattr(v, "to") else v for k, v in inputs.items()}
        
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model_to_use.generate(
                **inputs,
                max_new_tokens=384,
                temperature=0.3,
                top_p=0.85,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        lat = time.perf_counter() - t0
        raw = processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        cleaned = strip_thinking(raw)
        return cleaned, lat

    # Step 5: Execute 10-Question Benchmark
    benchmark_records = []
    log.info(f"Beginning 10-Category Comparative Benchmark...")

    for idx, item in enumerate(TEST_QUESTIONS, 1):
        cat = item["category"]
        q = item["query"]
        expected = item["expected_legal_terms"]
        is_legal = item["is_legal"]
        log.info(f"[{idx}/10] Testing Category: '{cat}' | Query: {q[:60]}...")

        # 1. Retrieve evidence (identical for both Base and LoRA)
        t_ret_start = time.perf_counter()
        if (is_constitutional_query(q) or is_golden_triangle_query(q)) and is_legal:
            chunks, _ = const_retriever.retrieve(q, top_k=7)
        else:
            chunks = retriever.retrieve(q, top_k=5)
        ret_latency = round(time.perf_counter() - t_ret_start, 3)
        citations = format_citations(chunks)

        # 2. Run BASE + RAG
        # Disable adapter context temporarily
        with lora_model.disable_adapter():
            base_ans, base_lat = generate_rag_response(lora_model, q, chunks, citations)
        base_eval = evaluate_response_quality(q, base_ans, chunks, expected, is_legal, base_lat)

        # 3. Run LORA + RAG
        lora_ans, lora_lat = generate_rag_response(lora_model, q, chunks, citations)
        lora_eval = evaluate_response_quality(q, lora_ans, chunks, expected, is_legal, lora_lat)

        rec = {
            "index": idx,
            "category": cat,
            "query": q,
            "retrieval": {
                "chunk_count": len(chunks),
                "citations_count": len(citations),
                "latency_s": ret_latency,
            },
            "base_rag": {
                "answer": base_ans,
                "metrics": base_eval,
            },
            "lora_rag": {
                "answer": lora_ans,
                "metrics": lora_eval,
            },
        }
        benchmark_records.append(rec)
        log.info(f"     Base Lat: {base_eval['latency_s']}s (Grounding: {base_eval['grounding_score']}) | LoRA Lat: {lora_eval['latency_s']}s (Grounding: {lora_eval['grounding_score']})")

    # Step 6: Consolidate Metrics
    base_avg_lat = round(sum(r["base_rag"]["metrics"]["latency_s"] for r in benchmark_records) / len(benchmark_records), 2)
    lora_avg_lat = round(sum(r["lora_rag"]["metrics"]["latency_s"] for r in benchmark_records) / len(benchmark_records), 2)

    base_avg_grounding = round(sum(r["base_rag"]["metrics"]["grounding_score"] for r in benchmark_records) / len(benchmark_records), 3)
    lora_avg_grounding = round(sum(r["lora_rag"]["metrics"]["grounding_score"] for r in benchmark_records) / len(benchmark_records), 3)

    base_avg_hallucination = round(sum(r["base_rag"]["metrics"]["hallucination_rate"] for r in benchmark_records) / len(benchmark_records), 3)
    lora_avg_hallucination = round(sum(r["lora_rag"]["metrics"]["hallucination_rate"] for r in benchmark_records) / len(benchmark_records), 3)

    base_total_cits = sum(r["base_rag"]["metrics"]["citations_count"] for r in benchmark_records)
    lora_total_cits = sum(r["lora_rag"]["metrics"]["citations_count"] for r in benchmark_records)

    is_production_ready = (
        is_valid
        and len(nan_params) == 0
        and len(inf_params) == 0
        and lora_avg_hallucination <= base_avg_hallucination + 0.05
        and lora_avg_grounding >= 0.85
        and not any(r["lora_rag"]["metrics"]["has_nans"] for r in benchmark_records)
    )

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "NVIDIA DGX B200",
        "base_model": MODEL_PATH,
        "adapter_path": str(adapter_path),
        "adapter_audit": report,
        "in_memory_tensor_audit": {
            "total_lora_params": total_lora_params,
            "nan_params_count": len(nan_params),
            "inf_params_count": len(inf_params),
            "clean": len(nan_params) == 0 and len(inf_params) == 0,
        },
        "summary_metrics": {
            "base_avg_latency_s": base_avg_lat,
            "lora_avg_latency_s": lora_avg_lat,
            "base_avg_grounding": base_avg_grounding,
            "lora_avg_grounding": lora_avg_grounding,
            "base_avg_hallucination_rate": base_avg_hallucination,
            "lora_avg_hallucination_rate": lora_avg_hallucination,
            "base_total_citations": base_total_cits,
            "lora_total_citations": lora_total_cits,
            "is_production_ready": is_production_ready,
        },
        "detailed_results": benchmark_records,
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    log.info("=" * 80)
    log.info("POST-TRAINING VALIDATION & BENCHMARK SUMMARY:")
    log.info(f"  In-Memory Tensors    : 100% Clean (0 NaNs, 0 Infs across {total_lora_params:,} parameters)")
    log.info(f"  Base RAG Avg Latency : {base_avg_lat}s | Grounding: {base_avg_grounding} | Hallucination: {base_avg_hallucination}")
    log.info(f"  LoRA RAG Avg Latency : {lora_avg_lat}s | Grounding: {lora_avg_grounding} | Hallucination: {lora_avg_hallucination}")
    log.info(f"  Production Ready     : {is_production_ready}")
    log.info(f"  Results Exported     : {out_p}")
    log.info("=" * 80)

    return final_payload


def main():
    parser = argparse.ArgumentParser(description="LegalMindAI Complete Post-Training Validation")
    parser.add_argument("--adapter", default="fine_tuning/adapters/lora-legal-v2", help="Adapter path")
    parser.add_argument("--output", default="evaluation/lora_legal_v2_comparison_data.json", help="Output JSON path")
    args = parser.parse_args()
    execute_full_validation_suite(args.adapter, args.output)


if __name__ == "__main__":
    main()
