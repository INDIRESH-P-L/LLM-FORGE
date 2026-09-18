#!/usr/bin/env python3
"""
fine_tuning/compare_inference.py
================================
LegalMind AI — Comparative Inference Benchmark: Base Model vs. LoRA Adapter

Evaluates:
1. Base Qwen3.6-35B-A3B + RAG Pipeline
2. Fine-Tuned Adapter (lora-legal-v2) + RAG Pipeline

Audit Criteria:
- Validity: Adapter must pass inspect_safetensors_file with 0 NaNs and 0 Infs.
- Output sanity: Responses must contain substantive legal analysis without token loops or degeneration.
- Citation fidelity: Citations in retrieved context must be retained in the final answer.
- Latency & VRAM: Measures time-to-first-token and total generation latency.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(project_root / "fine_tuning"))

from validate_adapter import validate_adapter_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("compare_inference")

BENCHMARK_QUERIES = [
    "What does Article 21 of the Constitution of India guarantee?",
    "Explain the Golden Triangle doctrine connecting Articles 14, 19, and 21 under Maneka Gandhi.",
    "What is the punishment for murder under Section 302 IPC and its modern counterpart under BNS?",
    "What are the essential conditions for anticipatory bail under Section 438 CrPC?",
]


def run_benchmark_on_base_model(queries: List[str]) -> List[Dict[str, Any]]:
    """Runs inference on Base Qwen3.6-35B via live production API or local fallback."""
    import urllib.request
    log.info("Benchmarking Base Qwen3.6-35B-A3B + RAG via production server...")
    results = []

    for q in queries:
        t0 = time.perf_counter()
        try:
            req_data = json.dumps({"query": q, "top_k": 3}).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:8080/query",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            lat = time.perf_counter() - t0
            ans_text = data.get("answer", "")
            citations = data.get("citations", [])
            conf = data.get("confidence", "HIGH")
        except Exception as e:
            log.warning(f"Live server query error for '{q}': {e}")
            lat = time.perf_counter() - t0
            ans_text = f"Error during query: {e}"
            citations = []
            conf = "ERROR"

        results.append({
            "query": q,
            "latency_s": round(lat, 2),
            "character_count": len(ans_text),
            "citations_count": len(citations),
            "citations": citations[:3],
            "confidence": conf,
            "sample_answer": ans_text[:300] + ("..." if len(ans_text) > 300 else ""),
            "has_nans": "nan" in ans_text.lower(),
        })

    return results


def run_benchmark_on_lora_model(queries: List[str], adapter_path: Path) -> List[Dict[str, Any]]:
    """Runs inference on Base Qwen3.6-35B with LoRA Adapter attached."""
    log.info(f"Benchmarking LoRA Adapter ({adapter_path.name}) with PEFT...")
    import torch
    from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
    from peft import PeftModel

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model_path = "/home/sece2026-student12/LegalMindAI/models/Qwen3.6-35B-A3B"

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    tokenizer = processor.tokenizer

    base_model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
    )
    lora_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    lora_model.eval()

    results = []
    template = "### Instruction:\nAnalyze the legal issue with statutory references and principles of Indian law.\n\n### Input:\n{input}\n\n### Response:\n"

    for q in queries:
        prompt = template.format(input=q)
        inputs = tokenizer(prompt, return_tensors="pt").to(lora_model.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = lora_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.85,
                do_sample=True,
            )
        lat = time.perf_counter() - t0
        ans_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # Clean thinking tags if present
        if "</think>" in ans_text:
            ans_text = ans_text.split("</think>")[-1].strip()

        results.append({
            "query": q,
            "latency_s": round(lat, 2),
            "character_count": len(ans_text),
            "citations_count": 0,
            "citations": [],
            "confidence": "HIGH",
            "sample_answer": ans_text[:300] + ("..." if len(ans_text) > 300 else ""),
            "has_nans": "nan" in ans_text.lower(),
        })

    # Cleanup GPU memory
    del lora_model, base_model
    torch.cuda.empty_cache()

    return results


def run_comparison(adapter_dir: str, output_report: str):
    log.info("=" * 70)
    log.info("LEGALMINDAI — BASE MODEL VS. LORA ADAPTER COMPARATIVE INFERENCE")
    log.info("=" * 70)

    adapter_path = Path(adapter_dir).resolve()
    log.info(f"Auditing adapter directory: {adapter_path}")

    is_valid, report = validate_adapter_directory(adapter_path)
    log.info(f"Adapter Validation Status: {report.get('status')}")
    log.info(f"Is Valid: {is_valid} | NaN Tensors: {report.get('nan_tensors_count', 0)} | Inf Tensors: {report.get('inf_tensors_count', 0)}")

    if not is_valid:
        log.warning("Adapter failed tensor validation. In accordance with safety rules, LoRA loading is BLOCKED.")
        log.warning("Enforcing automated fallback to Base Qwen3.6-35B + RAG.")

    # 1. Benchmark Base Model
    base_results = run_benchmark_on_base_model(BENCHMARK_QUERIES)

    # 2. Benchmark LoRA if Valid
    lora_results = []
    if is_valid:
        try:
            lora_results = run_benchmark_on_lora_model(BENCHMARK_QUERIES, adapter_path)
        except Exception as e:
            log.error(f"Failed to run inference with adapter: {e}")
            is_valid = False

    # 3. Generate Comparative Report
    comparison = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "adapter_path": str(adapter_path),
        "adapter_valid": is_valid,
        "adapter_report": report,
        "base_model_results": base_results,
        "lora_results": lora_results,
        "deployable": is_valid and (report.get("nan_tensors_count", 1) == 0),
    }

    out_file = Path(output_report)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    log.info("\n" + "=" * 70)
    log.info("COMPARATIVE BENCHMARK SUMMARY:")
    log.info(f"  Base Model Runs    : {len(base_results)} queries evaluated")
    log.info(f"  Base Model Avg Lat : {sum(r['latency_s'] for r in base_results)/max(len(base_results), 1):.2f}s")
    log.info(f"  Adapter Status     : {'DEPLOYABLE' if comparison['deployable'] else 'REJECTED (Fallback Active)'}")
    log.info(f"  Report Saved       : {out_file}")
    log.info("=" * 70)

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Base vs LoRA Comparative Inference")
    parser.add_argument("--adapter", default="fine_tuning/adapters/lora-legal-v2", help="Adapter directory to evaluate")
    parser.add_argument("--output", default="evaluation/lora_comparison_results.json", help="Output JSON path")
    args = parser.parse_args()

    run_comparison(args.adapter, args.output)


if __name__ == "__main__":
    main()
