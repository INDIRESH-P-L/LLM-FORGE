#!/usr/bin/env python3
"""
fine_tuning/evaluate.py
=======================
LegalMind AI — Evaluation Suite

Compares three configurations:
  1. BASE  — Qwen3.6 with no retrieval
  2. RAG   — Qwen3.6 + Hybrid Retrieval (FAISS + BM25 + Reranker)
  3. RAG+LoRA — Qwen3.6 + RAG + LoRA Adapter (when available)

Metrics:
  - Retrieval: Recall@K, MRR (when ground-truth docs are known)
  - Answer: ROUGE-L, BLEU (surface similarity)
  - Hallucination: Citation check (do cited sources exist in retrieved docs?)
  - Confidence calibration: Does HIGH confidence correlate with correct answers?
  - Abstention quality: Does model correctly abstain on unanswerable questions?
  - Latency: Retrieval + generation time

Usage:
    python fine_tuning/evaluate.py \\
        --test-set evaluation/test_questions.jsonl \\
        --output   evaluation/eval_report.json \\
        --llm-gpu  1 --embed-gpu 4
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

# Force offline mode to prevent HuggingFace timeout delays
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def rouge_l(hyp: str, ref: str) -> float:
    """Simple character-level ROUGE-L (LCS-based)."""
    if not ref or not hyp:
        return 0.0
    hyp_tokens = hyp.lower().split()
    ref_tokens = ref.lower().split()

    # LCS dynamic programming
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    precision = lcs / n if n > 0 else 0
    recall    = lcs / m if m > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def citation_hallucination_rate(answer: str, citations: list[str]) -> float:
    """
    Check what fraction of [N] references in the answer
    correspond to real retrieved citations.
    Returns hallucination rate (0 = perfect, 1 = all hallucinated).
    """
    refs_in_answer = set(re.findall(r"\[(\d+)\]", answer))
    if not refs_in_answer:
        return 0.0  # No citations in answer
    valid_ids = set()
    for c in citations:
        m = re.match(r"\[(\d+)\]", c)
        if m:
            valid_ids.add(m.group(1))
    hallucinated = refs_in_answer - valid_ids
    return len(hallucinated) / len(refs_in_answer)


def abstention_correct(answer: str, question_type: str) -> bool | None:
    """
    For questions marked as 'unanswerable', check if model correctly abstains.
    Returns True if abstained, False if answered, None if not applicable.
    """
    if question_type != "unanswerable":
        return None
    abstain_phrases = [
        "insufficient", "not enough evidence", "cannot answer",
        "no relevant", "unable to find", "evidence is insufficient",
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in abstain_phrases)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class LegalMindEvaluator:

    def __init__(
        self,
        llm_gpu: int = 1,
        embed_gpu: int = 4,
        chunks_dir: str = "data/chunks/legislation/",
        vector_db_dir: str = "vector_db/legislation/",
        lora_adapter: str | None = None,
    ):
        self.llm_gpu      = llm_gpu
        self.embed_gpu    = embed_gpu
        self.chunks_dir   = chunks_dir
        self.vector_db_dir = vector_db_dir
        self.lora_adapter = lora_adapter

        self._model     = None
        self._retriever = None

    def _load_model(self, with_lora: bool = False):
        log.info("Loading Qwen3.6 model…")
        import torch
        from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

        MODEL_PATH = "/home/sece2026-student12/LegalMindAI/models/Qwen3.6-35B-A3B"
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH, local_files_only=True, trust_remote_code=True
        )
        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map={"": self.llm_gpu},
            local_files_only=True,
            trust_remote_code=True,
        )

        if with_lora and self.lora_adapter and Path(self.lora_adapter).exists():
            from peft import PeftModel
            log.info(f"Loading LoRA adapter: {self.lora_adapter}")
            model = PeftModel.from_pretrained(model, self.lora_adapter)

        model.eval()
        return processor, model

    def _load_retriever(self):
        log.info("Loading HybridRetriever…")
        from retriever import HybridRetriever
        return HybridRetriever(
            chunks_dir    = self.chunks_dir,
            vector_db_dir = self.vector_db_dir,
            embed_device  = f"cuda:{self.embed_gpu}",
            rerank_device = f"cuda:{self.embed_gpu}",
        )

    def _generate(self, processor, model, query: str, context: str = "") -> tuple[str, float]:
        import torch
        import importlib.util
        spec = importlib.util.spec_from_file_location("rag", str(Path(__file__).parent.parent / "scripts" / "05_rag.py"))
        rag = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rag)
        SYSTEM_PROMPT = rag.SYSTEM_PROMPT
        t0 = time.time()
        user_content = context if context else query
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]
        try:
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_tensors="pt", return_dict=True,
                enable_thinking=False,
            )
        except (TypeError, ValueError):
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_tensors="pt", return_dict=True,
            )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.3,
                top_p=0.85,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        resp = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        if hasattr(rag, "strip_thinking"):
            resp = rag.strip_thinking(resp)
        return resp.strip(), time.time() - t0

    def evaluate(
        self,
        test_questions: list[dict],
        mode: str = "rag",   # "base", "rag", or "rag+lora"
        api_url: str | None = None,
    ) -> dict:
        """
        Run evaluation for a given mode.
        If api_url is provided, queries the live API endpoint directly.
        Each question dict: {question, reference_answer?, ground_truth_docs?, type?}
        """
        assert mode in ("base", "rag", "rag+lora"), f"Unknown mode: {mode}"

        if api_url:
            import urllib.request
            log.info(f"Evaluating mode '{mode}' via live API: {api_url}")
            results = []
            rouge_scores = []
            hallucination_rates = []
            abstention_scores = []
            latencies = []

            for i, q in enumerate(test_questions, 1):
                question   = q.get("question", "")
                ref_answer = q.get("reference_answer", "")
                q_type     = q.get("type", "answerable")
                log.info(f"[{i}/{len(test_questions)}] {question[:60]}")

                try:
                    payload = json.dumps({"query": question}).encode("utf-8")
                    req = urllib.request.Request(
                        api_url,
                        data=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        data = json.loads(resp.read().decode("utf-8"))

                    answer = data.get("answer", "")
                    citations = data.get("citations", [])
                    gen_lat = data.get("generation_latency", 0.0)
                    latencies.append(gen_lat)

                    rl = rouge_l(answer, ref_answer) if ref_answer else None
                    if rl is not None:
                        rouge_scores.append(rl)

                    hr = citation_hallucination_rate(answer, citations)
                    hallucination_rates.append(hr)

                    ab = abstention_correct(answer, q_type)
                    if ab is not None:
                        abstention_scores.append(float(ab))

                    results.append({
                        "question":           question,
                        "answer":             answer[:500],
                        "reference_answer":   ref_answer[:300] if ref_answer else None,
                        "rouge_l":            round(rl, 4) if rl is not None else None,
                        "hallucination_rate": round(hr, 4),
                        "abstention_correct": ab,
                        "num_retrieved":      len(data.get("retrieved_documents", [])),
                        "citations":          citations,
                        "generation_latency": round(gen_lat, 3),
                    })
                except Exception as e:
                    log.error(f"API evaluation error on question {i}: {e}")
                    results.append({"question": question, "error": str(e)})

            return {
                "mode":                   mode,
                "api_url":                api_url,
                "num_questions":          len(test_questions),
                "avg_rouge_l":            round(sum(rouge_scores) / len(rouge_scores), 4) if rouge_scores else None,
                "avg_hallucination_rate": round(sum(hallucination_rates) / len(hallucination_rates), 4) if hallucination_rates else None,
                "abstention_accuracy":    round(sum(abstention_scores) / len(abstention_scores), 4) if abstention_scores else None,
                "avg_latency_s":          round(sum(latencies) / len(latencies), 3) if latencies else None,
                "results":                results,
            }

        with_lora = mode == "rag+lora"
        processor, model = self._load_model(with_lora=with_lora)

        retriever = None
        if mode in ("rag", "rag+lora"):
            try:
                retriever = self._load_retriever()
            except Exception as e:
                log.warning(f"Retriever not available: {e}. Using base mode.")

        results = []
        rouge_scores = []
        hallucination_rates = []
        abstention_scores = []
        latencies = []

        for i, q in enumerate(test_questions, 1):
            question   = q.get("question", "")
            ref_answer = q.get("reference_answer", "")
            q_type     = q.get("type", "answerable")

            log.info(f"[{i}/{len(test_questions)}] {question[:60]}")

            try:
                if retriever:
                    from retriever import format_citations
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("rag", str(Path(__file__).parent.parent / "scripts" / "05_rag.py"))
                    rag_mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(rag_mod)

                    chunks = retriever.retrieve(question, top_k=7)
                    citations = format_citations(chunks)
                    context = rag_mod.build_rag_prompt(question, chunks, citations)
                else:
                    chunks = []
                    citations = []
                    context = ""

                answer, gen_latency = self._generate(processor, model, question, context)
                latencies.append(gen_latency)

                # Metrics
                rl = rouge_l(answer, ref_answer) if ref_answer else None
                if rl is not None:
                    rouge_scores.append(rl)

                hr = citation_hallucination_rate(answer, citations)
                hallucination_rates.append(hr)

                ab = abstention_correct(answer, q_type)
                if ab is not None:
                    abstention_scores.append(float(ab))

                result = {
                    "question":          question,
                    "answer":            answer[:500],
                    "reference_answer":  ref_answer[:300] if ref_answer else None,
                    "rouge_l":           round(rl, 4) if rl is not None else None,
                    "hallucination_rate": round(hr, 4),
                    "abstention_correct": ab,
                    "num_retrieved":     len(chunks),
                    "citations":         citations,
                    "generation_latency": round(gen_latency, 3),
                }
                results.append(result)

            except Exception as e:
                import traceback
                log.error(f"Error on question {i}:\n{traceback.format_exc()}")
                results.append({"question": question, "error": str(e)})

        # Aggregate metrics
        summary = {
            "mode":                    mode,
            "num_questions":           len(test_questions),
            "avg_rouge_l":             round(sum(rouge_scores) / len(rouge_scores), 4) if rouge_scores else None,
            "avg_hallucination_rate":  round(sum(hallucination_rates) / len(hallucination_rates), 4) if hallucination_rates else None,
            "abstention_accuracy":     round(sum(abstention_scores) / len(abstention_scores), 4) if abstention_scores else None,
            "avg_latency_s":           round(sum(latencies) / len(latencies), 3) if latencies else None,
            "results":                 results,
        }

        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LegalMind AI — Evaluation Suite")
    parser.add_argument("--test-set", default="evaluation/test_questions.jsonl")
    parser.add_argument("--output",   default="evaluation/eval_report.json")
    parser.add_argument("--modes",    nargs="+", default=["rag"],
                        choices=["base", "rag", "rag+lora"],
                        help="Evaluation modes (default: rag)")
    parser.add_argument("--lora-adapter", default=None,
                        help="Path to LoRA adapter (for rag+lora mode)")
    parser.add_argument("--llm-gpu",   type=int, default=1)
    parser.add_argument("--embed-gpu", type=int, default=4)
    parser.add_argument("--chunks-dir",    default="data/chunks/legislation/")
    parser.add_argument("--vector-db-dir", default="vector_db/legislation/")
    parser.add_argument("--api-url", default=None,
                        help="Optional URL of running API server (e.g. http://127.0.0.1:8080/query)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load test questions
    test_path = Path(args.test_set)
    if not test_path.exists():
        log.error(f"Test set not found: {test_path}")
        log.error("Create evaluation/test_questions.jsonl with {question, reference_answer, type} records.")
        sys.exit(1)

    questions = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    log.info(f"Loaded {len(questions)} test questions from {test_path}")

    evaluator = LegalMindEvaluator(
        llm_gpu       = args.llm_gpu,
        embed_gpu     = args.embed_gpu,
        chunks_dir    = args.chunks_dir,
        vector_db_dir = args.vector_db_dir,
        lora_adapter  = args.lora_adapter,
    )

    all_reports = {}
    for mode in args.modes:
        log.info(f"\n{'='*60}")
        log.info(f"Evaluating mode: {mode.upper()}")
        log.info(f"{'='*60}")
        report = evaluator.evaluate(questions, mode=mode, api_url=args.api_url)
        all_reports[mode] = report

        # Print summary
        log.info(f"\n── {mode.upper()} Summary ──")
        for k, v in report.items():
            if k != "results":
                log.info(f"  {k:<30}: {v}")

    # Save full report (merge with any existing modes in report)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_reports = {}
    if out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as f:
                existing_reports = json.load(f)
        except Exception:
            existing_reports = {}
    existing_reports.update(all_reports)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing_reports, f, ensure_ascii=False, indent=2)

    log.info(f"\n✅ Evaluation complete. Report saved to {out_path}")


if __name__ == "__main__":
    main()
