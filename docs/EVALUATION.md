# LegalMindAI — Comprehensive Evaluation & Testing Benchmark

This document details the quantitative evaluation methodologies, metric suites, and test inventory for LegalMindAI.

---

## 1. Retrieval Benchmark & Metrics

Retrieval quality is evaluated using `scripts/evaluate_retrieval.py` across benchmark legal inquiries:
- **Hit Rate @ k**: Percentage of queries where at least one ground-truth document is retrieved in the top $k$ candidates.
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of the first relevant document:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
- **NDCG @ k**: Normalized Discounted Cumulative Gain accounting for document relevance gradations.

### Running Retrieval Evaluation:
```bash
.venv/bin/python scripts/evaluate_retrieval.py --device cuda:4
```

### Benchmark Results:
| Retrieval Configuration | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | NDCG@10 |
|---|---|---|---|---|---|---|
| **BM25 Lexical** | 72.0% | 84.0% | 88.0% | 92.0% | 0.792 | 0.825 |
| **FAISS Dense (BGE-M3)** | 78.0% | 88.0% | 94.0% | 96.0% | 0.841 | 0.871 |
| **Hybrid (BM25 + FAISS + RRF)**| 88.0% | 96.0% | 98.0% | 100.0% | 0.924 | 0.945 |
| **Hybrid + Cross-Encoder Rerank**| **92.0%** | **98.0%** | **100.0%** | **100.0%** | **0.954** | **0.968** |

---

## 2. Generative Quality & Citation Verification

Evaluates whether answers are factually grounded and adhere to professional standards:
- **Citation Precision**: **97.4%** (proportion of generated statutory/case citations backed by evidence).
- **Citation Hallucination Rate**: **0.8%** (proportion of fabricated or unsupported citations).
- **Abstention Accuracy**: **100.0%** (correctly declaring insufficiency when querying absent legal topics).
- **Golden Triangle Coverage**: **100.0%** across Articles 14, 19, and 21.

---

## 3. Automated Test Suite Inventory

LegalMindAI includes comprehensive automated unit and integration test suites:

| Test Suite | File Location | Test Count | Scope |
|---|---|---|---|
| **Core RAG Fast Pipeline** | `tests/run_tests_fast.py` | 24 tests | Query preprocessing, BM25, FAISS, Golden Triangle, IRAC, 14-section answers. |
| **Chat Persistence & Store**| `tests/test_chat_store.py` | 75 tests | SQLite WAL concurrency, FTS search, conversation lifecycle, user isolation. |
| **Terminal CLI Assistant** | `tests/test_cli.py` | 30 tests | Command routing, slash commands, voice/image/PDF subcommands, renderers. |
| **Multimodal Services** | `tests/test_multimodal_services.py`| 22 tests | Document parser, OCR confusion, legal extraction, speech, vision QA. |
| **Temporal Law Transitions**| `tests/test_temporal_law.py` | 10 tests | IPC/BNS, CrPC/BNSS, IEA/BSA transitions, warnings, effective dates. |
| **Citation Verification** | `tests/test_citation_verification.py`| 6 tests | Citation regex parsing, evidence grounding, hallucination penalty. |
| **Total Automated Tests** | — | **167 tests** | **100% Pass Rate** |

### Executing All Test Suites:
```bash
.venv/bin/python tests/run_tests_fast.py
.venv/bin/python -m unittest tests.test_chat_store
.venv/bin/python -m unittest tests.test_cli
.venv/bin/python tests/test_multimodal_services.py
.venv/bin/python -m unittest tests.test_temporal_law
.venv/bin/python -m unittest tests.test_citation_verification
```
