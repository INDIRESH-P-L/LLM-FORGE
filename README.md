# ⚖️ LegalMind AI — Indian Legal Intelligence & Multimodal Assistant

**LegalMind AI** is an enterprise-grade Indian Legal Research, Multimodal Assistant, and Structured Legal Reasoning System powered by **Qwen3.6-35B-A3B** (MoE, 35B total / 3B active parameters, native BF16) running on the **NVIDIA DGX B200** platform.

The system performs authoritative statutory interpretation, constitutional analysis (including Golden Triangle Articles 14, 19, and 21), precedent relationship graph traversal, legal document analysis (PDF/DOCX/Images), interactive voice typing, and rigorous citation verification.

---

## Key Highlights

- **Dual-Index Hybrid Retrieval**: Combines Okapi BM25 lexical search with BGE-M3 dense vector retrieval via Reciprocal Rank Fusion (RRF, $k=60$) and cross-encoder re-ranking.
- **Constitutional Golden Triangle Specialization**: Deterministic, balanced statutory retrieval across Articles 14, 19, and 21, synthesized into structured 14-section legal opinions.
- **Temporal Statutory Law Engine**: Automatically tracks the 1 July 2024 repeal of the Indian Penal Code, CrPC, and Evidence Act, mapping historical sections to the Bharatiya Nyaya Sanhita (BNS), BNSS, and BSA with statutory warnings.
- **Precedent Relationship Graph**: Curated landmark Supreme Court knowledge graph supporting queries for overruled cases, distinguished decisions, and followed ratios.
- **Citation Verification & Anti-Hallucination Guard**: Regex extraction and evidence-grounding engine that computes hallucination penalties and protects against fabricated citations.
- **Multimodal Document Intelligence**: Asynchronous PDF/DOCX parsing, OCR confusion matrix correction, court seal/stamp recognition, and local Whisper voice typing.
- **Antigravity-Style Terminal CLI**: Full-screen developer assistant (`bin/legalmind`) featuring streaming tokens, slash commands (`/voice`, `/upload`, `/ocr`, `/speak`), and real-time citation rendering.
- **Apple Space Black Web UI**: Minimalist macOS Space Black canvas with fluid iOS 18 glass refractions, Dynamic Island navigation, and zero layout collisions.
- **Persistence & User Isolation**: High-concurrency SQLite WAL storage (`data/legalmind.db`) with FTS5 message search.
- **100% Test Stability**: 167 automated unit and integration tests passing across all components.

---

## Hardware Resource Allocation (NVIDIA DGX B200)

| Component | Target Device | Memory Footprint | Rationale |
|---|---|---|---|
| **Qwen3.6-35B-A3B LLM** | **GPU 1** (`cuda:1`) | ~38.4 GB VRAM | High-throughput generation in native `bfloat16` (no quantization). |
| **BGE-M3 Dense Embeddings** | **GPU 4** (`cuda:4`) | ~4.2 GB VRAM | Rapid 1024-dim dense encoding of queries and documents. |
| **BGE-Reranker-v2-m3** | **GPU 4** (`cuda:4`) | ~2.8 GB VRAM | High-precision cross-attention re-scoring of candidate chunks. |
| **GPU 3 Guardrail** | **PROHIBITED** | 0 GB (Unused) | Strictly avoided due to external host processes. |
| **BM25 & SQLite WAL** | **Host CPU / NVMe** | ~450 MB RAM | Ultra-fast in-memory lexical matching and atomic persistence. |

---

## Quickstart Guide

### 1. Environment Activation:
```bash
cd ~/LegalMindAI
source .venv/bin/activate
```

### 2. Launch the Backend Server:
```bash
export CUDA_VISIBLE_DEVICES=1,4
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
```

### 3. Launch the Antigravity Terminal CLI:
```bash
bin/legalmind
# or one-shot query:
bin/legalmind ask "What is the Golden Triangle under the Constitution of India?"
```

### 4. Access the Web UI:
Open any web browser at:
```
http://localhost:8080/
# or host IP
http://192.168.4.99:8080/
```

---

## Comprehensive Documentation Sitemap

| Document | Purpose |
|---|---|
| [System Architecture](docs/ARCHITECTURE.md) | End-to-end architecture, data flows, GPU allocation, subsystem boundaries. |
| [Installation & Setup](docs/INSTALLATION.md) | Step-by-step setup, dependencies, GPU verification, verification tests. |
| [Dataset Specifications](docs/DATASETS.md) | Manifest and schemas for all 10 primary and secondary legal datasets. |
| [Data Pipeline](docs/DATA_PIPELINE.md) | Ingestion, statutory boundary chunking, BGE-M3 embedding, and FAISS indexing. |
| [RAG Pipeline](docs/RAG_PIPELINE.md) | BM25, FAISS, Reciprocal Rank Fusion, cross-encoder reranking, and metrics. |
| [Legal Safety & Guardrails](docs/LEGAL_SAFETY.md) | Anti-hallucination guardrails, boundary delimiters, disclaimers, `<think>` filtering. |
| [REST API Reference](docs/API_REFERENCE.md) | Endpoints, JSON schemas, curl examples for query, multimodal, and intelligence. |
| [CLI Reference](docs/CLI_REFERENCE.md) | Antigravity-style CLI commands, subcommands, flags, and interactive shortcuts. |
| [Multimodal Features](docs/MULTIMODAL_FEATURES.md) | Document parsing, OCR confusion correction, vision QA, and voice typing. |
| [Troubleshooting Guide](docs/TROUBLESHOOTING.md) | GPU initialization, adapter NaN fallback, port conflicts, index parity. |
| [Safe LoRA Fine-Tuning](docs/TRAINING.md) | Strict standard LoRA protocols, hyperparameters, and checkpoint validation. |
| [Evaluation Benchmark](docs/EVALUATION.md) | Quantitative IR metrics (Hit@k, MRR, NDCG), citation benchmarks, test suites. |
| [Production Deployment](docs/DEPLOYMENT.md) | Server management, zero-downtime restarts, health monitoring, rollbacks. |
| [Known Limitations](docs/KNOWN_LIMITATIONS.md) | Statutory boundaries, regional state enactments, and regulatory notices. |
| [Multimodal API Contract](docs/multimodal_api_contract.md) | API schemas and specifications for UI and conversation history integration. |
| [System Audit Report](docs/FINAL_SYSTEM_AUDIT.md) | Exhaustive 6-category audit of all repository components. |
| [Hallucination Report](evaluation/hallucination_report.md) | Formal quantitative analysis of citation accuracy and hallucination defense. |
| [LoRA Comparison Report](evaluation/lora_comparison_report.md) | Base Qwen3.6-35B + RAG vs. LoRA adapter comparative evaluation. |
| [Final Evaluation Report](evaluation/final_evaluation_report.md) | Consolidated system verification report. |

---

## Verification Test Commands

Run the full automated test suite (167 test cases, 100% passing):
```bash
.venv/bin/python tests/run_tests_fast.py
.venv/bin/python -m unittest tests.test_chat_store
.venv/bin/python -m unittest tests.test_cli
.venv/bin/python tests/test_multimodal_services.py
.venv/bin/python -m unittest tests.test_temporal_law
.venv/bin/python -m unittest tests.test_citation_verification
```
