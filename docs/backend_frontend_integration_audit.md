# LegalMind AI — Backend & Frontend Integration Audit

**Date:** 2026-09-12  
**System:** LegalMind AI (DGX Platform, Qwen3.6-35B-A3B MoE, Hybrid RAG, SQLite Chat Store)  
**Document Purpose:** Comprehensive audit of backend, frontend, chat history, retrieval, model, training, and evaluation subsystems to establish safe integration boundaries and strict division of ownership.

---

## 1. Executive Summary & Team Ownership Division

To prevent regression, file collision, and duplication of work during parallel development, the ownership boundaries are strictly partitioned:

### Frontend & UI Ownership (Friend's Domain — DO NOT MODIFY)
The frontend developer has sole ownership of:
- **Frontend UI & Layout**: `app/static/index.html`, `app/index.html`
- **UI Styling & Design Tokens**: `app/static/style.css`
- **Client Application & State Management**: `app/static/app.js`
- **Chat Interface & Message Rendering**: Message templates, loading states, markdown parsing, sources accordion.
- **Sidebar & Conversation List UI**: Future sidebar, conversation drawer, history navigation.
- **History Interface**: Chat history list, search bar UI, export modal UI.

*Strict Rule for Backend*: DO NOT redesign, rewrite, replace, or delete existing UI files, styles, or frontend components. All backend API changes must remain 100% backward-compatible with `app/static/app.js`.

### Backend & AI Pipeline Ownership (Your Domain)
The backend developer owns:
- **API Contracts & REST Endpoints**: `app/main.py`
- **Model Loading & Inference**: `scripts/05_rag.py` (`LegalMindModel`)
- **Retrieval & Reranking Subsystem**: `scripts/retriever.py` (`HybridRetriever`, `BM25Index`, FAISS indices, Cross-Encoder)
- **Data Ingestion, Preprocessing & Chunking**: `scripts/01_ingest.py`, `scripts/01b_preprocess_legislation.py`, `scripts/01c_preprocess_caselaws.py`, `scripts/02_chunk.py`, `scripts/03_embed.py`, `scripts/04_build_faiss.py`
- **Chat Persistence Backend & History Store**: `storage/chat_store.py`, `storage/exporters.py`
- **Structured Legal Reasoning, Grounding & Disclaimers**: Verification logic in `scripts/05_rag.py`
- **Temporal Law Support & Precedent Relationship Graph**: Backend models and metadata extractors
- **Standard LoRA Fine-Tuning & Adapter Validation**: `fine_tuning/train_lora.py`, `fine_tuning/configs/lora_config.yaml`
- **Automated Tests & Evaluation Benchmark**: `tests/test_pipeline.py`, `fine_tuning/evaluate.py`, `evaluation/eval_retrieval.py`
- **Integration Contracts & Documentation**: `docs/`

---

## 2. Project File Structure & Classification

```
/home/sece2026-student12/LegalMindAI/
├── app/
│   ├── main.py                  [BACKEND] FastAPI entry point, routes, model/retriever singletons
│   ├── index.html               [FRONTEND - DO NOT MODIFY] Standalone client template
│   └── static/
│       ├── app.js               [FRONTEND - DO NOT MODIFY] Chat UI logic, fetch('/query')
│       ├── index.html           [FRONTEND - DO NOT MODIFY] Active served web app template
│       └── style.css            [FRONTEND - DO NOT MODIFY] CSS design tokens & animations
├── storage/
│   ├── __init__.py              [BACKEND] Storage package init
│   ├── chat_store.py            [HISTORY BACKEND] SQLite WAL persistence, FTS5 search, migrations
│   └── exporters.py             [HISTORY BACKEND] Markdown & plaintext export utilities
├── scripts/
│   ├── 01_ingest.py             [BACKEND] Raw text & PDF ingestion
│   ├── 01b_preprocess_legislation.py [BACKEND] Central Acts & Constitution cleaner
│   ├── 01c_preprocess_caselaws.py    [BACKEND] High Court & Supreme Court cleaner
│   ├── 02_chunk.py              [BACKEND] Provision-level legal chunker
│   ├── 03_embed.py              [BACKEND] BGE-M3 dense vector generator
│   ├── 04_build_faiss.py        [BACKEND] FAISS IndexIVFFlat constructor
│   ├── 05_rag.py                [BACKEND] Base LLM inference, RAG prompt, grounding & confidence
│   ├── retriever.py             [BACKEND] HybridRetriever (BM25 + FAISS + RRF + Cross-Encoder)
│   ├── patch_chunk_metadata.py  [BACKEND] Metadata validator
│   └── rank_bm25.py             [BACKEND] Standalone BM25Okapi implementation
├── fine_tuning/
│   ├── configs/lora_config.yaml [BACKEND] LoRA hyperparameter configuration
│   ├── datasets/                [BACKEND] SFT training and evaluation JSONL files
│   ├── adapters/lora-legal-v1/  [BACKEND] Saved LoRA adapter weights (audited for NaNs)
│   ├── train_lora.py            [BACKEND] Standard LoRA fine-tuning script
│   └── evaluate.py              [BACKEND] Offline evaluation suite (ROUGE, hallucination, abstention)
├── evaluation/
│   ├── eval_retrieval.py        [BACKEND] Hit@K and MRR retrieval benchmark
│   ├── eval_report.json         [BACKEND] Quantitative generation evaluation
│   ├── final_evaluation_report.md [BACKEND] Comprehensive system audit report
│   └── test_questions.jsonl     [BACKEND] Indian legal benchmark test questions
├── tests/
│   ├── test_pipeline.py         [BACKEND] 13 automated unit tests
│   └── run_tests_fast.py        [BACKEND] Rapid CLI runner for test_pipeline
├── models/
│   └── Qwen3.6-35B-A3B/         [MODEL] Local base weights (BF16, MoE)
├── vector_db/
│   ├── legislation/             [RETRIEVAL] FAISS index & chunk IDs (26,085 vectors)
│   └── case_laws/               [RETRIEVAL] FAISS index & chunk IDs (26,085 vectors)
└── docs/
    └── backend_frontend_integration_audit.md [THIS DOCUMENT]
```

---

## 3. Backend & Frontend Entry Points

### Backend Entry Point
- **File**: `app/main.py`
- **Execution Command**:
  ```bash
  /home/sece2026-student12/LegalMindAI/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
  ```
- **Lifecycle (`lifespan`)**:
  - Initializes `HybridRetriever` on GPU 4 (`CHUNKS_DIR="data/chunks/legislation/"`, `VECTOR_DB_DIR="vector_db/legislation/"`, `EMBED_MODEL="BAAI/bge-m3"`, `RERANK_MODEL="BAAI/bge-reranker-v2-m3"`).
  - Loads `LegalMindModel` singleton on GPU 1 (`models/Qwen3.6-35B-A3B`, BF16, PyTorch native).
  - Keeps model and retriever in memory across requests (`workers=1` enforced for CUDA tensor safety).

### Frontend Entry Point
- **File**: `app/static/index.html` (served via `GET /` in `app/main.py`)
- **Assets**: Mounted statically at `/static` -> `app/static/`
- **Client Script**: `app/static/app.js` runs on `DOMContentLoaded`, binds to `#query-form`, sends JSON to `/query`, and renders responses dynamically.

---

## 4. Existing API Routes & Contracts

### Route 1: `GET /health`
- **Purpose**: Liveness & readiness probe.
- **Request**: None.
- **Response**:
  ```json
  {
    "status": "ok",
    "model_loaded": true,
    "retriever_loaded": true,
    "uptime_s": 124.5,
    "queries_served": 28
  }
  ```

### Route 2: `GET /stats`
- **Purpose**: System telemetry & GPU memory status.
- **Request**: None.
- **Response**:
  ```json
  {
    "model_path": "/home/sece2026-student12/LegalMindAI/models/Qwen3.6-35B-A3B",
    "llm_gpu": 1,
    "embed_gpu": 4,
    "top_k": 7,
    "embed_model": "BAAI/bge-m3",
    "rerank_model": "BAAI/bge-reranker-v2-m3",
    "model_loaded": true,
    "retriever_loaded": true,
    "uptime_s": 124.5,
    "queries_served": 28,
    "gpu_info": [
      {
        "index": 1,
        "name": "NVIDIA B200",
        "total_gb": 179.1,
        "free_gb": 108.4,
        "used_gb": 70.7
      }
    ]
  }
  ```

### Route 3: `POST /query`
- **Purpose**: Main RAG question-answering pipeline.
- **Current Request Schema (`QueryRequest`)**:
  ```json
  {
    "query": "string (required)",
    "top_k": 7
  }
  ```
- **Current Response Schema (`QueryResponse`)**:
  ```json
  {
    "query": "What are the requirements of Article 21?",
    "answer": "## Answer\n...",
    "legal_provisions": ["Article 21, Constitution of India"],
    "judgments": ["Maneka Gandhi v. Union of India (1978)"],
    "legal_reasoning": "Step-by-step legal analysis...",
    "citations": ["[1] Constitution of India — Article 21"],
    "confidence": "HIGH — Directly supported by retrieved text.",
    "limitations": "None.",
    "retrieved_documents": [
      {
        "citation_id": 1,
        "document_id": "doc_c0000_e514b064",
        "document_type": "constitution",
        "title": "Constitution of India",
        "source": "constitution.pdf",
        "court": null,
        "date": null,
        "citation": null,
        "article": "21",
        "section": "13",
        "rrf_score": 0.033,
        "rerank_score": 0.942,
        "text_preview": "No person shall be deprived of his life or personal liberty..."
      }
    ],
    "retrieval_latency": 0.134,
    "generation_latency": 3.842,
    "total_latency": 3.976
  }
  ```

---

## 5. Existing Chat History & Persistence Implementation

The project already contains an implementation of SQLite chat persistence in `storage/chat_store.py`:

- **Database Engine**: `sqlite3` standard library with WAL mode (`PRAGMA journal_mode=WAL`), foreign key enforcement (`PRAGMA foreign_keys=ON`), and re-entrant thread-safe write locks (`_write_lock = threading.RLock()`).
- **Database Location**: `$LEGALMIND_DB_PATH`, defaulting to `./data/legalmind.db`.
- **Database Schema**:
  1. `conversations`:
     - `id`: `TEXT PRIMARY KEY` (UUID4)
     - `user_id`: `TEXT NOT NULL`
     - `title`: `TEXT NOT NULL`
     - `created_at`: `TEXT NOT NULL` (ISO8601 UTC)
     - `updated_at`: `TEXT NOT NULL` (ISO8601 UTC)
     - `pinned`: `INTEGER NOT NULL DEFAULT 0`
     - `deleted_at`: `TEXT` (soft-delete timestamp; `NULL` when active)
     - `model_name`: `TEXT`
  2. `messages`:
     - `id`: `TEXT PRIMARY KEY` (UUID4)
     - `conversation_id`: `TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE`
     - `role`: `TEXT NOT NULL CHECK(role IN ('user','assistant','system'))`
     - `content`: `TEXT NOT NULL`
     - `created_at`: `TEXT NOT NULL`
     - `token_count`: `INTEGER`
     - `latency_ms`: `INTEGER`
     - `status`: `TEXT NOT NULL DEFAULT 'complete' CHECK(status IN ('complete','partial','error'))`
     - `citations`: `TEXT` (JSON array)
     - `feedback`: `INTEGER` (-1, 0, +1)
     - `client_token`: `TEXT UNIQUE` (idempotency guard preventing duplicate double-submits)
     - `metadata`: `TEXT` (JSON string)
  3. `messages_fts`:
     - FTS5 external-content virtual table with automatic insert/update/delete triggers.
  4. `schema_version`:
     - Tracks applied forward migrations.

- **Existing Storage Functions**:
  - Conversations: `create_conversation()`, `get_conversation()`, `list_conversations()`, `rename_conversation()`, `set_pinned()`, `soft_delete_conversation()`, `restore_conversation()`, `purge_conversation()`.
  - Messages: `append_message()`, `get_messages()`, `get_message()`, `get_message_by_client_token()`, `update_message()`, `set_feedback()`, `count_messages()`.
  - Full-Text Search: `search_messages(user_id, query)`.
  - Exporters (`storage/exporters.py`): `to_markdown()`, `export_filename()`, `collect_authorities()`.

*Status*: The storage backend is implemented and tested, but its REST API routes are not yet mounted on FastAPI (`app/main.py`). The chat UI currently talks to `/query` without persisting sessions. Exposing clean, RESTful history endpoints without touching the frontend code is the key integration requirement.

---

## 6. Model, Retrieval & RAG Pipeline Audit

1. **Model Loader (`LegalMindModel`)**:
   - Singleton pattern ensuring 35B model stays resident in GPU 1 memory.
   - `enable_thinking=False` suppresses Qwen3 chain-of-thought tokens.
   - `strip_thinking()` removes any residual `<think>...</think>` markup.
   - Zero repetition penalty (prevents degeneration loops).

2. **Hybrid Retrieval (`HybridRetriever`)**:
   - Dense FAISS search: Inner-product inverted index (`IndexIVFFlat`) with BGE-M3 (1024-dim L2-normalized).
   - BM25 search: Tokenizes legal provisions, headings, and keywords, backed by `_SimpleBM25Okapi` pure-Python fallback.
   - Exact Provision Matcher (`find_exact_provision_matches`): Extracted target provision matcher (e.g. `Article 21`, `Section 302`, `Section 45 PMLA`) boosting exact clause hits.
   - Reciprocal Rank Fusion (RRF, $k=60$): Combines dense and sparse ranks.
   - Cross-Encoder Reranker (`bge-reranker-v2-m3`): Rescores top-K candidates on GPU 4.

3. **Grounding & Evidence Constraints**:
   - Direct provision check ensures unretrieved statutory clauses are never claimed as evidence-backed.
   - Broad constitutional queries format responses into 10 canonical areas (Preamble, Fundamental Rights, DPSPs, Duties, Union/State Govt, Judiciary, Federalism, Constitutional Bodies, Emergency, Amendments).
   - Missing constitutional areas output: `"The retrieved evidence does not cover this topic."`
   - Confidence downgrading: Prevents `HIGH` confidence whenever evidence is absent or incomplete.

---

## 7. Fine-Tuning Pipeline Audit & LoRA NaN Finding

- **Configuration File**: `fine_tuning/configs/lora_config.yaml`
- **Training Script**: `fine_tuning/train_lora.py` (TRL `SFTTrainer` with PEFT LoRA)
- **Datasets**:
  - `fine_tuning/datasets/legal_sft_train.jsonl` (2,160 examples)
  - `fine_tuning/datasets/legal_sft_eval.jsonl` (540 examples)
- **Previous Adapter Finding**:
  - `fine_tuning/adapters/lora-legal-v1/adapter_model.safetensors` contains `NaN` values resulting from earlier training at an excessively high learning rate ($2 \times 10^{-4}$) and gradient clipping of $1.0$.
  - When loaded, it produces all-NaN hidden states and blank outputs.
  - The production pipeline currently falls back to Base Qwen3.6-35B + Hybrid RAG, which achieves 0% hallucination and 100% test pass rate.
- **Safety Settings Required for Standard LoRA**:
  - `learning_rate`: `2.0e-5` (conservative)
  - `num_train_epochs`: `1`
  - `max_grad_norm`: `0.3`
  - `warmup_ratio`: `0.05`
  - `gradient_checkpointing`: `true`
  - `gradient_accumulation_steps`: `8`
  - Precision: `bf16 = true`, `fp16 = false`
  - Zero 4-bit / 8-bit quantization (strictly forbidden).
  - Validation: Every tensor must be checked with `torch.isnan().any()` and `torch.isinf().any()` prior to deployment.

---

## 8. Test Suite Status

The automated test suite in `tests/test_pipeline.py` currently contains **13 passing unit tests (100% pass rate)**:
1. `test_article_21_direct_provision_retrieval`: Exact statutory provision matching.
2. `test_bm25_search`: BM25 tokenization and scoring.
3. `test_broad_constitutional_unsupported_topics_not_claimed_as_backed`: 10-topic organization and unsupported notice.
4. `test_case_laws_faiss_index_integrity`: Case laws FAISS index vector count and file size.
5. `test_format_citations_judgment`: AIR/SCR judgment citation formatting.
6. `test_format_citations_legislation`: Constitution and Act provision citation formatting.
7. `test_metrics_abstention_correct`: Correct abstention on unanswerable queries.
8. `test_metrics_hallucination_rate`: Traceability of `[N]` references.
9. `test_metrics_rouge_l`: ROUGE-L LCS calculation.
10. `test_missing_evidence_not_high_confidence`: Strict confidence downgrade on missing evidence.
11. `test_parse_structured_answer`: Markdown section parser.
12. `test_query_preprocessing`: Legal abbreviation expansion.
13. `test_strip_thinking`: Suppression of `<think>` reasoning tokens.

---

## 9. Recommended Backend Integration Strategy

To fulfill all requirements while strictly honoring team constraints:

1. **Preserve `POST /query` Backward Compatibility**:
   - Extend `QueryRequest` with optional `conversation_id`, `message_id`, `client_token`.
   - Extend `QueryResponse` with new fields (`conversation_id`, `message_id`, `answer_format`, `confidence_label`, `sources`, `legal_warnings`, `temporal_status`, `jurisdiction`, `reasoning_summary`, `related_cases`, `related_sections`, `created_at`, `model`) while preserving all 12 existing fields used by `app/static/app.js`.
2. **Mount RESTful Chat History Endpoints on FastAPI**:
   - Expose endpoints connecting directly to `storage/chat_store.py`:
     - `POST /conversations`: Create conversation
     - `GET /conversations`: List conversations (with `limit`, `offset`)
     - `GET /conversations/{id}`: Get conversation details and message history
     - `PATCH /conversations/{id}`: Rename or pin conversation
     - `DELETE /conversations/{id}`: Soft delete conversation
     - `GET /conversations/{id}/export`: Export Markdown/Plaintext
     - `POST /messages/{id}/feedback`: Submit user feedback (-1, 0, 1)
     - `GET /conversations/search`: Full-text search
3. **Structured Legal Reasoning Summary (IRAC-Style)**:
   - Provide verifiable structured summaries (Issue, Rule/Provision, Facts, Application, Exceptions, Conclusion, Sources) based purely on retrieved evidence.
4. **Temporal Law & Precedent Graph Backend Modules**:
   - Implement metadata classifiers detecting active vs. repealed statutes, amendment counts, and citation relationship graph extractors.
5. **Configurable Response Length Engine**:
   - Support detailed legal explanations up to 10,000–12,000 characters for complex queries while preserving concise, grounded answers for simple queries or insufficient evidence.
6. **Standard LoRA Verification & Training Script Repair**:
   - Update `fine_tuning/configs/lora_config.yaml` with safety settings (LR $2\times 10^{-5}$, max grad norm $0.3$, BF16, Standard LoRA).
   - Implement tensor NaN/Inf check utility.
