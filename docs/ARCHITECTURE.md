# LegalMindAI — System Architecture & Technical Design

## 1. System Overview

LegalMindAI is an enterprise-grade Indian Legal Research, Multimodal Assistant, and Legal Reasoning System engineered for statutory interpretation, constitutional analysis, case law retrieval, and legal document analysis.

```mermaid
flowchart TB
    subgraph Clients["Client Layer"]
        UI["Web UI (macOS Space Black / iOS 18 Glass)"]
        CLI["LegalMind Terminal CLI (Antigravity-style)"]
        API_Ext["External API Consumers / Services"]
    end

    subgraph Gateway["API Gateway & Service Layer (FastAPI)"]
        AppMain["app.main: FastAPI Application (Port 8080)"]
        ChatAPI["app.chat_api: Conversation & RAG Query"]
        MultiModalAPI["app.api: Speech, Images, Documents, Audio"]
        ChatStore["storage.chat_store: SQLite WAL Persistence"]
    end

    subgraph Multimodal["Multimodal Services (app/services)"]
        VisionSvc["vision_service: Legal Image & Stamp QA"]
        DocSvc["document_service: PDF/DOCX/TXT Parser"]
        OCR["ocr_service: Preprocessing & Confusion Matrix"]
        ExtractSvc["legal_extraction_service: Metadata & Act Identification"]
        SpeechSvc["speech_service: Faster-Whisper Voice Typing"]
        TTSSvc["tts_service: Neural Text-to-Speech Engine"]
    end

    subgraph Reasoning["Legal Intelligence & Reasoning Layer (scripts/)"]
        RespGen["response_generator: 12 & 14-Section Synthesizer"]
        IRAC["legal_reasoning: Auditable IRAC Engine"]
        Temporal["temporal_law: IPC/BNS & CrPC/BNSS Registry"]
        Precedent["precedent_graph: Landmark SC Precedent Graph"]
        CitationVerif["citation_verifier: Grounding & Anti-Hallucination"]
    end

    subgraph RAG["Retrieval Engine (scripts/retriever.py)"]
        Router["Query Preprocessing & Intent Classifier"]
        GT_Router["constitutional_retrieval: Golden Triangle Router"]
        BM25["BM25 Lexical Search (CPU In-Memory)"]
        FAISS_Leg["FAISS Dense Index: Legislation & Constitution"]
        FAISS_Case["FAISS Dense Index: Supreme Court Case Laws"]
        RRF["Reciprocal Rank Fusion (RRF k=60)"]
        Reranker["BAAI/bge-reranker-v2-m3 Cross-Encoder"]
    end

    subgraph Hardware["Hardware & GPU Allocation (DGX B200)"]
        GPU1["NVIDIA DGX B200 — GPU 1: Qwen3.6-35B-A3B LLM (Native BF16)"]
        GPU4["NVIDIA DGX B200 — GPU 4: BAAI/bge-m3 & bge-reranker"]
        GPU3["GPU 3: Strictly Excluded (Reserved for System Processes)"]
        CPU_RAM["Host CPU & RAM: SQLite WAL, BM25 Index, Document Parsers"]
    end

    UI --> AppMain
    CLI --> AppMain
    API_Ext --> AppMain

    AppMain --> ChatAPI
    AppMain --> MultiModalAPI
    ChatAPI --> ChatStore

    MultiModalAPI --> VisionSvc
    MultiModalAPI --> DocSvc
    MultiModalAPI --> OCR
    MultiModalAPI --> ExtractSvc
    MultiModalAPI --> SpeechSvc
    MultiModalAPI --> TTSSvc

    ChatAPI --> Router
    Router --> GT_Router
    Router --> BM25
    Router --> FAISS_Leg
    Router --> FAISS_Case
    BM25 --> RRF
    FAISS_Leg --> RRF
    FAISS_Case --> RRF
    RRF --> Reranker

    Reranker --> Reasoning
    Reasoning --> GPU1
    FAISS_Leg --> GPU4
    FAISS_Case --> GPU4
    Reranker --> GPU4
```

---

## 2. Hardware Resource Allocation

The host environment is an 8-GPU **NVIDIA DGX B200** server. Resource allocation is strictly compartmentalized to prevent contention:

| Subsystem | Execution Target | Memory Footprint | Rationale |
|---|---|---|---|
| **Qwen3.6-35B-A3B Base LLM** | **GPU 1** (`cuda:1`) | ~38.4 GB VRAM | High-throughput generation in native `bfloat16` without quantization. |
| **BGE-M3 Dense Embeddings** | **GPU 4** (`cuda:4`) | ~4.2 GB VRAM | Rapid 1024-dim dense encoding of user queries and documents. |
| **BGE-Reranker-v2-m3** | **GPU 4** (`cuda:4`) | ~2.8 GB VRAM | High-precision cross-attention re-scoring of top-K candidate chunks. |
| **GPU 3 Guardrail** | **PROHIBITED** | 0 GB (Unused) | Strictly avoided due to existing processes on the DGX server. |
| **BM25 Lexical Indices** | **Host RAM (CPU)** | ~450 MB RAM | Ultra-fast token matching across 52,000+ indexed legal chunks. |
| **SQLite WAL Chat Storage** | **NVMe SSD (data/)** | In-process | ACID compliance, concurrent read/write isolation. |

---

## 3. Subsystem Architecture

### 3.1 Web UI & Chat Storage
- **Static Frontend**: Located in `app/static/` and `app/index.html`. Features an Apple macOS Space Black aesthetic with fluid iOS 18 glass refractions, Dynamic Island navigation, and zero layout collisions.
- **Persistence Layer**: `data/legalmind.db` operating in **SQLite WAL (Write-Ahead Logging)** mode. Provides user-level conversation isolation, FTS5 full-text message search, and atomic state transitions. Maintained without breaking changes to support parallel frontend development.

### 3.2 Multimodal Processing Layer (`app/services/`)
- **Document Analysis (`document_service.py`)**: Asynchronous multi-format document parser supporting PDF, DOCX, TXT, and scanned image extractions.
- **Legal Extraction (`legal_extraction_service.py`)**: Regex and heuristic extractor identifying statutory acts, sections, courts, case citations, and document types.
- **Vision QA (`vision_service.py`)**: Analyzes document layout, court seals, signatures, and stamps.
- **OCR Service (`ocr_service.py`)**: Features non-destructive optical character confusion correction (e.g. correcting `Sec. 3O2` to `Sec. 302`, `Artic1e` to `Article`).
- **Speech Services (`speech_service.py`, `tts_service.py`)**: Local voice typing transcription and neural speech output.

### 3.3 Hybrid RAG & Query Routing Layer (`scripts/retriever.py`)
- **Intent Detection**: Directs constitutional queries to the specialized Golden Triangle engine (`scripts/constitutional_retrieval.py`).
- **Dual-Index Search**: Simultaneously queries `vector_db/legislation/` (acts and constitution) and `vector_db/case_laws/` (Supreme Court judgments).
- **Reciprocal Rank Fusion (RRF)**: Merges dense vector results ($S_{\text{dense}}$) with BM25 lexical results ($S_{\text{bm25}}$) using constant $k=60$.
- **Cross-Encoder Re-Ranking**: Top-30 candidate chunks are re-scored using `BAAI/bge-reranker-v2-m3` on GPU 4.

### 3.4 Structured Legal Reasoning & Generation Engine
- **14-Section Synthesis (`scripts/response_generator.py`)**: Generates exhaustive legal responses structured into 14 ordered sections for constitutional and landmark inquiries.
- **12-Section Synthesis**: Generates structured responses tailored to statutory compliance inquiries.
- **Evidence-Grounded IRAC (`scripts/legal_reasoning.py`)**: Emits structured, auditable IRAC records while strictly suppressing raw model `<think>` tokens.
- **Temporal Statutory Engine (`scripts/temporal_law.py`)**: Maps historical statutes (IPC, CrPC, IEA) to modern replacements (BNS, BNSS, BSA) with automatic in-force flags and warnings.
- **Precedent Relationship Graph (`scripts/precedent_graph.py`)**: Models citations, overruled cases, distinguished decisions, and followed ratios.
- **Citation Verification (`app/services/citation_verifier.py`)**: Verifies all cited authorities against retrieved evidence chunks to protect against hallucinations.
