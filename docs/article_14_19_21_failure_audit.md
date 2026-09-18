# LegalMind AI — Article 14, 19, 21 & Generation Failure Audit Report

**Date**: 2026-09-12  
**Investigation**: Forensic Audit of Constitutional RAG Pipeline Failure on Benchmark Question  
**Target Query**:  
> *"Explain the differences between Article 14, Article 19, and Article 21 of the Constitution of India. Discuss how these rights are connected, their reasonable restrictions, and important Supreme Court judgments. Give the answer in a clear legal order with verified sources."*

---

## 1. Executive Summary of Findings

During execution of the benchmark constitutional query, LegalMind AI returned:
- *"The retrieved legal corpus does not contain sufficient direct evidence..."*
- Rated confidence as `LOW` due to total absence of Part III (Fundamental Rights) articles.
- The user interface exhibited an interrupted generation state with message: `_(interrupted — the server stopped while this answer was being generated)_`.

A systematic forensic audit of the datasets, chunking manifests, vector indices, retrieval algorithms, and generation loops revealed **five distinct failure modes**:

1. **Constitutional Corpus Truncation**:
   The active legislation vector index (`vector_db/legislation/`) and chunk manifest (`data/chunks/legislation/chunks.jsonl`) contain only **48 chunks** attributed to the Constitution of India. These chunks represent fragmented sections of **Part XXI** (Temporary & Transitional Provisions, Arts. 369–371) and **Part XXII** (Short Title, Arts. 393–394) originating from a localized Andaman & Nicobar India Code scrape.
   **Crucially**: The primary statutory texts of **Article 14, Article 19 (clauses 1–6), Article 20, Article 21, Article 21A, Article 22, and Article 226 are completely absent** from `data/chunks/legislation/chunks.jsonl`.

2. **Single-Corpus Isolation (Siloed Case Law Retrieval)**:
   While landmark Supreme Court judgments (*Maneka Gandhi v. Union of India*, *A.K. Gopalan v. State of Madras*, *E.P. Royappa v. State of Tamil Nadu*, *Romesh Thappar v. State of Madras*, *State of West Bengal v. Anwar Ali Sarkar*, and *Olga Tellis v. BMC*) exist in `data/chunks/case_laws/indexed_chunks.jsonl`, the default RAG pipeline (`scripts/05_rag.py`) queried only `CHUNKS_DIR = data/chunks/legislation/`. Consequently, no relevant precedents were retrieved for constitutional grounding.

3. **Multi-Article Retrieval Bias & Domination**:
   Standard top-$k$ dense and BM25 retrieval cannot effectively balance multi-target queries spanning multiple distinct provisions (Articles 14, 19, and 21). Whichever provision has higher token frequency in generic legal commentary displaces the others, preventing simultaneous evidence collection across all three articles.

4. **Generation Interruption & Unhandled Server Shutdowns**:
   When answering broad multi-article inquiries without evidence, the model attempts complex general-knowledge synthesis (taking ~70–80 seconds). In-flight generations interrupted by server restarts or timeouts left orphaned rows in SQLite with generic `_(interrupted...)_` placeholder text, failing to return structured, retryable error responses (`GENERATION_INTERRUPTED`).

5. **Lack of Per-Article Confidence Decomposition**:
   Confidence estimation was holistic rather than decomposing evidence per constitutional provision, preventing granular reporting of which rights were supported by primary evidence.

---

## 2. Dataset & Index Audit

| Corpus / Index Path | Expected Content | Actual Content | Status | Root Cause |
|---------------------|------------------|----------------|--------|------------|
| `data/chunks/legislation/chunks.jsonl` | Complete Indian Statutes + Full Constitution | 15,045 chunks; only 48 Constitution chunks | **INCOMPLETE** | Scraped raw data contained only Part XXI/XXII for Constitution; Fundamental Rights (Part III) omitted. |
| `data/raw/huggingface/Indian_constitution/constitution_dataset.jsonl` | Authentic Diglot Constitution (May 2024) | 871 full-text chunks including Arts 14, 19(1)–(6), 20, 21, 21A, 22, 32, 226 | **AVAILABLE** | Authentic Ministry of Law & Justice text available on disk, but was never ingested into the legislation chunk store. |
| `vector_db/legislation/` | FAISS index for legislation | 15,045 vectors (indexed chunks) | **LACKS PART III** | FAISS vectors mirror incomplete `chunks.jsonl`. |
| `data/chunks/case_laws/indexed_chunks.jsonl` | Landmark SC Judgments | 26,085 SC/HC judgments (includes *Maneka Gandhi*, *Gopalan*, *Royappa*, *Romesh Thappar*, *Anwar Ali*, *Olga Tellis*) | **PRESENT BUT UNLINKED** | Case laws FAISS index was not cross-queried during constitutional RAG execution. |

---

## 3. Case Law Availability Analysis

| Landmark Precedent | Present in Corpus? | Document ID / Citation | Relevance to Golden Triangle / Arts 14, 19, 21 |
|-------------------|-------------------|------------------------|------------------------------------------------|
| *Maneka Gandhi v. Union of India* (1978) | **Yes** (`indexed_chunks.jsonl:14732`) | `1978INSC16` / `[1978] 2 S.C.R. 621` | Overruled *Gopalan*; established interlinking of Arts 14, 19, and 21; introduced "just, fair, and reasonable" procedure. |
| *A.K. Gopalan v. State of Madras* (1950) | **Yes** (`indexed_chunks.jsonl:3676`) | `1950INSC13` / `[1950] 1 S.C.R. 88` | Established the original "exclusive/siloed" doctrine of fundamental rights, later dismantled by *Maneka Gandhi*. |
| *E.P. Royappa v. State of Tamil Nadu* (1973) | **Yes** (`indexed_chunks.jsonl:12569`) | `1973INSC213` / `[1974] 2 S.C.R. 348` | New doctrine of equality; arbitrariness is antithetical to Article 14. |
| *State of West Bengal v. Anwar Ali Sarkar* (1952) | **Yes** (`indexed_chunks.jsonl:3983`) | `1952INSC1` / `[1952] 1 S.C.R. 284` | Reasonable classification test under Article 14; intelligible differentia and rational nexus. |
| *Romesh Thappar v. State of Madras* (1950) | **Yes** (`indexed_chunks.jsonl:3657`) | `1950INSC14` / `[1950] 1 S.C.R. 594` | Freedom of speech and press under Article 19(1)(a); public order limitations. |
| *Bennett Coleman v. Union of India* (1972) | **Yes** (`indexed_chunks.jsonl:10305`) | `1969INSC100` / `[1970] 1 S.C.R. 181` | Direct and inevitable effect test for fundamental freedoms under Article 19(1)(a). |
| *Olga Tellis v. Bombay Municipal Corp.* (1985) | **Yes** (`indexed_chunks.jsonl:17175`) | `1985INSC151` / `[1985] SUPP. 2 S.C.R. 51` | Right to livelihood as an integral facet of the right to life under Article 21. |
| *K.S. Puttaswamy v. Union of India* (2017) | **Documented in Graph** (`precedent_graph.py`) | `(2017) 10 SCC 1` | Right to privacy under Article 21 and the Golden Triangle (post-dates sample case dataset). |
| *Shreya Singhal v. Union of India* (2015) | **Documented in Graph** (`precedent_graph.py`) | `(2015) 5 SCC 1` | Section 66A IT Act struck down under Article 19(1)(a) & 19(2). |

---

## 4. Files Requiring Modifications vs. Intentionally Untouched

### A. Files Requiring Modifications / Creation
1. **`scripts/ingest_constitution_articles.py`** [NEW]:
   - Extract authentic Article 14, 19(1)–(6), 20, 21, 21A, 22, 32, 226 records with complete metadata from `data/raw/huggingface/Indian_constitution/constitution_dataset.jsonl`.
   - Append to `data/chunks/legislation/chunks.jsonl` and update FAISS / BM25 indices.
2. **`scripts/constitutional_retrieval.py`** [NEW]:
   - Multi-article router, Golden Triangle extractor, multi-pass retrieval merging statutory articles and landmark case laws without single-article dominance.
3. **`scripts/05_rag.py`** [MODIFY]:
   - Integrate constitutional router, generation timeout safety (`GENERATION_TIMEOUT_SECONDS = 180`), token accumulation, and 14-section ordered output policy.
4. **`app/chat_api.py`** & **`app/main.py`** [MODIFY]:
   - Guard against abrupt interruptions; commit partial outputs safely with status `"partial"`; emit structured error payload with code `GENERATION_INTERRUPTED`.
5. **`scripts/response_generator.py`** [MODIFY]:
   - Output structured per-article evidence coverage breakdown (`article_14`, `article_19`, `article_21`).
6. **`tests/test_pipeline.py`** [MODIFY]:
   - Expand unit test suite with multi-article retrieval, clause extraction, Golden Triangle routing, and generation stability tests.

### B. Files STRICTLY PRESERVED (Zero Modifications)
1. **`app/static/index.html`** — *Owner: Frontend/History teammate*
2. **`app/static/style.css`** — *Owner: Frontend/History teammate*
3. **`app/static/app.js`** — *Owner: Frontend/History teammate*
4. **`app/index.html`** — *Owner: Frontend/History teammate*
5. **Frontend chat history UI, components, conversation list, and CSS layout** — *100% untouched*.
