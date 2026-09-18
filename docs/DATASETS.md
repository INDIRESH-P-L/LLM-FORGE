# LegalMindAI — Dataset Specifications & Legal Corpus Manifest

This document outlines the complete dataset architecture and storage specifications of LegalMindAI, tracking all 10 core datasets across primary legal authorities and secondary reasoning datasets.

---

## 1. Primary Legal Datasets

### 1. Constitution of India
- **Identifier**: `ds-01-constitution`
- **Source**: Legislative Department, Ministry of Law and Justice, Government of India (Official Gazette).
- **Storage Location**: `data/processed/constitution_articles.jsonl`
- **Total Provisions**: 395 Articles, Preamble, and 12 Schedules.
- **Specialized Chunks**: Includes high-priority Golden Triangle provisions (Articles 14, 19, 21, 32, 226) segmented at sub-clause granularity for fine-grained retrieval.
- **Indexed Vector Path**: `vector_db/legislation/`

### 2. Central Legislation (India Code)
- **Identifier**: `ds-02-central-acts`
- **Source**: India Code digital repository (`indiacode.nic.in`) and Open India Law parquet corpus (`data/raw/open-india-law/`).
- **Storage Location**: `data/chunks/legislation/chunks.jsonl` (26,097 chunks).
- **Core Coverage**:
  - Indian Penal Code, 1860 (IPC)
  - Code of Criminal Procedure, 1973 (CrPC)
  - Indian Evidence Act, 1872 (IEA)
  - Code of Civil Procedure, 1908 (CPC)
  - Indian Contract Act, 1872 (ICA)
  - Specific Relief Act, Transfer of Property Act, Limitation Act, 1963.
- **Indexed Vector Path**: `vector_db/legislation/index.faiss`

### 3. Modern Criminal Sanhitas (2023 Enactments)
- **Identifier**: `ds-03-modern-sanhitas`
- **Source**: Gazette of India Extraordinary Notifications (Ministry of Home Affairs).
- **Storage Location**: Embedded in `data/chunks/legislation/chunks.jsonl` and `scripts/temporal_law.py`.
- **Core Coverage**:
  - Bharatiya Nyaya Sanhita, 2023 (BNS) [Replaced IPC on 2024-07-01]
  - Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS) [Replaced CrPC on 2024-07-01]
  - Bharatiya Sakshya Adhiniyam, 2023 (BSA) [Replaced IEA on 2024-07-01]
- **Section Mappings**: 50+ cross-code section equivalence mappings.

### 4. Supreme Court of India Judgments (Landmark Constitution Benches)
- **Identifier**: `ds-04-sc-judgments`
- **Source**: Supreme Court of India official records, SCR, SCC, AIR reporters.
- **Storage Location**: `data/chunks/case_laws/indexed_chunks.jsonl` (26,085 chunks).
- **Core Coverage**:
  - *Kesavananda Bharati v. State of Kerala* (1973) — Basic Structure
  - *Maneka Gandhi v. Union of India* (1978) — Golden Triangle & Due Process
  - *Justice K.S. Puttaswamy v. Union of India* (2017) — Right to Privacy
  - *Navtej Singh Johar v. Union of India* (2018) — Decriminalization of Sec. 377
  - *Joseph Shine v. Union of India* (2018) — Adultery Decriminalization
- **Indexed Vector Path**: `vector_db/case_laws/index.faiss`

### 5. High Court Judgments & Appellate Jurisprudence
- **Identifier**: `ds-05-hc-judgments`
- **Source**: Delhi, Bombay, Madras, and Calcutta High Court judgments.
- **Storage Location**: `data/chunks/case_laws/indexed_chunks.jsonl`
- **Usage**: Provides persuasive authority, commercial dispute precedents, and writ jurisprudence.

---

## 2. Secondary & Machine Learning Datasets

### 6. SFT Legal Reasoning & Instruction-Tuning Dataset
- **Identifier**: `ds-06-sft-reasoning`
- **Source**: Curated Indian legal reasoning demonstrations.
- **Storage Location**: `fine_tuning/datasets/sft_train.jsonl` (4,800 pairs), `sft_val.jsonl` (600 pairs).
- **Task Types**: Statutory interpretation, IRAC synthesis, issue extraction, citation validation, abstention formulation.

### 7. Indian Legal Knowledge Base (ILKB) Precedents Graph
- **Identifier**: `ds-07-ilkb-graph`
- **Storage Location**: `scripts/precedent_graph.py`
- **Graph Topology**: Directed acyclic graph of Supreme Court decisions with typed edges (`overruled`, `followed`, `distinguished`, `relied_upon`).

### 8. Legal Retrieval Pairs (Evaluation Benchmark)
- **Identifier**: `ds-08-retrieval-benchmark`
- **Storage Location**: `evaluation/test_questions.jsonl`
- **Content**: Answerable and unanswerable legal queries with ground-truth document IDs for computing Hit@K, MRR, and NDCG.

### 9. Citation Reference Grammar & Gazette Mappings
- **Identifier**: `ds-09-citation-grammar`
- **Storage Location**: `scripts/citation_verifier.py`
- **Content**: Regex grammars for Indian law reports (AIR, SCC, SCR, SCALE).

### 10. Document Upload OCR Confusion Lexicon
- **Identifier**: `ds-10-ocr-confusion`
- **Storage Location**: `app/services/ocr_service.py`
- **Content**: Optical substitution mappings for legal terms (`Sec.` vs `5ec.`, `302` vs `3O2`).

---

## 3. Storage Architecture Summary

| Dataset Name | Raw Format | Processed Format | Chunks Count | Vector Index Path |
|---|---|---|---|---|
| **Constitution of India** | Official PDF / Text | JSONL | 108 specialized | `vector_db/legislation/` |
| **Central Legislation** | Parquet (79 files) | JSONL | 26,097 chunks | `vector_db/legislation/` |
| **Case Laws & Precedents** | JSON / Text | JSONL | 26,085 chunks | `vector_db/case_laws/` |
| **SFT Training Dataset** | JSONL | JSONL | 5,400 samples | N/A (Fine-Tuning) |
| **Precedent Graph** | JSON / Python AST | Memory / Dict | 15 Landmark nodes | In-Memory Service |
| **Chat History Storage** | SQLite WAL | DB Table | Dynamic | `data/legalmind.db` |
