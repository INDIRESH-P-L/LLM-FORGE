# LegalMindAI — Complete REST API Reference

Base URL: `http://localhost:8080` (or host IP `http://192.168.4.99:8080`)

---

## 1. System & Health Endpoints

### `GET /health`
Returns the operational health and index vector counts of the backend.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "app": "LegalMind AI",
  "version": "1.0.0",
  "indexes": {
    "legislation_vectors": 26106,
    "case_law_vectors": 26085
  }
}
```

### `GET /stats`
Returns system statistics, index sizes, and persistence metrics.

---

## 2. Query & Legal Reasoning Endpoints

### `POST /query`
Main RAG generation endpoint. Executes hybrid retrieval, cross-encoder reranking, and structured answer synthesis.

**Request Body (`application/json`):**
```json
{
  "query": "Explain the Golden Triangle under the Constitution of India and how Maneka Gandhi connected Articles 14, 19, and 21.",
  "top_k": 7,
  "conversation_id": "conv_abc123",
  "mode": "detailed"
}
```

**Response (200 OK):**
```json
{
  "answer": "## 1. Direct Answer\nArticles 14, 19, and 21 form the Golden Triangle...",
  "confidence": "HIGH — Direct statutory texts for Articles 14, 19, and 21 retrieved.",
  "evidence_coverage": 0.95,
  "retrieved_documents": [
    {
      "citation_id": 1,
      "document_id": "const_art_21",
      "document_type": "constitution",
      "title": "Constitution of India — Article 21",
      "score": 0.942
    }
  ],
  "citations": [
    "[1] Constitution of India — Article 21",
    "[2] Supreme Court of India — AIR 1978 SC 597 (Maneka Gandhi)"
  ],
  "reasoning_steps": [
    {
      "step": 1,
      "phase": "Issue Identified",
      "description": "Determination of legal rights regarding Golden Triangle...",
      "grounded": true
    }
  ],
  "temporal_context": {
    "has_temporal_context": false,
    "warnings": []
  }
}
```

---

## 3. Multimodal Endpoints

### `POST /documents/upload`
Uploads and parses PDF, DOCX, or TXT documents, extracting full text and legal metadata.

**Form Data:**
- `file`: Multipart file (`.pdf`, `.docx`, `.txt`)
- `conversation_id`: Optional string

**Response (200 OK):**
```json
{
  "document_id": "doc_9f8a7b6c",
  "filename": "Special_Leave_Petition.pdf",
  "num_pages": 12,
  "character_count": 28450,
  "metadata": {
    "detected_acts": ["Constitution of India", "Code of Civil Procedure, 1908"],
    "detected_sections": ["Article 136", "Order XXXIX Rule 1"],
    "court": "Supreme Court of India"
  },
  "status": "ready"
}
```

### `POST /images/upload`
Uploads document images (court stamps, agreements, summons) for OCR and visual analysis.

**Form Data:**
- `file`: Multipart image (`.png`, `.jpg`, `.jpeg`, `.webp`)
- `prompt`: Optional inquiry regarding the image

**Response (200 OK):**
```json
{
  "image_id": "img_1a2b3c4d",
  "extracted_text": "BEFORE THE HON'BLE HIGH COURT OF DELHI AT NEW DELHI...",
  "detected_elements": ["Court Seal", "Notary Stamp", "Advocate Signature"],
  "analysis": "Affidavit executed under Section 139 of the Code of Civil Procedure."
}
```

### `POST /speech-to-text`
Transcribes spoken legal queries into text using local Whisper ASR.

**Form Data:**
- `audio`: Multipart audio file (`.wav`, `.mp3`, `.ogg`, `.webm`)

**Response (200 OK):**
```json
{
  "transcribed_text": "What are the essential conditions for anticipatory bail under Section 438 of the Code of Criminal Procedure?",
  "language": "en",
  "duration_seconds": 5.4
}
```

### `POST /text-to-speech`
Synthesizes speech audio for a legal response.

**Request Body (`application/json`):**
```json
{
  "text": "Article 21 guarantees the right to life and personal liberty."
}
```

**Response (200 OK):** Audio binary stream (`audio/wav`).

---

## 4. Legal Intelligence Endpoints

### `GET /temporal/check?q={query}`
Checks if an inquiry references historical, amended, or repealed statutes (e.g. IPC, CrPC, IEA).

**Example Request:**
```bash
curl "http://localhost:8080/temporal/check?q=Section%20302%20IPC"
```

**Response (200 OK):**
```json
{
  "has_temporal_context": true,
  "act_name": "Indian Penal Code, 1860",
  "status": "repealed",
  "repealed_date": "2024-07-01",
  "repealed_by": "Bharatiya Nyaya Sanhita, 2023",
  "modern_counterpart_section": "Section 103 BNS",
  "warnings": [
    "Note: Section 302 IPC has been replaced by Section 103 of the Bharatiya Nyaya Sanhita, 2023 with effect from 1 July 2024."
  ]
}
```

### `GET /precedents/{case_name}`
Retrieves precedent nodes and citations for landmark Supreme Court judgments.

**Example Request:**
```bash
curl "http://localhost:8080/precedents/Kesavananda%20Bharati"
```

**Response (200 OK):**
```json
{
  "case_found": true,
  "case": {
    "case_name": "Kesavananda Bharati v. State of Kerala",
    "court": "Supreme Court of India",
    "year": 1973,
    "citation": "AIR 1973 SC 1461 : (1973) 4 SCC 225",
    "bench_size": 13,
    "overruled_cases": ["I.C. Golaknath v. State of Punjab (1967)"],
    "legal_issue": "Constitutional validity of Amendments; Basic Structure Doctrine."
  }
}
```

### `GET /conversations`
Lists stored conversations from SQLite WAL storage.
