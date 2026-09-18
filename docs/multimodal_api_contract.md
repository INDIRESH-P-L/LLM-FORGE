# LegalMindAI — Multimodal API Contract

This document provides the integration specification for the frontend and conversation history developers. All multimodal backend services are modular and fully decoupled from the core database and UI components.

---

## 1. Overview & General Principles

- **Base URL**: `http://<host>:8080` (all endpoints are also prefixed with `/api/` for backwards compatibility).
- **Core Architecture**: All questions about documents and images route through the unified LegalMindAI RAG inference pipeline (`Qwen3.6-35B-A3B` + legislation and case law vector indexes).
- **No Side Effects on History**: Uploading, analyzing, or evicting documents/images does not modify or delete chat history rows in `data/legalmind.db`.
- **Payload Limits**: Maximum file size is **25 MB**. Supported formats include `.pdf`, `.docx`, `.txt`, `.png`, `.jpg`, `.jpeg`, `.webp`.

---

## 2. Endpoints Specification

### A. Speech-to-Text (Voice Typing)

#### `POST /speech-to-text` (or `/api/speech-to-text`)
Transcribes an audio stream into editable text without automatically querying the model.

- **Content-Type**: `multipart/form-data`
- **Request Fields**:
  | Field | Type | Required | Description |
  |---|---|---|---|
  | `file` | File Binary | Yes | Audio recording (WAV, MP3, OGG, WEBM, M4A) |
  | `language` | String | No | Target ISO language code (`en`, `hi`, `ta`). Default: `en` |
  | `conversation_id` | String | No | Optional active conversation identifier |

- **Response (200 OK)**:
  ```json
  {
    "success": true,
    "text": "Explain Article 21 of the Constitution of India in simple language",
    "language": "en",
    "confidence": 0.94,
    "warnings": []
  }
  ```

- **Error Codes**:
  - `400 Bad Request`: Audio file empty, truncated, or unsupported format.
  - `500 Internal Server Error`: Audio decoding or transcription failure.

---

### B. Image Upload & Vision Q&A

#### 1. `POST /images/upload` (or `/api/images/upload`)
Uploads, validates, and runs initial OCR extraction on an image document.

- **Content-Type**: `multipart/form-data`
- **Request Fields**:
  | Field | Type | Required | Description |
  |---|---|---|---|
  | `file` | File Binary | Yes | Image file (JPG, JPEG, PNG, WEBP, max 25MB) |

- **Response (200 OK)**:
  ```json
  {
    "success": true,
    "image_id": "img_a1b2c3d4e5",
    "filename": "legal_notice.jpg",
    "mime_type": "image/jpeg",
    "size_bytes": 142850,
    "status": "uploaded"
  }
  ```

#### 2. `POST /images/analyze` (or `/api/images/analyze`)
Returns OCR-extracted text and identified legal entities for an uploaded image.

- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "image_id": "img_a1b2c3d4e5"
  }
  ```

- **Response (200 OK)**:
  ```json
  {
    "success": true,
    "image_id": "img_a1b2c3d4e5",
    "document_type": "legal_notice",
    "extracted_text": "LEGAL NOTICE under Section 138 of the Negotiable Instruments Act...",
    "entities": {
      "case_number": null,
      "court": null,
      "date": "14/08/2024",
      "parties": ["ABC Pvt Ltd", "XYZ Trading Co"],
      "sections": ["Section 138", "Section 142"],
      "acts": ["Negotiable Instruments Act"],
      "judges": []
    },
    "warnings": [],
    "confidence": "medium"
  }
  ```

#### 3. `POST /images/ask` (or `/api/images/ask`)
Submits a question grounded in the image text and retrieved legal corpus.

- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "image_id": "img_a1b2c3d4e5",
    "question": "What is the statutory deadline specified in this notice?",
    "language": "en"
  }
  ```

- **Response (200 OK)**:
  ```json
  {
    "success": true,
    "answer": "According to the uploaded legal notice, the recipient is required to make payment within 15 days of receipt of the notice under Section 138(c) of the Negotiable Instruments Act, 1881.",
    "extracted_text": "...notice under Section 138... payment within 15 days...",
    "citations": [
      "Negotiable Instruments Act, 1881 (Section 138)"
    ],
    "confidence": "medium",
    "evidence_coverage": 0.88,
    "warnings": [],
    "document_id": "img_a1b2c3d4e5"
  }
  ```

---

### C. Document Upload & Multi-Format Q&A

#### 1. `POST /documents/upload` (or `/api/documents/upload`)
Uploads and parses a PDF, DOCX, TXT, or scanned image document.

- **Content-Type**: `multipart/form-data`
- **Request Fields**:
  | Field | Type | Required | Description |
  |---|---|---|---|
  | `file` | File Binary | Yes | Legal document (PDF, DOCX, TXT, max 25MB) |

- **Response (200 OK)**:
  ```json
  {
    "success": true,
    "document_id": "doc_f9e8d7c6b5",
    "filename": "maneka_gandhi_judgment.pdf",
    "mime_type": "application/pdf",
    "size_bytes": 284102,
    "status": "uploaded"
  }
  ```

#### 2. `POST /documents/analyze` (or `/api/documents/analyze`)
Generates structured domain breakdown (case details, legal issues, ratio decidendi).

- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "document_id": "doc_f9e8d7c6b5"
  }
  ```

- **Response (200 OK)**:
  ```json
  {
    "success": true,
    "document_id": "doc_f9e8d7c6b5",
    "document_type": "judgment",
    "extracted_text": "IN THE SUPREME COURT OF INDIA...",
    "entities": {
      "case_name": "Maneka Gandhi v. Union of India",
      "court": "Supreme Court of India",
      "case_number": "Writ Petition No. 231 of 1977",
      "date": "25 January 1978",
      "parties": ["Maneka Gandhi", "Union of India"],
      "judges": ["M. Hameedullah Beg", "P.N. Bhagwati", "V.R. Krishna Iyer"],
      "sections": ["Article 14", "Article 19", "Article 21"],
      "acts": ["Constitution of India", "Passports Act, 1967"],
      "citations": ["1978 AIR 597", "1978 SCR (2) 621"],
      "deadlines": []
    },
    "structured_analysis": {
      "document_genre": "judgment",
      "court": "Supreme Court of India",
      "case_number": "Writ Petition No. 231 of 1977",
      "parties": ["Maneka Gandhi", "Union of India"],
      "statutory_acts": ["Constitution of India", "Passports Act, 1967"],
      "sections_referenced": ["Article 14", "Article 19", "Article 21"]
    },
    "warnings": [],
    "confidence": "high"
  }
  ```

#### 3. `POST /documents/ask` (or `/api/documents/ask`)
Asks follow-up questions regarding the active document.

- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "document_id": "doc_f9e8d7c6b5",
    "question": "What test was laid down regarding Article 21 and procedure established by law?",
    "conversation_id": "c986f951-872f-4886-b48f-8d9ad7ab0a7d"
  }
  ```

- **Response (200 OK)**:
  ```json
  {
    "success": true,
    "answer": "The Supreme Court held that the 'procedure established by law' under Article 21 must not be arbitrary, fanciful, or oppressive; it must be right, just, and fair. The Court established that Articles 14, 19, and 21 form a 'Golden Triangle' and cannot be read in mutual exclusion.",
    "extracted_text": "The procedure contemplated by Article 21 must be right, just and fair and not arbitrary...",
    "citations": [
      "Constitution of India (Article 21)",
      "Constitution of India (Article 14)",
      "Maneka Gandhi v. Union of India, AIR 1978 SC 597"
    ],
    "confidence": "high",
    "evidence_coverage": 0.95,
    "warnings": [],
    "document_id": "doc_f9e8d7c6b5"
  }
  ```

#### 4. `GET /documents/{document_id}`
Retrieves document metadata and paragraph count.

- **Response (200 OK)**:
  ```json
  {
    "success": true,
    "document_id": "doc_f9e8d7c6b5",
    "filename": "maneka_gandhi_judgment.pdf",
    "document_type": "judgment",
    "paragraph_count": 48,
    "char_count": 32150,
    "created_at": 1789283400.5,
    "metadata": {
      "page_count": 14,
      "extension": ".pdf"
    }
  }
  ```

#### 5. `DELETE /documents/{document_id}`
Evicts the active document from temporary server context.

- **Response (200 OK)**:
  ```json
  {
    "success": true,
    "document_id": "doc_f9e8d7c6b5",
    "status": "deleted"
  }
  ```

---

### D. Text-to-Speech (TTS)

#### `POST /text-to-speech` (or `/api/text-to-speech`)
Synthesizes speech or provides synthesis metadata for legal answers.

- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "text": "Article 21 protects life and personal liberty except according to procedure established by law.",
    "language": "en"
  }
  ```

- **Response (200 OK)**:
  ```json
  {
    "success": true,
    "audio_id": "audio_e4d3c2b1a0",
    "language": "en",
    "duration_seconds": 4.8,
    "word_count": 14,
    "warnings": [],
    "status": "ready"
  }
  ```

---

## 3. Frontend Integration Example (JavaScript)

```javascript
// Example: Uploading a document and asking follow-up questions
async function analyzeLegalDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  // 1. Upload
  const uploadRes = await fetch("/documents/upload", {
    method: "POST",
    body: formData,
  });
  const { document_id } = await uploadRes.json();

  // 2. Ask follow-up question
  const queryRes = await fetch("/documents/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      document_id: document_id,
      question: "What are the core arguments and ratio decidendi?",
    }),
  });
  const data = await queryRes.json();
  console.log("Answer:", data.answer);
  console.log("Citations:", data.citations);
}
```
