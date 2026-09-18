# LegalMind AI — Frontend & History Integration Contract
**Version:** 2.0.0  
**Last Updated:** 2026-09-12  
**Target Environment:** `source /opt/llm-training/bin/activate`  
**Base Model:** `Qwen/Qwen3.6-35B-A3B`  

---

## 1. Ownership & Collaboration Boundaries

To preserve strict separation of concerns and prevent merge conflicts during concurrent development, ownership is partitioned as follows:

| Area | Owner | Files Included | Modification Rule |
| :--- | :--- | :--- | :--- |
| **Frontend UI & Styling** | **Friend** | `app/static/index.html`, `app/static/style.css`, `app/static/app.js`, `app/index.html` | **DO NOT TOUCH** by Backend. Backend must never alter styling, markup, colors, or DOM IDs. |
| **History UI & State** | **Friend** | Sidebar components, conversation lists, message rendering, chat state | **DO NOT TOUCH** by Backend. Frontend owns rendering and client state. |
| **Backend APIs & RAG** | **Backend (You)** | `app/main.py`, `app/chat_api.py`, `scripts/05_rag.py`, `scripts/retriever.py` | Maintained by Backend; strictly backward compatible with existing `fetch('/query')`. |
| **Storage & Persistence** | **Backend (You)** | `storage/chat_store.py`, `storage/exporters.py`, `data/legalmind.db` | High-performance SQLite WAL database with FTS5 search. |
| **AI / NLP Pipelines** | **Backend (You)** | `scripts/temporal_law.py`, `scripts/precedent_graph.py`, `scripts/citation_verifier.py`, `scripts/legal_reasoning.py`, `scripts/response_generator.py` | Standalone modules providing legal reasoning, citation verification, and temporal law support. |
| **Fine-Tuning & Weights** | **Backend (You)** | `fine_tuning/configs/lora_config.yaml`, `fine_tuning/validate_adapter.py` | Standard LoRA fine-tuning only; strict NaN/Inf validation; fallback to base model. |
| **Shared Contracts** | **Joint** | `docs/frontend_backend_contract.md`, `docs/backend_frontend_integration_audit.md` | Single source of truth for API schemas and integration interfaces. |

---

## 2. API Endpoints Overview

The backend exposes all conversation, history, and search endpoints with dual routing: both under `/api/*` and at root `/*` to accommodate any frontend URL pattern.

| Method | Endpoint | Description | Auth / Identity |
| :--- | :--- | :--- | :--- |
| `POST` | `/query` | Main RAG inference endpoint (extended & backward compatible) | Cookie / `uid` parameter |
| `POST` | `/api/chat` | Durable conversation chat endpoint with idempotency | Cookie / `uid` parameter |
| `GET` | `/conversations` (or `/api/conversations`) | List user conversations (paginated, pinned first) | Cookie / `uid` parameter |
| `POST` | `/conversations` (or `/api/conversations`) | Create a new conversation | Cookie / `uid` parameter |
| `GET` | `/conversations/{id}` (or `/api/conversations/{id}`) | Fetch conversation details & all message history | Cookie / `uid` parameter |
| `PATCH` | `/conversations/{id}` (or `/api/conversations/{id}`) | Rename conversation or toggle pin status | Cookie / `uid` parameter |
| `DELETE` | `/conversations/{id}` (or `/api/conversations/{id}`) | Soft-delete a conversation (reversible) | Cookie / `uid` parameter |
| `POST` | `/conversations/{id}/restore` | Restore soft-deleted conversation (undo operation) | Cookie / `uid` parameter |
| `GET` | `/conversations/{id}/messages` | Retrieve chronological messages for a conversation | Cookie / `uid` parameter |
| `POST` | `/conversations/{id}/messages` | Manually append user/assistant message to conversation | Cookie / `uid` parameter |
| `GET` | `/conversations/{id}/export` | Export conversation transcript as Markdown (`.md`) or Text (`.txt`) | Cookie / `uid` parameter |
| `POST` | `/messages/{message_id}/feedback` | Record user feedback (`+1` thumbs up, `-1` thumbs down) | Cookie / `uid` parameter |
| `GET` | `/search` (or `/api/search`) | Full-text search (SQLite FTS5) across user's message history | Cookie / `uid` parameter |
| `GET` | `/precedents/{case_name}` | Precedent relationship graph (overruled, followed, cited cases) | Public |
| `GET` | `/precedents` | Search landmark precedent database | Public |
| `GET` | `/temporal/check` | Analyze query for historical vs. modern statutory transitions | Public |
| `GET` | `/health` | Server health, model loaded status, and uptime | Public |
| `GET` | `/stats` | System metrics, GPU memory breakdown, query counts | Public |

---

## 3. Request & Response JSON Specifications

### 3.1 `POST /query` (Main RAG & Persistence)
This endpoint maintains 100% backward compatibility with `app/static/app.js` while adding all extended fields required for full history integration.

#### Request Schema
```json
{
  "query": "What is Article 21 of the Constitution of India?",
  "top_k": 7,
  "conversation_id": "conv_a8b9c0d1-e2f3-4a5b-6c7d-8e9f0a1b2c3d",
  "client_token": "browser-uuid-or-timestamp-to-prevent-double-click"
}
```
*Note: `conversation_id`, `client_token`, and `top_k` are optional. If `conversation_id` is omitted, the backend automatically provisions a new conversation and returns its ID.*

#### Response Schema
```json
{
  "query": "What is Article 21 of the Constitution of India?",
  "answer": "## 1. Direct Answer\nArticle 21 of the Constitution of India guarantees...",
  "legal_provisions": [
    "Article 21, Constitution of India",
    "Article 14, Constitution of India",
    "Article 19, Constitution of India"
  ],
  "judgments": [
    "Maneka Gandhi v. Union of India (1978)",
    "K.S. Puttaswamy v. Union of India (2017)"
  ],
  "legal_reasoning": "Article 21 provides that no person shall be deprived of life or personal liberty except according to procedure established by law...",
  "citations": [
    "[1] Constitution of India — Article 21 (Protection of life and personal liberty)",
    "[2] Maneka Gandhi v. Union of India, AIR 1978 SC 597 : (1978) 1 SCC 248"
  ],
  "confidence": "HIGH — Directly grounded in Article 21 constitutional text and binding Supreme Court precedents.",
  "limitations": "Subject to reasonable restrictions and lawful procedures established by competent legislation.",
  "retrieved_documents": [
    {
      "citation_id": 1,
      "document_id": "doc_const_art21",
      "document_type": "constitution",
      "title": "Constitution of India",
      "source": "data/legislation/central/Constitution_of_India.json",
      "court": "Constituent Assembly of India",
      "date": "1950-01-26",
      "citation": "Article 21",
      "article": "21",
      "section": null,
      "rrf_score": 0.033,
      "rerank_score": 0.985,
      "text_preview": "21. Protection of life and personal liberty.—No person shall be deprived of his life or personal liberty except according to procedure established by law."
    }
  ],
  "retrieval_latency": 0.812,
  "generation_latency": 14.350,
  "total_latency": 15.162,
  "conversation_id": "a9d701ec-38e2-4bd5-94f7-7b19811c0ef2",
  "message_id": "msg_8f19da52-192a-4bc1-bfd8-cb10825e4071",
  "question": "What is Article 21 of the Constitution of India?",
  "answer_format": "detailed_legal_explanation",
  "confidence_score": 0.95,
  "confidence_label": "high",
  "sources": [
    {
      "title": "Constitution of India",
      "citation": "Article 21",
      "court": "Constituent Assembly of India",
      "date": "1950-01-26",
      "source_id": "doc_const_art21",
      "excerpt": "21. Protection of life and personal liberty...",
      "citation_id": 1,
      "document_type": "constitution"
    }
  ],
  "legal_warnings": [],
  "temporal_status": "current",
  "jurisdiction": "India",
  "reasoning_summary": [
    {"step": 1, "phase": "Issue Identified", "description": "Determination of scope under Article 21.", "grounded": true},
    {"step": 2, "phase": "Relevant Legal Provision", "description": "Article 21 of the Constitution.", "grounded": true},
    {"step": 3, "phase": "Applicable Rule", "description": "Procedure must be just, fair, and reasonable.", "grounded": true},
    {"step": 4, "phase": "Relevant Facts", "description": "Inquiry on fundamental right to life and liberty.", "grounded": true},
    {"step": 5, "phase": "Application of the Rule", "description": "Directly applies to all individuals in India.", "grounded": true},
    {"step": 6, "phase": "Exceptions or Limitations", "description": "Procedure established by valid enacted law.", "grounded": true},
    {"step": 7, "phase": "Conclusion", "description": "Guarantees fundamental substantive protection.", "grounded": true},
    {"step": 8, "phase": "Sources", "description": "[1] Constitution of India", "grounded": true}
  ],
  "related_cases": [
    "Maneka Gandhi v. Union of India",
    "Justice K.S. Puttaswamy (Retd.) v. Union of India"
  ],
  "related_sections": [
    "Article 21"
  ],
  "model": "Qwen/Qwen3.6-35B-A3B",
  "created_at": "2026-09-12T16:30:00Z"
}
```

---

### 3.2 `GET /conversations` (List Conversations)

#### Query Parameters
- `limit` (int, default: 100, max: 500)
- `offset` (int, default: 0)
- `include_deleted` (bool, default: false)

#### Response Schema
```json
{
  "user_id": "aeebb52e-1d18-4f36-8b42-c4953cf3ac3c",
  "conversations": [
    {
      "id": "a9d701ec-38e2-4bd5-94f7-7b19811c0ef2",
      "user_id": "aeebb52e-1d18-4f36-8b42-c4953cf3ac3c",
      "title": "Article 21 — Protection of life and personal liberty",
      "created_at": "2026-09-12T16:20:00Z",
      "updated_at": "2026-09-12T16:21:15Z",
      "pinned": 1,
      "deleted_at": null,
      "model_name": "Qwen3.6-35B-A3B",
      "message_count": 4,
      "preview": "In conclusion, Article 21 guarantees..."
    }
  ]
}
```

---

### 3.3 `GET /conversations/{conversation_id}` (Get Conversation with Messages)

#### Response Schema
```json
{
  "conversation": {
    "id": "a9d701ec-38e2-4bd5-94f7-7b19811c0ef2",
    "user_id": "aeebb52e-1d18-4f36-8b42-c4953cf3ac3c",
    "title": "Article 21 — Protection of life and personal liberty",
    "created_at": "2026-09-12T16:20:00Z",
    "updated_at": "2026-09-12T16:21:15Z",
    "pinned": 1,
    "model_name": "Qwen3.6-35B-A3B"
  },
  "messages": [
    {
      "id": "msg_111",
      "conversation_id": "a9d701ec-38e2-4bd5-94f7-7b19811c0ef2",
      "role": "user",
      "content": "What is Article 21 of the Constitution?",
      "created_at": "2026-09-12T16:20:00Z",
      "status": "complete",
      "feedback": 0,
      "metadata": {
        "client_token": "c7a8b9",
        "status": "complete"
      }
    },
    {
      "id": "msg_112",
      "conversation_id": "a9d701ec-38e2-4bd5-94f7-7b19811c0ef2",
      "role": "assistant",
      "content": "Article 21 protects life and personal liberty...",
      "created_at": "2026-09-12T16:20:45Z",
      "status": "complete",
      "feedback": 1,
      "metadata": {
        "confidence": "HIGH",
        "citations": ["[1] Constitution of India — Article 21"],
        "latency_ms": 14200,
        "model_name": "Qwen3.6-35B-A3B"
      }
    }
  ]
}
```

---

### 3.4 `GET /precedents/{case_name}` (Precedent Relationship Graph)

#### Example Request
`GET /precedents/Kesavananda%20Bharati`

#### Response Schema
```json
{
  "case_found": true,
  "case": {
    "case_name": "Kesavananda Bharati v. State of Kerala",
    "court": "Supreme Court of India",
    "year": 1973,
    "citation": "AIR 1973 SC 1461 : (1973) 4 SCC 225",
    "judge": "S.M. Sikri, C.J. (13-Judge Constitution Bench)",
    "bench_size": 13,
    "legal_issue": "Constitutional validity of 24th, 25th, 29th Amendments; Basic Structure Doctrine and amending power under Article 368.",
    "statutes_cited": [
      "Constitution of India, Articles 13, 31, 31C, 368"
    ],
    "cases_cited": [
      "I.C. Golaknath v. State of Punjab (1967)",
      "Sajjan Singh v. State of Rajasthan (1965)",
      "Shankari Prasad v. Union of India (1951)"
    ],
    "overruled_cases": [
      "I.C. Golaknath v. State of Punjab (1967) (prospectively overruled)"
    ],
    "followed_cases": [],
    "distinguished_cases": [],
    "relied_upon_cases": [
      "Shankari Prasad v. Union of India (1951)",
      "Sajjan Singh v. State of Rajasthan (1965)"
    ]
  },
  "outward": {
    "overruled_cases": ["I.C. Golaknath v. State of Punjab (1967) (prospectively overruled)"],
    "followed_cases": [],
    "distinguished_cases": [],
    "relied_upon_cases": ["Shankari Prasad v. Union of India (1951)"],
    "statutes_cited": ["Constitution of India, Articles 13, 31, 31C, 368"]
  },
  "inward": {
    "overruled_by": [],
    "followed_by": ["S.R. Bommai v. Union of India"],
    "cited_by": []
  }
}
```

---

### 3.5 `GET /temporal/check?q=...` (Temporal Law Check)

#### Example Request
`GET /temporal/check?q=Section%20302%20IPC%20and%20Section%20438%20CrPC`

#### Response Schema
```json
{
  "temporal_status": "repealed",
  "is_repealed_law_cited": true,
  "repealed_acts": [
    "Indian Penal Code, 1860",
    "Code of Criminal Procedure, 1973"
  ],
  "current_acts": [],
  "suggested_transitions": [
    {
      "historical_act": "IPC",
      "historical_section": "302",
      "modern_act": "BNS",
      "modern_section": "103",
      "title": "Punishment for murder"
    },
    {
      "historical_act": "CrPC",
      "historical_section": "438",
      "modern_act": "BNSS",
      "modern_section": "482",
      "title": "Direction for grant of bail to person apprehending arrest (Anticipatory Bail)"
    }
  ],
  "warnings": [
    "Notice on Legal Status: 'Indian Penal Code, 1860' was repealed and replaced by 'Bharatiya Nyaya Sanhita, 2023' on 2024-07-01.",
    "Section Transition: Section 302 of IPC corresponds to Section 103 of BNS (Punishment for murder).",
    "Notice on Legal Status: 'Code of Criminal Procedure, 1973' was repealed and replaced by 'Bharatiya Nagarik Suraksha Sanhita, 2023' on 2024-07-01.",
    "Section Transition: Section 438 of CrPC corresponds to Section 482 of BNSS (Direction for grant of bail to person apprehending arrest (Anticipatory Bail))."
  ],
  "jurisdiction": "India"
}
```

---

## 4. User Identity & Authentication Assumptions

1. **Anonymous Identity by Default:**  
   Every client is assigned a unique `user_id` (UUID4) persisted in an `httpOnly` cookie (`legalmind_uid`, 10-year expiration).
2. **URL Override Support:**  
   The client can pass `?uid=...` in query parameters. The URL parameter takes precedence and automatically updates the session cookie.
3. **Future Authentication Seam:**  
   When OAuth/User Login is introduced, only `get_or_create_user_id()` in `app/chat_api.py` needs to return the authenticated subject ID. All database queries, conversations, and message relationships will carry over without any schema migration.

---

## 5. Error Codes and Standard Formats

Errors return standard JSON payloads:
```json
{
  "detail": "Descriptive error message"
}
```

Common status codes:
- `200 OK`: Request succeeded.
- `400 Bad Request`: Query was empty or invalid parameter supplied.
- `404 Not Found`: Conversation, message, or precedent not found.
- `500 Internal Server Error`: Unhandled backend exception (detailed error logged on server).
- `503 Service Unavailable`: GPU Model not yet loaded or initialized.

---

## 6. Guidance for the Frontend & History Developer

1. **Continue Using Existing `fetch('/query')`:**
   You do not need to rewrite your fetch calls. The endpoint continues to return `answer`, `confidence`, `retrieved_documents`, `retrieval_latency`, and `generation_latency`.
2. **Adding History to the UI:**
   - To render the sidebar of previous chats, call `GET /conversations`.
   - When the user selects a conversation from the list, call `GET /conversations/{conversation_id}` to retrieve the full chat exchange.
   - To pass conversation context in new questions, simply include `"conversation_id": "..."` in your `fetch('/query')` payload.
3. **Reversible Deletion:**
   When the user clicks "Delete" on a conversation, invoke `DELETE /conversations/{id}`. To provide an "Undo" toast, call `POST /conversations/{id}/restore`.
4. **Markdown Rendering:**
   The `answer` field is pre-formatted in standard GitHub Flavored Markdown with clean headers (`## 1. Direct Answer`, etc.). Your existing `marked.parse(data.answer)` handles this out of the box.
