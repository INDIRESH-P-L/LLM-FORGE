# LegalMindAI CLI — Backend API Contract

## Overview

This specification details the contract between the **LegalMindAI CLI** and the **LegalMindAI FastAPI Backend** (`http://127.0.0.1:8080`).

The CLI operates as an API client against existing backend endpoints without introducing duplicate database tables, redundant model loaders, or touching frontend web UI assets.

---

## 1. Subsystem Health & Diagnostics

### `GET /health`
Checks server availability, model readiness, and retriever status.

- **Request**: No body or query parameters.
- **Response Schema** (`200 OK`):
  ```json
  {
    "status": "ok",
    "model_loaded": true,
    "retriever_loaded": true,
    "uptime_s": 1420.5,
    "queries_served": 18
  }
  ```

### `GET /stats`
Returns system memory, GPU utilization, and model metadata.

- **Response Schema** (`200 OK`):
  ```json
  {
    "gpu": {
      "device": "NVIDIA DGX A100",
      "memory_used": "14.2 GB / 80 GB"
    },
    "model": {
      "name": "Qwen/Qwen3.6-35B-A3B",
      "path": "models/Qwen3.6-35B-A3B"
    }
  }
  ```

---

## 2. Conversational Multi-Turn Chat

### `POST /api/chat`
Primary endpoint for interactive conversations, multi-turn follow-ups, and document questions.

- **Request Body**:
  ```json
  {
    "query": "Explain the connection between Article 14 and Article 21.",
    "conversation_id": "c1f03f7a-4c28-48be-8f7b-91d5eb84318c",
    "client_token": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
    "top_k": 7,
    "mode": "detailed"
  }
  ```

  - `query` (*string, required*): The user inquiry or document-injected inquiry.
  - `conversation_id` (*string, optional*): UUID of the active conversation. If omitted, a new conversation is created.
  - `client_token` (*string, optional*): Idempotency token to prevent duplicate inference.
  - `top_k` (*integer, optional*): Number of evidence chunks to retrieve.
  - `mode` (*string, optional*): Target depth (`simple`, `detailed`, `student`, `case-analysis`).

- **Response Schema** (`200 OK`):
  ```json
  {
    "conversation_id": "c1f03f7a-4c28-48be-8f7b-91d5eb84318c",
    "conversation": {
      "id": "c1f03f7a-4c28-48be-8f7b-91d5eb84318c",
      "title": "Article 14 and Article 21 Connection",
      "message_count": 2,
      "model_name": "Qwen/Qwen3.6-35B-A3B",
      "created_at": "2026-09-13T06:12:00Z"
    },
    "user_message": {
      "id": "m1-user-id",
      "role": "user",
      "content": "Explain the connection between Article 14 and Article 21.",
      "created_at": "2026-09-13T06:12:00Z"
    },
    "assistant_message": {
      "id": "m2-assistant-id",
      "role": "assistant",
      "content": "### Connection Between Article 14 and Article 21\n\n...",
      "metadata": {
        "status": "complete",
        "confidence": "HIGH",
        "evidence_coverage": 92.0,
        "latency_ms": 3200,
        "citations": [
          {
            "id": "leg_const_art_14",
            "title": "Constitution of India",
            "section": "14",
            "source_url": "https://legislative.gov.in/constitution-of-india/",
            "document_type": "constitution"
          },
          {
            "id": "leg_const_art_21",
            "title": "Constitution of India",
            "section": "21",
            "source_url": "https://legislative.gov.in/constitution-of-india/",
            "document_type": "constitution"
          }
        ],
        "warnings": []
      }
    },
    "citations": [...],
    "confidence": "HIGH",
    "evidence_coverage": 92.0,
    "warnings": []
  }
  ```

---

## 3. Direct Stateless Query

### `POST /query`
Backward-compatible direct query endpoint without session persistence.

- **Request Body**:
  ```json
  {
    "query": "What is Section 302 of the Indian Penal Code?",
    "mode": "simple",
    "top_k": 5
  }
  ```

- **Response Schema** (`200 OK`):
  ```json
  {
    "query": "What is Section 302 of the Indian Penal Code?",
    "answer": "Section 302 of the Indian Penal Code prescribes the punishment for murder...",
    "sources": [
      {
        "title": "Indian Penal Code, 1860",
        "section": "302",
        "url": "https://indiacode.nic.in"
      }
    ],
    "confidence": "HIGH",
    "evidence_coverage": 95.0,
    "warnings": []
  }
  ```

---

## 4. Conversation History Management

### `GET /api/conversations`
Retrieves past conversations.

- **Query Parameters**: `limit` (*integer, default 20*).
- **Response Schema**:
  ```json
  [
    {
      "id": "c1f03f7a-4c28-48be-8f7b-91d5eb84318c",
      "title": "Article 14 and Article 21 Connection",
      "message_count": 4,
      "model_name": "Qwen/Qwen3.6-35B-A3B",
      "updated_at": "2026-09-13T06:15:00Z"
    }
  ]
  ```

### `GET /api/conversations/{conversation_id}`
Retrieves a specific conversation with all message exchanges.

---

## 5. Multimodal & Document Upload Flow

To avoid disturbing the backend with custom binary multipart routes when analyzing documents via CLI:
1. **Client-side Parsing**: The CLI parses `.pdf`, `.docx`, `.txt`, and `.png/.jpg` files using `fitz`, `pdfplumber`, and `PIL`.
2. **Context Injection**: The parsed legal text is injected into `POST /api/chat` with document metadata (`document_name`, `document_type`, `char_count`).
3. **Continuous Threading**: Subsequent turns in the CLI reference `conversation_id`, preserving full context without requiring server file uploads.

---

## 6. Error & Interruption Handling

### Generation Timeout / Interruption
When inference is canceled or times out:
```json
{
  "conversation_id": "c1f03f7a...",
  "assistant_message": {
    "content": "_(generation interrupted — timeout reached while generating answer)_",
    "metadata": {
      "status": "partial",
      "error_code": "GENERATION_INTERRUPTED"
    }
  },
  "failed": true,
  "error": "Generation timed out after 180 seconds."
}
```
The CLI catches this and displays:
```
[WARNING] Answer generation was interrupted.
[INFO] The displayed answer may be incomplete.
```
