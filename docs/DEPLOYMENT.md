# LegalMindAI — Production Deployment & Operations Guide

This guide describes operational procedures for deploying, maintaining, monitoring, and updating LegalMindAI on the NVIDIA DGX B200 platform.

---

## 1. Production Architecture Overview

- **Host**: `NvidiaComputing`
- **Application Server**: Uvicorn ASGI server running FastAPI on port `8080`.
- **Worker Configuration**: Single-worker architecture (`--workers 1`) to preserve in-memory GPU models and avoid redundant VRAM replication.
- **Model In-Memory Footprint**:
  - LLM: `models/Qwen3.6-35B-A3B` on GPU 1 (~38.4 GB VRAM).
  - Embeddings & Reranker: `BAAI/bge-m3` & `BAAI/bge-reranker-v2-m3` on GPU 4 (~7.0 GB VRAM).
  - Total VRAM: ~45.4 GB across 2 GPUs.
- **Persistence**: SQLite WAL database (`data/legalmind.db`).

---

## 2. Server Startup & Service Management

### Safe Manual Startup Command:
```bash
cd ~/LegalMindAI
export CUDA_VISIBLE_DEVICES=1,4
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
```

### Background Daemon Execution:
To run the server continuously in the background without terminal attachment:
```bash
nohup .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1 > logs/server.log 2>&1 &
```

### Checking Active Process:
```bash
ps aux | grep uvicorn
```

---

## 3. Zero-Downtime Safe Restart Protocol

**Mandatory Rule: Never terminate the active production server until the replacement has passed verification.**

LegalMindAI provides an automated safe restart utility (`scripts/restart_server.py`):
```bash
.venv/bin/python scripts/restart_server.py
```
This script:
1. Runs fast unit tests and syntax checks.
2. Checks GPU 1 and GPU 4 availability.
3. Spawns the new server process.
4. Waits for `GET /health` to return `200 OK`.
5. Only after health confirmation does it gracefully terminate the old process PID.

---

## 4. Health Checks & Monitoring

### Automated Health Endpoint:
```bash
curl -f http://localhost:8080/health || exit 1
```

### Verification Checklist:
- [ ] `GET /health` returns status `healthy` with both vector index counts > 26,000.
- [ ] `GET /stats` returns positive chunk counts and valid database size.
- [ ] `POST /query` responds with structured answer within expected latency (< 500ms TTFT).
- [ ] GPU 3 remains completely unallocated.

---

## 5. Rollback Procedures

If an updated vector index or code change introduces defects:
1. **Vector Index Rollback**:
   Restore previous FAISS index and metadata:
   ```bash
   cp vector_db/legislation/index.faiss.bak vector_db/legislation/index.faiss
   ```
2. **Adapter Rollback**:
   If an experimental adapter fails or encounters NaNs, the system automatically falls back to Base Qwen3.6-35B without manual intervention.
3. **Database Rollback**:
   `data/legalmind.db` operates with atomic transactions and WAL checkpoints; point-in-time backups are stored in `data/metadata/`.
