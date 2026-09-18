# LegalMindAI — Installation & Setup Guide

## 1. System Requirements

### Hardware:
- **Server**: NVIDIA DGX B200 / Multi-GPU Server
- **GPUs Required**: 2x GPUs (GPU 1 for LLM inference, GPU 4 for dense embeddings and reranking)
- **Host GPU Avoidance**: **GPU 3 is strictly avoided** (reserved for existing system tasks)
- **RAM**: Minimum 64 GB system RAM
- **Disk**: 150 GB NVMe SSD storage (models, datasets, vector indexes)

### Software & Drivers:
- **Operating System**: Linux (Ubuntu 22.04 LTS or newer)
- **CUDA**: CUDA 12.2+ (`/usr/local/cuda`)
- **Python**: Python 3.10 to 3.12
- **Base Environment**: PyTorch pre-installed in `/opt/llm-training`

---

## 2. Environment Setup

To ensure complete isolation without requiring sudo permissions, create a user-owned virtual environment linked to system site packages:

```bash
cd ~/LegalMindAI

# Create virtual environment referencing system CUDA/Torch packages
python3 -m venv --system-site-packages .venv

# Activate environment
source .venv/bin/activate
```

### Install Core & Multimodal Dependencies:
```bash
pip install -r requirements.txt
pip install pypdf python-docx rich soundfile httpx pytest
```

---

## 3. Model & Data Verification

### 3.1 Verify LLM Weights
Confirm that the base model `Qwen3.6-35B-A3B` is present:
```bash
ls -lh models/Qwen3.6-35B-A3B/
```
Expected files:
- `config.json`
- `tokenizer.json`
- `model.safetensors.index.json`
- Sharded model weights (`model-00001-of-00018.safetensors`, etc.)

### 3.2 Verify Vector Indexes
```bash
.venv/bin/python scripts/validate_all_indexes.py
```
Expected Output:
- `vector_db/legislation`: 26,106 vectors, dimension 1024 [PASS]
- `vector_db/case_laws`: 26,085 vectors, dimension 1024 [PASS]

### 3.3 Verify Legal Corpus Integrity
```bash
.venv/bin/python scripts/validate_legal_corpus.py
```
Expected Output:
- 52,000+ total indexed chunks
- Article 14, 19, 21 [PRESENT]
- IPC, CrPC, IEA, BNS [PRESENT]
- Zero empty or corrupt chunks [PASS]

---

## 4. Running Verification Tests

Run the complete fast regression test suite to ensure all services and API routes are healthy:
```bash
.venv/bin/python tests/run_tests_fast.py
.venv/bin/python -m unittest tests.test_chat_store
.venv/bin/python -m unittest tests.test_cli
.venv/bin/python tests/test_multimodal_services.py
.venv/bin/python -m unittest tests.test_temporal_law
.venv/bin/python -m unittest tests.test_citation_verification
```

---

## 5. Launching the Backend Server

Start the FastAPI application with Uvicorn:
```bash
CUDA_VISIBLE_DEVICES=1,4 .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
```

### Verify Server Health:
```bash
curl http://localhost:8080/health
```
Response:
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

---

## 6. Launching the Terminal CLI

Run the Antigravity-style interactive CLI:
```bash
bin/legalmind
# or
.venv/bin/python cli/main.py
```
To ask a one-shot legal question:
```bash
bin/legalmind ask "What is the Golden Triangle under the Indian Constitution?"
```
