# LegalMindAI — Troubleshooting & Diagnostic Guide

This guide covers common operational challenges, GPU allocation rules, adapter stability issues, and debugging procedures.

---

## 1. GPU Multi-Device Allocation & GPU 3 Avoidance

### Symptom:
Model loading fails with CUDA out-of-memory or attempts to query all 8 GPUs on the DGX server.

### Root Cause:
Running `import torch` on an 8x NVIDIA DGX B200 server without setting `CUDA_VISIBLE_DEVICES` queries every device, which can clash with external processes on GPU 3.

### Solution:
Always isolate execution to GPUs 1 and 4:
```bash
export CUDA_VISIBLE_DEVICES=1,4
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```
- GPU 1: Hosts Qwen3.6-35B-A3B LLM.
- GPU 4: Hosts BGE-M3 and BGE-Reranker.
- **GPU 3 is strictly avoided** at all times.

---

## 2. LoRA Adapter NaN Detection & Safety Fallback

### Symptom:
Model generation produces blank output, repeats `<unk>`, or outputs non-standard tokens.

### Root Cause:
The fine-tuning checkpoint `lora-legal-v1` suffered from gradient spikes during early training, embedding NaN values in weight matrices.

### Verification Command:
```bash
.venv/bin/python scripts/validate_lora_adapter.py --adapter fine_tuning/adapters/lora-legal-v1
```
### Resolution:
The system incorporates an automated safety gatekeeper:
1. `validate_adapter.py` inspects safetensors weights.
2. If any NaN tensor is found, the system **automatically falls back to Base Qwen3.6-35B-A3B + RAG**.
3. No manual action is required; the server logs an alert and continues serving requests reliably.

---

## 3. Vector Database Parity & Missing Metadata

### Symptom:
Retrieval fails with index mismatch or citations display as `"Unknown Document"`.

### Verification Command:
```bash
.venv/bin/python scripts/validate_all_indexes.py
```
### Resolution:
If metadata line counts do not match `chunk_ids.json`:
1. Run `scripts/patch_chunk_metadata.py` to restore missing citation metadata fields.
2. Confirm with `scripts/validate_legal_corpus.py`.

---

## 4. Port 8080 Conflict or Server Restart

### Symptom:
Uvicorn fails to bind to `0.0.0.0:8080` (`Address already in use`).

### Inspection:
Check active server PID:
```bash
ps aux | grep uvicorn
```
### Safe Restart Rule:
**Do not terminate the active production server until the replacement is validated.**
To perform a safe restart:
```bash
.venv/bin/python scripts/restart_server.py
```

---

## 5. Automated Browser Subagent Offline Notice

### Symptom:
Automated browser subagent fails with a CDN 404 fetching Playwright browser binaries.

### Resolution:
Institutional DGX security policies restrict headless binary downloads from third-party CDNs. To test the web UI, open **Vivaldi** (or any local browser) at:
`http://192.168.4.99:8080/` (or `http://localhost:8080/`).
Perform a hard refresh (`Ctrl + Shift + R`) to load the latest CSS styles.
