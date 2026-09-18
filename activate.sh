#!/bin/bash
# =============================================================
# LegalMind AI — Environment Activation Script
# =============================================================
# Usage: source activate.sh
#
# This script activates the user-owned .venv which:
#   - Inherits torch, transformers, peft, accelerate, datasets
#     from /opt/llm-training (the shared system environment)
#   - Adds trl, sentence-transformers, faiss, rank_bm25,
#     pymupdf, fastapi, uvicorn and other RAG packages
#     installed in .venv/
#
# GPU safety note: Do NOT launch scripts that modify MIG/NCCL config.
# Use GPUs: 1, 2, 4 (avoid GPU 3 — in use by another process)
# =============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: .venv not found at $VENV_DIR"
    echo "Create it with:"
    echo "  /opt/llm-training/bin/python -m venv --system-site-packages $VENV_DIR"
    echo "  $VENV_DIR/bin/pip install -r requirements.txt"
    return 1
fi

source "$VENV_DIR/bin/activate"
echo "✅ LegalMind AI environment activated"
echo "   Python  : $(which python)"
echo "   PyTorch : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "   GPUs    : $(python -c 'import torch; print(torch.cuda.device_count(), \"CUDA devices\")' 2>/dev/null || echo 'CUDA not available')"
echo ""
echo "Quick commands:"
echo "  python scripts/01_ingest.py --input data/raw/ --output data/processed/"
echo "  python scripts/02_chunk.py  --input data/processed/ --output data/chunks/"
echo "  python scripts/03_embed.py  --gpu 4"
echo "  python scripts/04_build_faiss.py"
echo "  python scripts/05_rag.py    --gpu 1 --embed-gpu 4"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1"
