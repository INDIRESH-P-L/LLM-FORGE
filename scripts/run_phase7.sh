#!/bin/bash
set -e

echo "Starting Phase 7: Case Laws Pipeline"
echo "===================================="

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

VENV="/home/sece2026-student12/LegalMindAI/.venv/bin/python"
DOCS="data/processed/case_laws/documents.jsonl"
CHUNKS="data/chunks/case_laws/chunks.jsonl"
EMBEDDINGS="embeddings/case_laws/"
VECTOR_DB="vector_db/case_laws/"

if [ -f "$CHUNKS" ]; then
    echo "[1/3] Case laws chunks already exist at $CHUNKS ($(du -h "$CHUNKS" | cut -f1)). Skipping chunking."
else
    echo "[1/3] Chunking case laws from $DOCS..."
    $VENV scripts/02_chunk.py --input "$DOCS" --output data/chunks/case_laws/ --verbose
fi

echo "[2/3] Embedding chunks to $EMBEDDINGS on GPU 3..."
$VENV scripts/03_embed.py --input "$CHUNKS" --output "$EMBEDDINGS" --gpu 3 "$@"

echo "[3/3] Building FAISS index in $VECTOR_DB..."
$VENV scripts/04_build_faiss.py --embeddings "$EMBEDDINGS" --output "$VECTOR_DB"

echo "Phase 7 Complete!"
