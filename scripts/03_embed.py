#!/usr/bin/env python3
"""
scripts/03_embed.py
===================
LegalMind AI — Embedding Generation

Reads chunks from data/chunks/chunks.jsonl, generates dense vector
embeddings using BGE-M3 (or fallback models), and saves:
  - embeddings/embeddings.npy    → float32 numpy array (N × dim)
  - embeddings/chunk_ids.json    → list of chunk_ids (index → chunk_id)
  - embeddings/metadata.jsonl    → chunk metadata without full text

Model preference order:
  1. BAAI/bge-m3              (multilingual, best for legal retrieval)
  2. BAAI/bge-large-en-v1.5  (English-only, strong alternative)
  3. intfloat/e5-large-v2     (lightweight fallback)

Usage:
    python scripts/03_embed.py --input data/chunks/ --output embeddings/
    python scripts/03_embed.py --input data/chunks/ --output embeddings/ \\
        --model BAAI/bge-m3 --batch-size 32 --gpu 4

Configuration:
    --model       HuggingFace model name or local path
    --batch-size  Number of chunks per GPU batch (default: 64)
    --gpu         CUDA device index to use (default: 4)
    --max-length  Maximum sequence length (default: 512 tokens)
    --overwrite   Recompute even if embeddings already exist
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# Force offline mode to prevent HuggingFace timeout delays
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("embed")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL      = "BAAI/bge-m3"
FALLBACK_MODELS    = [
    "BAAI/bge-large-en-v1.5",
    "intfloat/e5-large-v2",
]
DEFAULT_BATCH_SIZE = 64
DEFAULT_GPU        = 4           # GPU 4 — nearly free, spare from Qwen
DEFAULT_MAX_LENGTH = 512

# BGE-M3 prefix for asymmetric retrieval (query-side only; passage side: no prefix)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_embedding_model(model_name: str, device: str):
    """
    Load a sentence-transformers embedding model.
    Returns (model, actual_model_name).
    """
    try:
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name, device=device)
        log.info(f"Embedding model loaded on {device}")
        return model, model_name
    except Exception as e:
        log.error(f"Failed to load {model_name}: {e}")
        for fallback in FALLBACK_MODELS:
            if fallback == model_name:
                continue
            log.info(f"Trying fallback: {fallback}")
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(fallback, device=device)
                log.info(f"Loaded fallback model: {fallback}")
                return model, fallback
            except Exception as fe:
                log.warning(f"Fallback {fallback} also failed: {fe}")
        raise RuntimeError("No embedding model could be loaded.") from e


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def generate_embeddings(
    texts: list[str],
    model,
    batch_size: int,
    max_length: int,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts in batches.
    Returns numpy array of shape (N, embedding_dim).
    """
    all_embeddings = []
    total = len(texts)
    start_time = time.time()

    log.info(f"Generating embeddings for {total:,} chunks (batch_size={batch_size})")

    # Set max_length directly on the model
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = max_length
        
    for batch_start in range(0, total, batch_size):
        batch = texts[batch_start: batch_start + batch_size]
        batch_embeddings = model.encode(
            batch,
            batch_size=batch_size,
            normalize_embeddings=True,   # L2-normalise for cosine similarity
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        all_embeddings.append(batch_embeddings)

        done = min(batch_start + batch_size, total)
        elapsed = time.time() - start_time
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        log.info(
            f"  Embedded {done:>6,}/{total:,}  "
            f"({done/total*100:.1f}%)  "
            f"{rate:.1f} chunks/s  "
            f"ETA {eta:.0f}s"
        )

    return np.vstack(all_embeddings).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LegalMind AI — Embedding Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",  "-i", default="data/chunks/",
                        help="Directory containing chunks.jsonl (default: data/chunks/)")
    parser.add_argument("--output", "-o", default="embeddings/",
                        help="Output directory for embeddings (default: embeddings/)")
    parser.add_argument("--model",  "-m", default=DEFAULT_MODEL,
                        help=f"Embedding model name or path (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for embedding (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU,
                        help=f"CUDA device index (default: {DEFAULT_GPU})")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                        help=f"Max token sequence length (default: {DEFAULT_MAX_LENGTH})")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Maximum number of chunks to embed (default: None = all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute embeddings even if output exists")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU (ignores --gpu)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Paths ──────────────────────────────────────────────────────────────
    input_path  = Path(args.input)
    output_dir  = Path(args.output)
    if input_path.is_file():
        chunks_file = input_path
    else:
        chunks_file = input_path / "chunks.jsonl"

    if not chunks_file.exists():
        log.error(f"chunks.jsonl not found at {chunks_file}. Run 02_chunk.py first.")
        sys.exit(1)

    out_embeddings = output_dir / "embeddings.npy"
    out_ids        = output_dir / "chunk_ids.json"
    out_meta       = output_dir / "metadata.jsonl"

    if out_embeddings.exists() and not args.overwrite:
        log.warning(f"Embeddings already exist at {out_embeddings}")
        log.warning("Use --overwrite to recompute.")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ─────────────────────────────────────────────────────────────
    try:
        import torch
        if args.cpu or not torch.cuda.is_available():
            device = "cpu"
        else:
            device = f"cuda:{args.gpu}"
            # Validate GPU index
            if args.gpu >= torch.cuda.device_count():
                log.warning(f"GPU {args.gpu} not available (only {torch.cuda.device_count()} GPUs). Using cuda:0.")
                device = "cuda:0"
        log.info(f"Device: {device}")
    except ImportError:
        device = "cpu"
        log.warning("PyTorch not available — using CPU.")

    # ── Load chunks ─────────────────────────────────────────────────────────
    log.info(f"Reading chunks from {chunks_file}")
    chunks = []
    with open(chunks_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                    if args.max_chunks and len(chunks) >= args.max_chunks:
                        break
                except json.JSONDecodeError:
                    pass

    if not chunks:
        log.error("No chunks found. Run 02_chunk.py first.")
        sys.exit(1)

    log.info(f"Loaded {len(chunks):,} chunks")

    # ── Load model ──────────────────────────────────────────────────────────
    model, actual_model = load_embedding_model(args.model, device)

    # ── Generate embeddings ─────────────────────────────────────────────────
    texts = [c.get("text", "") for c in chunks]
    embeddings = generate_embeddings(
        texts,
        model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    log.info(f"Embedding shape: {embeddings.shape}  dtype: {embeddings.dtype}")

    # ── Save outputs ────────────────────────────────────────────────────────
    log.info(f"Saving embeddings → {out_embeddings}")
    np.save(str(out_embeddings), embeddings)

    chunk_ids = [c.get("chunk_id", f"chunk_{i}") for i, c in enumerate(chunks)]
    log.info(f"Saving chunk IDs → {out_ids}")
    with open(out_ids, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False, indent=2)

    log.info(f"Saving metadata → {out_meta}")
    with open(out_meta, "w", encoding="utf-8") as f:
        for chunk in chunks:
            meta = {k: v for k, v in chunk.items() if k != "text"}
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    # ── Save embedding config for downstream scripts ────────────────────────
    config = {
        "model_name":    actual_model,
        "embedding_dim": int(embeddings.shape[1]),
        "num_vectors":   int(embeddings.shape[0]),
        "max_length":    args.max_length,
        "normalized":    True,
        "embeddings_file": str(out_embeddings),
        "chunk_ids_file":  str(out_ids),
        "metadata_file":   str(out_meta),
    }
    config_file = output_dir / "embed_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info("Embedding generation complete:")
    log.info(f"  Vectors    : {embeddings.shape[0]:,} × {embeddings.shape[1]}")
    log.info(f"  Model      : {actual_model}")
    log.info(f"  Embeddings : {out_embeddings}")
    log.info(f"  Chunk IDs  : {out_ids}")
    log.info(f"  Metadata   : {out_meta}")
    log.info(f"  Config     : {config_file}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
