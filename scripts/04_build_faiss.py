#!/usr/bin/env python3
"""
scripts/04_build_faiss.py
=========================
LegalMind AI — FAISS Vector Index Builder

Reads embeddings from embeddings/ and builds a FAISS index for
efficient approximate nearest-neighbour (ANN) search.

Index type selection (automatic based on corpus size):
  < 10,000  vectors  → IndexFlatIP   (exact cosine, no training needed)
  ≥ 10,000  vectors  → IndexIVFFlat  (IVF quantised, faster at scale)

All embeddings are L2-normalised during generation (03_embed.py),
so inner-product (IP) search = cosine similarity.

Saves to vector_db/:
  index.faiss              → FAISS index (binary)
  chunk_ids.json           → chunk_id list (position → chunk_id)
  metadata.jsonl           → full chunk metadata (text + fields)
  index_config.json        → index configuration record

Usage:
    python scripts/04_build_faiss.py --embeddings embeddings/ --output vector_db/
    python scripts/04_build_faiss.py --embeddings embeddings/ --output vector_db/ \\
        --nlist 128 --use-gpu

Configuration:
    --nlist       Number of IVF cells (default: auto = sqrt(N))
    --use-gpu     Use GPU for index training/search (requires faiss-gpu)
    --overwrite   Rebuild even if index already exists
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("faiss_build")

# ---------------------------------------------------------------------------
# FAISS import with GPU fallback
# ---------------------------------------------------------------------------

def import_faiss(use_gpu: bool):
    try:
        import faiss
        log.info(f"FAISS loaded (version: {faiss.__version__ if hasattr(faiss,'__version__') else 'unknown'})")
        return faiss
    except ImportError:
        log.error("FAISS not installed. Run: pip install faiss-gpu  (or faiss-cpu)")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_index(
    embeddings: np.ndarray,
    use_gpu: bool,
    nlist: int | None,
) -> tuple:
    """
    Build and return (index, index_type_str).
    embeddings: float32 array (N, dim), L2-normalised.
    """
    faiss = import_faiss(use_gpu)
    N, dim = embeddings.shape
    log.info(f"Building FAISS index for {N:,} vectors of dim {dim}")

    # ── Choose index type ──────────────────────────────────────────────────
    if N < 10_000:
        # Exact flat inner-product index — no training required
        index = faiss.IndexFlatIP(dim)
        index_type = "IndexFlatIP"
        log.info(f"Index type: {index_type} (exact, N={N} < 10,000)")
    else:
        # IVF index with configurable nlist
        if nlist is None:
            nlist = max(64, int(math.sqrt(N)))
        nlist = min(nlist, N // 10)   # nlist must be << N
        quantiser = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index_type = f"IndexIVFFlat(nlist={nlist})"
        log.info(f"Index type: {index_type} (N={N:,})")

    # ── GPU resources ──────────────────────────────────────────────────────
    gpu_res = None
    if use_gpu:
        try:
            gpu_res = faiss.StandardGpuResources()
            index   = faiss.index_cpu_to_gpu(gpu_res, 0, index)
            log.info("FAISS index moved to GPU for training/search")
        except Exception as e:
            log.warning(f"GPU FAISS failed ({e}) — falling back to CPU.")
            use_gpu = False

    # ── Train (required for IVF) ───────────────────────────────────────────
    if not index.is_trained:
        log.info(f"Training IVF index on {N:,} vectors…")
        t0 = time.time()
        index.train(embeddings)
        log.info(f"Training complete in {time.time()-t0:.1f}s")

    # ── Add vectors ────────────────────────────────────────────────────────
    log.info("Adding vectors to index…")
    t0 = time.time()
    index.add(embeddings)
    log.info(f"Added {index.ntotal:,} vectors in {time.time()-t0:.1f}s")

    # ── Move back to CPU for serialisation ────────────────────────────────
    if use_gpu and gpu_res is not None:
        index = faiss.index_gpu_to_cpu(index)
        log.info("Index moved back to CPU for saving")

    return index, index_type


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LegalMind AI — FAISS Vector Index Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--embeddings", "-e", default="embeddings/",
                        help="Directory containing embeddings.npy (default: embeddings/)")
    parser.add_argument("--input", "-i", default=None,
                        help="Optional input chunks file (compatibility with run_phase7.sh)")
    parser.add_argument("--output", "-o", default="vector_db/",
                        help="Output directory for FAISS index (default: vector_db/)")
    parser.add_argument("--nlist", type=int, default=None,
                        help="IVF nlist (default: auto = sqrt(N))")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU for FAISS operations (requires faiss-gpu)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild index even if it already exists")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Paths ──────────────────────────────────────────────────────────────
    emb_dir     = Path(args.embeddings)
    output_dir  = Path(args.output)
    emb_file    = emb_dir / "embeddings.npy"
    ids_file    = emb_dir / "chunk_ids.json"
    meta_file   = emb_dir / "metadata.jsonl"
    config_file = emb_dir / "embed_config.json"

    out_index   = output_dir / "index.faiss"
    out_ids     = output_dir / "chunk_ids.json"
    out_meta    = output_dir / "metadata.jsonl"
    out_config  = output_dir / "index_config.json"

    # ── Pre-checks ─────────────────────────────────────────────────────────
    for f in [emb_file, ids_file, meta_file]:
        if not f.exists():
            log.error(f"Missing file: {f}. Run 03_embed.py first.")
            sys.exit(1)

    if out_index.exists() and not args.overwrite:
        log.warning(f"Index already exists at {out_index}. Use --overwrite to rebuild.")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load embeddings ────────────────────────────────────────────────────
    log.info(f"Loading embeddings from {emb_file}")
    embeddings = np.load(str(emb_file)).astype(np.float32)
    log.info(f"Loaded embeddings: {embeddings.shape}")

    with open(ids_file, encoding="utf-8") as f:
        chunk_ids = json.load(f)

    assert len(chunk_ids) == embeddings.shape[0], \
        f"Mismatch: {len(chunk_ids)} chunk_ids vs {embeddings.shape[0]} vectors"

    # ── Build index ────────────────────────────────────────────────────────
    faiss = import_faiss(args.use_gpu)
    index, index_type = build_index(embeddings, args.use_gpu, args.nlist)

    # ── Save index ─────────────────────────────────────────────────────────
    log.info(f"Writing FAISS index → {out_index}")
    faiss.write_index(index, str(out_index))

    # ── Copy chunk IDs and metadata to vector_db/ ──────────────────────────
    import shutil
    shutil.copy2(ids_file, out_ids)
    shutil.copy2(meta_file, out_meta)

    # ── Read embedding config ──────────────────────────────────────────────
    embed_cfg = {}
    if config_file.exists():
        with open(config_file) as f:
            embed_cfg = json.load(f)

    # ── Save index config ──────────────────────────────────────────────────
    index_cfg = {
        "index_type":    index_type,
        "num_vectors":   index.ntotal,
        "embedding_dim": int(embeddings.shape[1]),
        "metric":        "inner_product (cosine, L2-normalised)",
        "model_name":    embed_cfg.get("model_name", "unknown"),
        "index_file":    str(out_index),
        "chunk_ids_file": str(out_ids),
        "metadata_file":  str(out_meta),
    }
    with open(out_config, "w") as f:
        json.dump(index_cfg, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info("FAISS index built successfully:")
    log.info(f"  Index type  : {index_type}")
    log.info(f"  Vectors     : {index.ntotal:,}")
    log.info(f"  Dimension   : {embeddings.shape[1]}")
    log.info(f"  Index file  : {out_index}  ({out_index.stat().st_size//1024:,} KB)")
    log.info(f"  Config      : {out_config}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
