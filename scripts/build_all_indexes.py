#!/usr/bin/env python3
"""
scripts/build_all_indexes.py
============================
LegalMind AI — Universal Non-Destructive Vector Index Builder

Orchestrates index creation for:
- vector_db/legislation (Legislation & Constitution articles)
- vector_db/case_laws   (Supreme Court judgments & Precedents)

Safety Guarantee:
  - Non-destructive: Will NOT overwrite existing production indexes unless
    explicitly instructed via `--overwrite`.
  - Creates atomic temp files before finalizing index.
"""

import argparse
import logging
import sys
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_all_indexes")


def run_build_for_target(name: str, emb_dir: Path, out_dir: Path, overwrite: bool, use_gpu: bool) -> bool:
    log.info(f"Checking build prerequisites for target '{name}'...")
    emb_file = emb_dir / "embeddings.npy"
    ids_file = emb_dir / "chunk_ids.json"
    meta_file = emb_dir / "metadata.jsonl"

    if not emb_file.exists() or not ids_file.exists() or not meta_file.exists():
        log.warning(f"Target '{name}': Embeddings not found at {emb_dir}. Skipping build.")
        return False

    out_idx = out_dir / "index.faiss"
    if out_idx.exists() and not overwrite:
        log.info(f"Target '{name}': Index already exists at {out_idx}. Skipping (use --overwrite to force rebuild).")
        return True

    from scripts.04_build_faiss import build_index, import_faiss
    import numpy as np
    import json

    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading {name} embeddings from {emb_file}...")
    embeddings = np.load(str(emb_file)).astype(np.float32)
    with open(ids_file, "r", encoding="utf-8") as f:
        chunk_ids = json.load(f)

    if len(chunk_ids) != embeddings.shape[0]:
        log.error(f"Target '{name}' count mismatch: {len(chunk_ids)} chunk IDs vs {embeddings.shape[0]} vectors.")
        return False

    faiss = import_faiss(use_gpu)
    index, index_type = build_index(embeddings, use_gpu, nlist=None)

    # Atomic write to temporary file
    tmp_index = out_dir / "index.faiss.tmp"
    faiss.write_index(index, str(tmp_index))
    shutil.move(str(tmp_index), str(out_idx))

    shutil.copy2(ids_file, out_dir / "chunk_ids.json")
    shutil.copy2(meta_file, out_dir / "metadata.jsonl")

    cfg = {
        "index_type": index_type,
        "num_vectors": index.ntotal,
        "embedding_dim": int(embeddings.shape[1]),
        "metric": "inner_product (cosine, L2-normalised)",
        "target": name,
    }
    with open(out_dir / "index_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    log.info(f"Target '{name}' built successfully: {index.ntotal:,} vectors indexed at {out_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="LegalMind AI — Universal FAISS Index Builder")
    parser.add_argument("--overwrite", action="store_true", help="Force rebuild existing indexes")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for IVF clustering if available")
    parser.add_argument("--target", choices=["all", "legislation", "case_laws"], default="all", help="Target index to build")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    targets = []
    if args.target in ["all", "legislation"]:
        targets.append(("legislation", project_root / "embeddings" / "legislation", project_root / "vector_db" / "legislation"))
    if args.target in ["all", "case_laws"]:
        targets.append(("case_laws", project_root / "embeddings" / "case_laws", project_root / "vector_db" / "case_laws"))

    success = True
    for name, emb_dir, out_dir in targets:
        ok = run_build_for_target(name, emb_dir, out_dir, args.overwrite, args.use_gpu)
        if not ok and args.overwrite:
            success = False

    log.info("Build routine finished.")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
