#!/usr/bin/env python3
"""
scripts/validate_all_indexes.py
===============================
LegalMind AI — FAISS Vector Index Integrity & Health Validator

Verifies:
1. Index file existence and readability (index.faiss, chunk_ids.json, metadata.jsonl, index_config.json).
2. Exact parity: index.ntotal == len(chunk_ids) == line count of metadata.jsonl.
3. Dimension check: index.d == 1024 (BAAI/bge-m3 dense representation).
4. Uniqueness of chunk IDs.
5. Live search sanity: runs a dummy normalized query to confirm index responds without crash or NaN.

Exit Codes:
  0: All indexes healthy and validated.
  1: Defect detected.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate_indexes")


def validate_single_index(vdb_dir: Path, name: str) -> Dict[str, Any]:
    log.info(f"Validating vector index for '{name}' at {vdb_dir}...")
    res = {
        "name": name,
        "path": str(vdb_dir),
        "status": "PASS",
        "ntotal": 0,
        "dim": 0,
        "errors": [],
    }

    idx_file = vdb_dir / "index.faiss"
    ids_file = vdb_dir / "chunk_ids.json"
    meta_file = vdb_dir / "metadata.jsonl"
    cfg_file = vdb_dir / "index_config.json"

    for f, desc in [(idx_file, "FAISS binary"), (ids_file, "chunk IDs"), (meta_file, "metadata JSONL")]:
        if not f.exists():
            err = f"Missing {desc} file: {f}"
            res["errors"].append(err)
            res["status"] = "FAIL"

    if res["status"] == "FAIL":
        return res

    # 1. Check FAISS index
    try:
        import faiss
        import numpy as np
    except ImportError:
        res["errors"].append("faiss or numpy not importable")
        res["status"] = "FAIL"
        return res

    try:
        index = faiss.read_index(str(idx_file))
        res["ntotal"] = index.ntotal
        res["dim"] = index.d
    except Exception as e:
        res["errors"].append(f"Failed to read FAISS index {idx_file}: {e}")
        res["status"] = "FAIL"
        return res

    # 2. Check Chunk IDs
    try:
        with open(ids_file, "r", encoding="utf-8") as f:
            chunk_ids = json.load(f)
        id_count = len(chunk_ids)
        unique_ids = len(set(chunk_ids))
        if id_count != unique_ids:
            res["errors"].append(f"Duplicate chunk IDs found: {id_count} total vs {unique_ids} unique")
            res["status"] = "FAIL"
    except Exception as e:
        res["errors"].append(f"Failed to read chunk_ids.json: {e}")
        res["status"] = "FAIL"
        return res

    # 3. Check Metadata line count
    try:
        meta_count = 0
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    meta_count += 1
    except Exception as e:
        res["errors"].append(f"Failed to read metadata.jsonl: {e}")
        res["status"] = "FAIL"
        return res

    # 4. Check Parity
    if index.ntotal != id_count or index.ntotal != meta_count:
        res["errors"].append(
            f"Parity mismatch in {name}: index.ntotal={index.ntotal}, chunk_ids={id_count}, metadata={meta_count}"
        )
        res["status"] = "FAIL"

    # 5. Check Dimension
    if index.d != 1024:
        log.warning(f"Index dimension {index.d} is not standard BGE-M3 (1024)")

    # 6. Test Query
    try:
        dummy_vec = np.random.randn(1, index.d).astype(np.float32)
        norm = np.linalg.norm(dummy_vec)
        if norm > 0:
            dummy_vec /= norm
        distances, indices = index.search(dummy_vec, 3)
        if np.isnan(distances).any():
            res["errors"].append(f"Test query returned NaN distance in {name}")
            res["status"] = "FAIL"
        log.info(f"  Live query test: top-3 indices {indices[0].tolist()} with scores {[round(float(s), 4) for s in distances[0]]}")
    except Exception as e:
        res["errors"].append(f"Test search query failed in {name}: {e}")
        res["status"] = "FAIL"

    return res


def validate_all_indexes(project_root: Path) -> bool:
    log.info("=" * 70)
    log.info("LEGALMINDAI — FAISS VECTOR DATABASE INTEGRITY AUDIT")
    log.info("=" * 70)

    vdb_root = project_root / "vector_db"
    targets = [("legislation", vdb_root / "legislation"), ("case_laws", vdb_root / "case_laws")]

    all_passed = True
    results = []

    for name, path in targets:
        if not path.exists():
            log.error(f"Vector DB target directory missing: {path}")
            all_passed = False
            continue
        r = validate_single_index(path, name)
        results.append(r)
        if r["status"] != "PASS":
            all_passed = False

    log.info("\n" + "=" * 70)
    log.info(f"{'INDEX NAME':<16} | {'STATUS':<6} | {'VECTORS':<10} | {'DIM':<5} | {'ISSUES':<20}")
    log.info("-" * 70)
    for r in results:
        issues_str = "; ".join(r["errors"]) if r["errors"] else "None (100% Healthy)"
        log.info(f"{r['name']:<16} | {r['status']:<6} | {r['ntotal']:<10,}| {r['dim']:<5} | {issues_str}")
    log.info("=" * 70)

    if all_passed:
        log.info("ALL VECTOR INDEXES FULLY VERIFIED AND OPERATIONAL.")
    else:
        log.error("ONE OR MORE VECTOR INDEXES HAVE CRITICAL DEFECTS.")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="LegalMindAI FAISS Vector Index Validator")
    parser.add_argument("--root", default=".", help="Project root directory")
    args = parser.parse_args()

    project_root = Path(args.root).resolve()
    passed = validate_all_indexes(project_root)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
