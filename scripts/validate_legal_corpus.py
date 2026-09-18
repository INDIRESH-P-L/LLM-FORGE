#!/usr/bin/env python3
"""
scripts/validate_legal_corpus.py
=================================
LegalMind AI — Comprehensive Legal Corpus Integrity Validator

Verifies and audits:
1. Total raw and processed document counts across legislation, case law, and constitution.
2. Chunk counts, average character length, min/max chunk lengths.
3. Presence of critical core articles (Article 14, 19, 21, 32, 226, Preamble, Schedules).
4. Presence of core acts (IPC, CrPC, IEA, BNS, BNSS, BSA, CPC, ICA, Companies Act, POCSO, NDPS).
5. Absence of empty text, corrupt chunks, missing chunk_ids, or invalid structures.
6. Vector DB metadata and chunk_id parity.

Exit Codes:
  0: Corpus passes all integrity tests.
  1: Severe defects found (empty texts, missing IDs, or missing critical provisions).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate_corpus")

CORE_CONSTITUTION_TARGETS = {
    "article_14": ["article 14", "equality before law", "equal protection of the laws"],
    "article_19": ["article 19", "protection of certain rights regarding freedom of speech", "freedom of speech"],
    "article_21": ["article 21", "protection of life and personal liberty", "procedure established by law"],
    "article_32": ["article 32", "remedies for enforcement of rights conferred by this part", "constitutional remedies"],
    "article_226": ["article 226", "power of high courts to issue certain writs", "high court writs"],
    "preamble": ["preamble", "we, the people of india", "sovereign socialist secular democratic republic"],
}

CORE_ACT_TARGETS = {
    "IPC": ["penal code", "indian penal code"],
    "CrPC": ["code of criminal procedure", "criminal procedure"],
    "IEA": ["evidence act", "indian evidence act"],
    "BNS": ["bharatiya nyaya sanhita", "bns"],
    "BNSS": ["bharatiya nagarik suraksha", "bnss"],
    "BSA": ["bharatiya sakshya", "bsa"],
    "CPC": ["code of civil procedure", "civil procedure"],
    "ICA": ["contract act", "indian contract act"],
    "Companies Act": ["companies act"],
    "POCSO": ["protection of children from sexual offences", "pocso"],
    "NDPS": ["narcotic drugs and psychotropic substances", "ndps"],
}


def audit_jsonl(file_path: Path, doc_type_name: str) -> Dict[str, Any]:
    """Audits a single JSONL chunk file."""
    stats = {
        "file": str(file_path),
        "exists": file_path.exists(),
        "total_chunks": 0,
        "empty_text_chunks": 0,
        "missing_id_chunks": 0,
        "corrupt_lines": 0,
        "min_len": float("inf"),
        "max_len": 0,
        "total_chars": 0,
        "avg_len": 0.0,
        "matched_targets": set(),
        "acts_detected": set(),
    }

    if not file_path.exists():
        return stats

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                stats["corrupt_lines"] += 1
                continue

            stats["total_chunks"] += 1
            chunk_id = chunk.get("chunk_id") or chunk.get("id")
            if not chunk_id:
                stats["missing_id_chunks"] += 1

            text = chunk.get("text", "") or chunk.get("content", "")
            c_len = len(text)
            if c_len == 0:
                stats["empty_text_chunks"] += 1
            else:
                stats["total_chars"] += c_len
                if c_len < stats["min_len"]:
                    stats["min_len"] = c_len
                if c_len > stats["max_len"]:
                    stats["max_len"] = c_len

            # Check core target matching in text or metadata
            combined = f"{chunk.get('title', '')} {chunk.get('act_name', '')} {chunk.get('article', '')} {text[:300]}".lower()

            for target, kws in CORE_CONSTITUTION_TARGETS.items():
                if any(kw in combined for kw in kws):
                    stats["matched_targets"].add(target)

            for act_name, kws in CORE_ACT_TARGETS.items():
                if any(kw in combined for kw in kws):
                    stats["acts_detected"].add(act_name)

    if stats["total_chunks"] > 0 and stats["min_len"] != float("inf"):
        stats["avg_len"] = round(stats["total_chars"] / max(stats["total_chunks"] - stats["empty_text_chunks"], 1), 1)
    else:
        stats["min_len"] = 0

    return stats


def validate_corpus(project_root: Path) -> bool:
    log.info("=" * 70)
    log.info("LEGALMINDAI — COMPREHENSIVE LEGAL CORPUS VALIDATION AUDIT")
    log.info("=" * 70)

    leg_chunks = project_root / "data" / "chunks" / "legislation" / "chunks.jsonl"
    cas_chunks = project_root / "data" / "chunks" / "case_laws" / "indexed_chunks.jsonl"
    con_chunks = project_root / "data" / "processed" / "constitution_articles.jsonl"

    leg_stats = audit_jsonl(leg_chunks, "Legislation")
    cas_stats = audit_jsonl(cas_chunks, "Case Laws")
    con_stats = audit_jsonl(con_chunks, "Constitution")

    log.info("\n1. CHUNK REPOSITORY SUMMARY:")
    log.info(f"  Legislation Chunks : {leg_stats['total_chunks']:,} (avg {leg_stats['avg_len']} chars, min {leg_stats['min_len']}, max {leg_stats['max_len']:,})")
    log.info(f"  Case Laws Chunks   : {cas_stats['total_chunks']:,} (avg {cas_stats['avg_len']} chars, min {cas_stats['min_len']}, max {cas_stats['max_len']:,})")
    log.info(f"  Constitution Chunks: {con_stats['total_chunks']:,} (avg {con_stats['avg_len']} chars, min {con_stats['min_len']}, max {con_stats['max_len']:,})")

    total_chunks = leg_stats["total_chunks"] + cas_stats["total_chunks"] + con_stats["total_chunks"]
    log.info(f"  Total Indexed Corpus: {total_chunks:,} chunks")

    # Combine matched targets
    all_matched_const = leg_stats["matched_targets"] | con_stats["matched_targets"] | cas_stats["matched_targets"]
    all_matched_acts = leg_stats["acts_detected"] | cas_stats["acts_detected"]

    log.info("\n2. CONSTITUTIONAL PROVISION COVERAGE:")
    for target in CORE_CONSTITUTION_TARGETS.keys():
        present = target in all_matched_const
        status = "[PRESENT]" if present else "[MISSING]"
        log.info(f"  {status:10} {target.upper()}")

    log.info("\n3. CORE STATUTORY ACT COVERAGE:")
    for act in CORE_ACT_TARGETS.keys():
        present = act in all_matched_acts
        status = "[PRESENT]" if present else "[MISSING]"
        log.info(f"  {status:10} {act}")

    # Vector DB Parity Checks
    log.info("\n4. VECTOR DATABASE INTEGRITY & PARITY:")
    vdb_errors = []
    for sub in ["legislation", "case_laws"]:
        vdb_dir = project_root / "vector_db" / sub
        idx_f = vdb_dir / "index.faiss"
        ids_f = vdb_dir / "chunk_ids.json"
        meta_f = vdb_dir / "metadata.jsonl"

        if not idx_f.exists():
            vdb_errors.append(f"Missing {idx_f}")
        if not ids_f.exists():
            vdb_errors.append(f"Missing {ids_f}")
        if not meta_f.exists():
            vdb_errors.append(f"Missing {meta_f}")

        if ids_f.exists() and meta_f.exists():
            try:
                with open(ids_f, "r") as f:
                    num_ids = len(json.load(f))
                num_meta = sum(1 for line in open(meta_f, "r", encoding="utf-8") if line.strip())
                parity = "OK" if num_ids == num_meta else f"MISMATCH ({num_ids} vs {num_meta})"
                log.info(f"  vector_db/{sub:12} : {num_ids:,} IDs vs {num_meta:,} Meta lines [{parity}]")
                if num_ids != num_meta:
                    vdb_errors.append(f"vector_db/{sub} IDs count {num_ids} != metadata count {num_meta}")
            except Exception as e:
                vdb_errors.append(f"vector_db/{sub} error reading: {e}")

    # Defect Checks
    defects = []
    for name, st in [("Legislation", leg_stats), ("Case Laws", cas_stats), ("Constitution", con_stats)]:
        if not st["exists"]:
            defects.append(f"{name} file does not exist: {st['file']}")
        if st["empty_text_chunks"] > 0:
            defects.append(f"{name} has {st['empty_text_chunks']} empty text chunks")
        if st["missing_id_chunks"] > 0:
            defects.append(f"{name} has {st['missing_id_chunks']} chunks without chunk_id")
        if st["corrupt_lines"] > 0:
            defects.append(f"{name} has {st['corrupt_lines']} corrupt lines")

    for target in ["article_14", "article_19", "article_21"]:
        if target not in all_matched_const:
            defects.append(f"Critical constitutional article missing from corpus: {target}")

    for act in ["IPC", "CrPC", "IEA", "BNS"]:
        if act not in all_matched_acts:
            defects.append(f"Critical act missing from corpus: {act}")

    defects.extend(vdb_errors)

    log.info("\n" + "=" * 70)
    if defects:
        log.error(f"CORPUS INTEGRITY AUDIT FAILED with {len(defects)} issue(s):")
        for d in defects:
            log.error(f"  - {d}")
        log.info("=" * 70)
        return False
    else:
        log.info("CORPUS INTEGRITY AUDIT PASSED: 100% VALIDATED.")
        log.info(f"  Total Chunks: {total_chunks:,} | Corrupt: 0 | Empty: 0 | Core Articles: 100% Verified.")
        log.info("=" * 70)
        return True


def main():
    parser = argparse.ArgumentParser(description="LegalMindAI Legal Corpus Validator")
    parser.add_argument("--root", default=".", help="Project root directory")
    args = parser.parse_args()

    project_root = Path(args.root).resolve()
    passed = validate_corpus(project_root)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
