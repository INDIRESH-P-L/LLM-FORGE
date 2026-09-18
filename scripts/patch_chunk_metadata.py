#!/usr/bin/env python3
"""
scripts/patch_chunk_metadata.py
================================
One-time patch script to add missing citation metadata fields to existing
legislation chunks.jsonl so that retriever.py / 05_rag.py can format
proper citations instead of "Unknown Document".

This script DOES NOT re-run chunking. It patches in-place (writes a new file
then replaces the original).

Legislation chunks schema after patch:
  - document_id     → derived from act_name + chunk_id prefix
  - document_type   → "act" or "constitution"
  - title           → act_name
  - source          → "data/processed/legislation/cleaned_acts.jsonl"

Run:
    python scripts/patch_chunk_metadata.py
"""
import json
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("patch")

LEGISLATION_CHUNKS = Path("data/chunks/legislation/chunks.jsonl")
BACKUP = Path("data/chunks/legislation/chunks.jsonl.bak")

# Documents whose act_name contains these strings → constitution type
CONSTITUTION_KEYWORDS = ["constitution of india", "constitution"]


def classify_doc_type(act_name: str) -> str:
    an_lower = act_name.lower()
    for kw in CONSTITUTION_KEYWORDS:
        if kw in an_lower:
            return "constitution"
    return "act"


def patch_legislation_chunks():
    if not LEGISLATION_CHUNKS.exists():
        log.error(f"File not found: {LEGISLATION_CHUNKS}")
        return

    # Backup original
    if not BACKUP.exists():
        shutil.copy2(LEGISLATION_CHUNKS, BACKUP)
        log.info(f"Backup created: {BACKUP}")
    else:
        log.info(f"Backup already exists: {BACKUP} — skipping backup")

    tmp = LEGISLATION_CHUNKS.with_suffix(".jsonl.tmp")
    patched = skipped = 0

    with open(LEGISLATION_CHUNKS, encoding="utf-8") as fin, \
         open(tmp, "w", encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"Line {line_no}: JSON error — {e}")
                skipped += 1
                continue

            act_name = chunk.get("act_name", "")

            # Patch in missing fields (don't overwrite if already present)
            if "document_type" not in chunk:
                chunk["document_type"] = classify_doc_type(act_name)
            if "title" not in chunk:
                chunk["title"] = act_name or "Unknown Act"
            if "source" not in chunk:
                chunk["source"] = "data/processed/legislation/cleaned_acts.jsonl"
            if "document_id" not in chunk:
                # Use act_name normalised as document_id base
                doc_id_base = act_name.lower().replace(" ", "_").replace(",", "")[:40]
                chunk["document_id"] = doc_id_base or chunk.get("chunk_id", "leg_doc")

            # Ensure chunk_id exists
            if "chunk_id" not in chunk:
                chunk["chunk_id"] = f"leg_chunk_{line_no}"

            fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            patched += 1

            if patched % 10000 == 0:
                log.info(f"  Patched {patched:,} chunks…")

    # Replace original with patched version
    shutil.move(str(tmp), str(LEGISLATION_CHUNKS))
    log.info(f"\nDone! Patched: {patched:,} chunks | Skipped: {skipped}")
    log.info(f"Output: {LEGISLATION_CHUNKS}")


if __name__ == "__main__":
    patch_legislation_chunks()
