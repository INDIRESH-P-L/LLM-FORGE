#!/usr/bin/env python3
"""
scripts/01b_preprocess_legislation.py
=====================================
Phase 2: Legislation Preprocessing

Reads the extracted acts from data/raw/legislation/selected_acts.jsonl,
cleans the text, enriches the metadata, and writes to
data/processed/legislation/cleaned_acts.jsonl.
"""

import json
import os
import re
import sys
from pathlib import Path

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Normalize whitespaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def process_legislation(input_path: Path, output_path: Path, log_path: Path):
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    skipped_count = 0
    act_counts = {}

    print(f"Processing legislation from: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_count += 1
                continue

            text = clean_text(record.get("text", ""))
            if len(text) < 20:
                skipped_count += 1
                continue

            act_name = record.get("act_name", "Unknown Act")
            section_no = record.get("section_number", "")
            section_title = record.get("section_title", "")
            
            # Enrich text with context for better RAG retrieval
            enriched_text = f"Act: {act_name}\n"
            if section_no:
                enriched_text += f"Section: {section_no}"
                if section_title:
                    enriched_text += f" - {section_title}"
                enriched_text += "\n"
            enriched_text += f"\n{text}"
            
            cleaned_record = {
                "chunk_id": record.get("chunk_id", ""),
                "act_name": act_name,
                "chapter": record.get("chapter", ""),
                "section_number": section_no,
                "section_title": section_title,
                "jurisdiction": record.get("jurisdiction", ""),
                "char_count": len(enriched_text),
                "text": enriched_text,
                "original_text": text
            }
            
            fout.write(json.dumps(cleaned_record, ensure_ascii=False) + "\n")
            processed_count += 1
            act_counts[act_name] = act_counts.get(act_name, 0) + 1

    print("Processing complete.")
    print(f"Total processed: {processed_count}")
    print(f"Total skipped (too short/invalid): {skipped_count}")
    
    # Generate report
    with open(log_path, "w", encoding="utf-8") as flog:
        flog.write("Legislation Preprocessing Report\n")
        flog.write("================================\n")
        flog.write(f"Input file: {input_path}\n")
        flog.write(f"Output file: {output_path}\n")
        flog.write(f"Total chunks processed: {processed_count}\n")
        flog.write(f"Total chunks skipped: {skipped_count}\n\n")
        flog.write("Breakdown by Act:\n")
        for act, count in sorted(act_counts.items(), key=lambda x: x[1], reverse=True):
            flog.write(f"  - {act}: {count}\n")

    print(f"Report saved to: {log_path}")

if __name__ == "__main__":
    base_dir = Path("/home/sece2026-student12/LegalMindAI")
    in_file = base_dir / "data" / "raw" / "legislation" / "selected_acts.jsonl"
    out_file = base_dir / "data" / "processed" / "legislation" / "cleaned_acts.jsonl"
    log_file = base_dir / "logs" / "legislation_preprocessing.txt"
    
    process_legislation(in_file, out_file, log_file)
