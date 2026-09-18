#!/usr/bin/env python3
"""
scripts/01c_preprocess_caselaws.py
===================================
Phase 3 & 4: Case-Law Preprocessing

A memory-efficient pipeline for processing the 10.9 GB Indian case-law dataset.
Reads parquet files in batches using PyArrow, cleans the text, extracts 
important metadata, and writes to a structured JSONL for chunking.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("caselaws")

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def process_caselaws(input_dir: Path, output_file: Path, log_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    parquet_files = sorted(input_dir.rglob("*.parquet"))
    log.info(f"Found {len(parquet_files)} parquet files to process.")
    
    total_processed = 0
    total_skipped = 0
    batch_size = 1000  # Record batch size for memory efficiency
    
    # We will write everything sequentially to a single large JSONL or multiple.
    # To be memory efficient and robust, we'll write to a single file but 
    # could rotate if needed. For now, a single JSONL stream is fine.
    
    with open(output_file, "w", encoding="utf-8") as fout:
        for p_file in parquet_files:
            log.info(f"Processing: {p_file.name}")
            try:
                pf = pq.ParquetFile(p_file)
                for batch in pf.iter_batches(batch_size=batch_size):
                    df = batch.to_pandas()
                    
                    for _, row in df.iterrows():
                        # Extract necessary fields
                        case_title = str(row.get("case_title", ""))
                        court_name = str(row.get("court_name", ""))
                        decision_date = str(row.get("decision_date", ""))
                        idx_text = str(row.get("indexable_text", ""))
                        headnote = str(row.get("headnote_text", ""))
                        
                        # Use indexable text, fallback to headnote
                        raw_text = idx_text if idx_text and idx_text.strip() != "None" else headnote
                        raw_text = raw_text if raw_text and raw_text.strip() != "None" else ""
                        
                        text = clean_text(raw_text)
                        
                        if len(text) < 50:
                            total_skipped += 1
                            continue
                            
                        # Create a standardized document record 
                        # matching what 02_chunk.py expects
                        record = {
                            "document_id": str(row.get("id", "")),
                            "document_type": "judgment",
                            "title": case_title,
                            "court": court_name,
                            "date": decision_date,
                            "citation": str(row.get("neutral_citation", "")),
                            "case_number": str(row.get("docket_number", "")),
                            "source": p_file.name,
                            "text": text,
                        }
                        
                        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_processed += 1
                        
            except Exception as e:
                log.error(f"Error processing {p_file.name}: {e}")
                
    log.info(f"Processing complete. Processed: {total_processed}, Skipped: {total_skipped}")
    
    with open(log_file, "w", encoding="utf-8") as flog:
        flog.write("Case-Law Preprocessing Report\n")
        flog.write("=============================\n")
        flog.write(f"Total Parquet Files: {len(parquet_files)}\n")
        flog.write(f"Total Judgments Processed: {total_processed}\n")
        flog.write(f"Total Judgments Skipped (too short): {total_skipped}\n")
        flog.write(f"Output File: {output_file}\n")
        
if __name__ == "__main__":
    base_dir = Path("/home/sece2026-student12/LegalMindAI")
    in_dir = base_dir / "data" / "raw" / "case_laws"
    out_file = base_dir / "data" / "processed" / "case_laws" / "documents.jsonl"
    log_file = base_dir / "logs" / "caselaws_preprocessing.txt"
    
    process_caselaws(in_dir, out_file, log_file)
