#!/usr/bin/env python3
"""
scripts/02_chunk.py
===================
LegalMind AI — Structure-Aware Legal Chunking

Reads processed documents (JSONL from 01_ingest.py) and splits them
into overlapping chunks that respect legal document structure.

Chunking strategy:
  - Constitution / Acts  : Chapter → Article/Section → Sub-section
  - Judgments            : Facts → Issues → Arguments → Reasoning → Decision
  - Unknown              : Paragraph-aware sliding window

Each chunk retains full metadata from the parent document.

Usage:
    python scripts/02_chunk.py --input data/processed/ --output data/chunks/
    python scripts/02_chunk.py --input data/processed/documents.jsonl --output data/chunks/

Configuration (CLI flags):
    --chunk-size     Target chunk size in characters (default: 1200)
    --overlap        Overlap between adjacent chunks in characters (default: 200)
    --min-chunk      Minimum chunk size to keep (default: 100)
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Generator

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chunk")

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 1200   # characters
DEFAULT_OVERLAP    = 200    # characters
MIN_CHUNK_SIZE     = 100    # characters


# ---------------------------------------------------------------------------
# Regex section markers
# ---------------------------------------------------------------------------

# Constitution / general legislation
_RE_ARTICLE   = re.compile(
    r"(?:^|\n)[\s]*(?:Article|ART\.?)\s+(\d+[A-Z]?(?:\s+[A-Z])?)\s*[.—\-:]?\s*(.{0,120})",
    re.IGNORECASE | re.MULTILINE,
)
_RE_SECTION   = re.compile(
    r"(?:^|\n)[\s]*(?:Section|Sec\.|S\.)\s*(\d+[A-Z]?(?:\.\d+[A-Z]?)*)\s*[.—\-:]?\s*(.{0,120})",
    re.IGNORECASE | re.MULTILINE,
)
_RE_CHAPTER   = re.compile(
    r"(?:^|\n)[\s]*CHAPTER\s+([IVXLCDM]+|\d+)\s*[-—:]?\s*(.{0,120})",
    re.IGNORECASE | re.MULTILINE,
)
_RE_PART      = re.compile(
    r"(?:^|\n)[\s]*PART\s+([IVXLCDM]+|[A-Z]|\d+)\s*[-—:]?\s*(.{0,120})",
    re.IGNORECASE | re.MULTILINE,
)

# Judgment structural markers
_RE_JUDGMENT_SECTIONS = re.compile(
    r"(?:^|\n)[\s]*(FACTS?|ISSUES?|ARGUMENTS?|CONTENTIONS?|REASONING|ANALYSIS|"
    r"DISCUSSION|FINDINGS?|DECISION|ORDER|JUDGMENT|HELD|OBSERVATIONS?|PRECEDENTS?|"
    r"BACKGROUND|INTRODUCTION)\s*[:\-—]?\s*(?:\n|$)",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def make_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    h = hashlib.sha256(f"{doc_id}:{chunk_index}:{text[:200]}".encode()).hexdigest()[:8]
    return f"{doc_id}_c{chunk_index:04d}_{h}"


def _base_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    """Extract inheritable metadata from parent document (no text field)."""
    keys = [
        "document_id", "document_type", "title", "court", "date",
        "citation", "case_number", "act_name", "source", "source_path",
    ]
    return {k: doc[k] for k in keys if k in doc}


def sliding_window_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
    min_size: int,
) -> list[str]:
    """
    Paragraph-aware sliding window chunker.
    Prefers to break at paragraph boundaries (double newlines),
    falls back to sentence boundaries, then hard character split.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph exceeds chunk_size, flush first
        if current and len(current) + len(para) + 2 > chunk_size:
            if len(current) >= min_size:
                chunks.append(current.strip())
            # Carry-over overlap
            sentences = re.split(r"(?<=[.!?])\s+", current)
            carry = ""
            for s in reversed(sentences):
                if len(carry) + len(s) + 1 <= overlap:
                    carry = s + " " + carry
                else:
                    break
            current = carry.strip()

        current = (current + "\n\n" + para).strip() if current else para

    if current.strip() and len(current.strip()) >= min_size:
        chunks.append(current.strip())

    # Handle over-sized paragraphs (hard split)
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size * 1.5:
            result.append(chunk)
        else:
            # Hard split at sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            sub = ""
            for sent in sentences:
                if sub and len(sub) + len(sent) > chunk_size:
                    if len(sub) >= min_size:
                        result.append(sub.strip())
                    sub = sent
                else:
                    sub = (sub + " " + sent).strip()
            if sub and len(sub) >= min_size:
                result.append(sub.strip())

    return result


# ---------------------------------------------------------------------------
# Structure-aware chunkers
# ---------------------------------------------------------------------------

def chunk_by_structure(
    text: str,
    pattern: re.Pattern,
    chunk_size: int,
    overlap: int,
    min_size: int,
    section_label: str = "section",
) -> Generator[dict[str, Any], None, None]:
    """
    Split text at structural boundaries identified by `pattern`.
    Each match becomes a candidate chunk header; text up to the
    next match forms the chunk body.
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return

    boundaries = [m.start() for m in matches] + [len(text)]

    for i, match in enumerate(matches):
        header_text = match.group(0).strip()
        body = text[match.end(): boundaries[i + 1]].strip()

        # Extract section number if captured
        groups = [g for g in match.groups() if g]
        section_num   = groups[0].strip() if groups else ""
        section_title = groups[1].strip() if len(groups) > 1 else ""

        combined = (header_text + "\n\n" + body).strip()

        if len(combined) <= chunk_size * 1.5:
            # Fits in one chunk
            if len(combined) >= min_size:
                yield {
                    "text": combined,
                    section_label: section_num,
                    "section_title": section_title,
                }
        else:
            # Sub-chunk the body with sliding window
            sub_chunks = sliding_window_chunks(body, chunk_size, overlap, min_size)
            for j, sub in enumerate(sub_chunks):
                yield {
                    "text": (header_text + "\n\n" + sub).strip() if j == 0 else sub,
                    section_label: section_num,
                    "section_title": section_title,
                    "sub_chunk": j,
                }


def chunk_constitution(
    text: str,
    chunk_size: int,
    overlap: int,
    min_size: int,
) -> list[dict[str, Any]]:
    """Article-level chunking for Constitution / mixed Acts."""
    chunks = []
    # Try article-level first, fall back to section-level
    article_chunks = list(chunk_by_structure(text, _RE_ARTICLE, chunk_size, overlap, min_size, "article"))
    if article_chunks:
        chunks = article_chunks
    else:
        chunks = list(chunk_by_structure(text, _RE_SECTION, chunk_size, overlap, min_size, "section"))

    if not chunks:
        # Fall back to sliding window
        for t in sliding_window_chunks(text, chunk_size, overlap, min_size):
            chunks.append({"text": t})

    return chunks


def chunk_act(
    text: str,
    chunk_size: int,
    overlap: int,
    min_size: int,
) -> list[dict[str, Any]]:
    """Section-level chunking for Acts/Regulations."""
    chunks = list(chunk_by_structure(text, _RE_SECTION, chunk_size, overlap, min_size, "section"))
    if not chunks:
        # Try chapter-level
        chunks = list(chunk_by_structure(text, _RE_CHAPTER, chunk_size, overlap, min_size, "chapter"))
    if not chunks:
        for t in sliding_window_chunks(text, chunk_size, overlap, min_size):
            chunks.append({"text": t})
    return chunks


def chunk_judgment(
    text: str,
    chunk_size: int,
    overlap: int,
    min_size: int,
) -> list[dict[str, Any]]:
    """Judgment-section-level chunking (Facts/Issues/Reasoning/Decision)."""
    chunks = list(chunk_by_structure(
        text, _RE_JUDGMENT_SECTIONS, chunk_size, overlap, min_size, "judgment_section"
    ))
    if not chunks:
        for t in sliding_window_chunks(text, chunk_size, overlap, min_size):
            chunks.append({"text": t})
    return chunks


def chunk_generic(
    text: str,
    chunk_size: int,
    overlap: int,
    min_size: int,
) -> list[dict[str, Any]]:
    """Sliding window fallback for unknown document types."""
    return [{"text": t} for t in sliding_window_chunks(text, chunk_size, overlap, min_size)]


# ---------------------------------------------------------------------------
# Document → chunks dispatcher
# ---------------------------------------------------------------------------

def chunk_document(
    doc: dict[str, Any],
    chunk_size: int,
    overlap: int,
    min_size: int,
) -> list[dict[str, Any]]:
    """Dispatch to the appropriate chunker based on document_type."""
    doc_type = doc.get("document_type", "unknown")
    text = doc.get("text", "")
    base = _base_metadata(doc)

    if doc_type == "constitution":
        raw_chunks = chunk_constitution(text, chunk_size, overlap, min_size)
    elif doc_type == "act":
        raw_chunks = chunk_act(text, chunk_size, overlap, min_size)
    elif doc_type == "judgment":
        raw_chunks = chunk_judgment(text, chunk_size, overlap, min_size)
    else:
        raw_chunks = chunk_generic(text, chunk_size, overlap, min_size)

    result = []
    for i, chunk_data in enumerate(raw_chunks):
        chunk_text = chunk_data.pop("text", "")
        if not chunk_text or len(chunk_text) < min_size:
            continue

        chunk_id = make_chunk_id(doc.get("document_id", "doc"), i, chunk_text)

        record: dict[str, Any] = {
            "chunk_id":      chunk_id,
            "chunk_index":   i,
            "char_count":    len(chunk_text),
            "text":          chunk_text,
        }
        record.update(base)       # inherit parent metadata
        record.update(chunk_data) # add structure-specific fields (section, article, etc.)

        result.append(record)

    return result


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------

def process_jsonl(
    input_file: Path,
    output_dir: Path,
    chunk_size: int,
    overlap: int,
    min_size: int,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "chunks.jsonl"

    mode = "w" if overwrite else "a"
    total_docs = 0
    total_chunks = 0

    with open(input_file, encoding="utf-8") as fin, \
         open(out_file, mode, encoding="utf-8") as fout:

        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"Skipping malformed JSON on line {line_no}: {e}")
                continue

            chunks = chunk_document(doc, chunk_size, overlap, min_size)
            for chunk in chunks:
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            total_docs += 1
            total_chunks += len(chunks)
            log.info(
                f"  [{total_docs:>4d}] {doc.get('title','?')[:55]:<55} "
                f"→ {len(chunks):>3d} chunk(s)"
            )

    log.info(f"\n{'='*60}")
    log.info(f"Chunking complete:")
    log.info(f"  Documents : {total_docs}")
    log.info(f"  Chunks    : {total_chunks}")
    log.info(f"  Output    : {out_file}")
    log.info(f"  Chunk size: {chunk_size} chars, overlap: {overlap} chars")
    log.info(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LegalMind AI — Structure-Aware Legal Chunking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Input JSONL file or directory with documents.jsonl")
    parser.add_argument("--output", "-o", default="data/chunks/",
                        help="Output directory for chunks.jsonl (default: data/chunks/)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Target chunk size in characters (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                        help=f"Overlap between chunks in characters (default: {DEFAULT_OVERLAP})")
    parser.add_argument("--min-chunk", type=int, default=MIN_CHUNK_SIZE,
                        help=f"Minimum chunk size to retain (default: {MIN_CHUNK_SIZE})")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output file")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path  = Path(args.input)
    output_path = Path(args.output)

    # Resolve input to a JSONL file
    if input_path.is_dir():
        jsonl_file = input_path / "documents.jsonl"
        if not jsonl_file.exists():
            log.error(f"No documents.jsonl found in {input_path}")
            log.error("Run 01_ingest.py first.")
            sys.exit(1)
    elif input_path.is_file() and input_path.suffix == ".jsonl":
        jsonl_file = input_path
    else:
        log.error(f"Input must be a JSONL file or directory containing documents.jsonl")
        sys.exit(1)

    log.info(f"Chunking: {jsonl_file}")
    log.info(f"Config  : chunk_size={args.chunk_size}, overlap={args.overlap}, min={args.min_chunk}")

    process_jsonl(
        input_file  = jsonl_file,
        output_dir  = output_path,
        chunk_size  = args.chunk_size,
        overlap     = args.overlap,
        min_size    = args.min_chunk,
        overwrite   = args.overwrite,
    )


if __name__ == "__main__":
    main()
