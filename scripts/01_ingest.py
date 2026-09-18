#!/usr/bin/env python3
"""
scripts/01_ingest.py
====================
LegalMind AI — Legal Document Ingestion

Ingests legal documents (PDF, TXT, HTML) from a source directory,
extracts structured text while preserving legal metadata, and
saves cleaned records as JSONL to the output directory.

Usage:
    python scripts/01_ingest.py --input data/raw/ --output data/processed/
    python scripts/01_ingest.py --input data/raw/constitution.pdf --output data/processed/

Supported document types:
    - PDF   (via PyMuPDF / pypdf fallback)
    - TXT   (plain text)
    - HTML  (via BeautifulSoup)

Output format (JSONL):
    {
      "document_id": "...",
      "document_type": "act" | "judgment" | "constitution" | "regulation" | "unknown",
      "title": "...",
      "court": "...",        # judgments only
      "date": "...",         # judgments only
      "citation": "...",     # judgments only
      "source": "filename",
      "text": "..."
    }
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest")

# ---------------------------------------------------------------------------
# Optional imports — handled gracefully
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    log.warning("PyMuPDF (fitz) not available — will try pypdf fallback.")

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    log.warning("BeautifulSoup4 not available — HTML support disabled.")

# ---------------------------------------------------------------------------
# Regex patterns for Indian legal document classification
# ---------------------------------------------------------------------------

# Judgment metadata patterns
_RE_CASE_NAME   = re.compile(r"(?:IN THE|BEFORE)[^\n]*\n(.+?)\s*vs?\.?\s*(.+?)(?:\n|$)", re.IGNORECASE)
_RE_DATE        = re.compile(r"(?:Date[d\s:]*|Decided\s*on\s*:?\s*)(\d{1,2}[\s/-]\w+[\s/-]\d{4}|\d{4}-\d{2}-\d{2})", re.IGNORECASE)
_RE_CITATION    = re.compile(r"\((\d{4})\)\s+\d+\s+SCC\s+\d+|\d{4}\s+AIR\s+SC\s+\d+|\d{4}\s+SCR\s+\d+", re.IGNORECASE)
_RE_COURT       = re.compile(r"(Supreme Court of India|High Court of [A-Za-z\s]+|District Court[^,\n]*)", re.IGNORECASE)
_RE_CASE_NO     = re.compile(r"(?:Civil|Criminal|Writ|Special Leave)?\s*(?:Appeal|Petition|Application|Suit)\s*(?:No\.?|Number)\s*[\d/]+\s*(?:of\s+\d{4})?", re.IGNORECASE)

# Act / legislation patterns
_RE_ACT_NAME    = re.compile(r"THE\s+([A-Z][A-Z\s,]+(?:ACT|CODE|RULES|REGULATIONS|ORDINANCE|AMENDMENT)[\s,\d]*)", re.IGNORECASE)
_RE_SECTION     = re.compile(r"(?:^|\n)\s*(?:Section|Sec\.|S\.)\s*(\d+[A-Z]?(?:\.\d+[A-Z]?)*)", re.IGNORECASE | re.MULTILINE)
_RE_CHAPTER     = re.compile(r"(?:^|\n)\s*CHAPTER\s+([IVXLCDM]+|\d+)\s*[-—:]?\s*(.{0,80})", re.IGNORECASE | re.MULTILINE)
_RE_ARTICLE     = re.compile(r"(?:^|\n)\s*Article\s+(\d+[A-Z]?)", re.IGNORECASE | re.MULTILINE)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalise whitespace and unicode, remove control characters."""
    text = unicodedata.normalize("NFKC", text)
    # Remove null bytes and other control chars except newlines/tabs
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc" or ch in "\n\t")
    # Collapse excessive blank lines (keep max 2 consecutive newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse horizontal whitespace runs
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def make_document_id(source: str, text: str) -> str:
    """Deterministic document ID based on source name + content hash."""
    h = hashlib.sha256(f"{source}::{text[:1000]}".encode()).hexdigest()[:12]
    stem = Path(source).stem[:40]
    return f"{stem}_{h}"


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf_pymupdf(path: Path) -> str:
    """Extract text from PDF using PyMuPDF (preferred)."""
    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


def extract_pdf_pypdf(path: Path) -> str:
    """Extract text from PDF using pypdf (fallback)."""
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages.append(t)
    return "\n".join(pages)


def extract_pdf(path: Path) -> str:
    if HAS_PYMUPDF:
        try:
            return extract_pdf_pymupdf(path)
        except Exception as e:
            log.warning(f"PyMuPDF failed for {path.name}: {e}. Trying pypdf…")
    if HAS_PYPDF:
        return extract_pdf_pypdf(path)
    raise RuntimeError(f"No PDF parser available to process {path}")


# ---------------------------------------------------------------------------
# TXT extraction
# ---------------------------------------------------------------------------

def extract_txt(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Could not decode {path} with any supported encoding.")


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------

def extract_html(path: Path) -> str:
    if not HAS_BS4:
        raise RuntimeError("BeautifulSoup4 not installed — cannot process HTML files.")
    raw = extract_txt(path)
    soup = BeautifulSoup(raw, "lxml" if _lxml_available() else "html.parser")
    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _lxml_available() -> bool:
    try:
        import lxml
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Document type classification
# ---------------------------------------------------------------------------

_JUDGMENT_KEYWORDS = {
    "judgment", "judgement", "order", "petition", "appellant",
    "respondent", "plaintiff", "defendant", "writ", "bench",
    "coram", "hon'ble", "decided", "disposed", "appeal", "decree",
}

_ACT_KEYWORDS = {
    "section", "sub-section", "clause", "schedule", "chapter",
    "amendment", "ordinance", "regulation", "rules", "hereby enacted",
    "be it enacted", "act, 19", "act, 20",
}

_CONSTITUTION_KEYWORDS = {
    "article", "part", "schedule", "fundamental rights",
    "directive principles", "parliament", "president of india",
    "constitution of india",
}


def classify_document(filename: str, text: str) -> str:
    """Heuristically classify document type."""
    fname_lower = filename.lower()
    text_lower = text[:3000].lower()

    if "constitution" in fname_lower:
        return "constitution"

    hits_judgment = sum(1 for kw in _JUDGMENT_KEYWORDS if kw in text_lower)
    hits_act = sum(1 for kw in _ACT_KEYWORDS if kw in text_lower)
    hits_constitution = sum(1 for kw in _CONSTITUTION_KEYWORDS if kw in text_lower)

    if hits_constitution >= 4 and "article" in text_lower:
        return "constitution"
    if hits_judgment >= 5:
        return "judgment"
    if hits_act >= 4:
        return "act"
    return "unknown"


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_judgment_metadata(text: str) -> dict[str, str]:
    meta: dict[str, str] = {}

    # Court
    m = _RE_COURT.search(text[:2000])
    if m:
        meta["court"] = m.group(1).strip()

    # Date
    m = _RE_DATE.search(text[:3000])
    if m:
        meta["date"] = m.group(1).strip()

    # Citation
    m = _RE_CITATION.search(text[:5000])
    if m:
        meta["citation"] = m.group(0).strip()

    # Case number
    m = _RE_CASE_NO.search(text[:3000])
    if m:
        meta["case_number"] = m.group(0).strip()

    return meta


def extract_act_metadata(text: str, filename: str) -> dict[str, str]:
    meta: dict[str, str] = {}

    m = _RE_ACT_NAME.search(text[:2000])
    if m:
        meta["act_name"] = m.group(0).strip()
    else:
        # Fall back to filename
        meta["act_name"] = Path(filename).stem.replace("_", " ").title()

    return meta


def extract_title(text: str, filename: str, doc_type: str) -> str:
    """Best-effort title extraction."""
    # First non-empty line of the document
    for line in text.split("\n"):
        line = line.strip()
        if len(line) > 10:
            return line[:200]
    return Path(filename).stem.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def ingest_file(path: Path) -> dict[str, Any] | None:
    """Process a single file and return a document record."""
    suffix = path.suffix.lower()

    try:
        if suffix == ".pdf":
            raw_text = extract_pdf(path)
        elif suffix == ".txt":
            raw_text = extract_txt(path)
        elif suffix in (".html", ".htm"):
            raw_text = extract_html(path)
        else:
            log.debug(f"Skipping unsupported file type: {path}")
            return None
    except Exception as e:
        log.error(f"Failed to extract text from {path.name}: {e}")
        return None

    text = clean_text(raw_text)

    if len(text) < 50:
        log.warning(f"Skipping near-empty document: {path.name} ({len(text)} chars)")
        return None

    doc_type = classify_document(path.name, text)
    doc_id   = make_document_id(path.name, text)
    title    = extract_title(text, path.name, doc_type)

    record: dict[str, Any] = {
        "document_id":   doc_id,
        "document_type": doc_type,
        "title":         title,
        "source":        path.name,
        "source_path":   str(path.resolve()),
        "char_count":    len(text),
        "text":          text,
    }

    # Enrich with type-specific metadata
    if doc_type == "judgment":
        record.update(extract_judgment_metadata(text))
    elif doc_type in ("act", "constitution"):
        record.update(extract_act_metadata(text, path.name))

    return record


def ingest_directory(input_path: Path, output_path: Path, overwrite: bool = False) -> None:
    """Ingest all supported files from a directory."""
    SUPPORTED = {".pdf", ".txt", ".html", ".htm"}

    files = [f for f in input_path.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED]
    log.info(f"Found {len(files)} file(s) to ingest in {input_path}")

    if not files:
        log.warning("No supported files found. Place PDFs/TXTs/HTMLs in the input directory.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    out_jsonl = output_path / "documents.jsonl"
    metadata_out = output_path / "metadata_index.json"

    metadata_index = []
    written = 0
    skipped = 0

    mode = "w" if overwrite else "a"
    with open(out_jsonl, mode, encoding="utf-8") as fout:
        for i, fpath in enumerate(sorted(files), 1):
            log.info(f"[{i}/{len(files)}] Ingesting: {fpath.name}")
            record = ingest_file(fpath)
            if record is None:
                skipped += 1
                continue
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            # Build metadata index (exclude full text for smaller index)
            meta = {k: v for k, v in record.items() if k != "text"}
            metadata_index.append(meta)
            written += 1
            log.info(
                f"  ✓ [{record['document_type']:12s}] {record['title'][:60]} "
                f"({record['char_count']:,} chars)"
            )

    # Write metadata index
    with open(metadata_out, "w", encoding="utf-8") as f:
        json.dump(metadata_index, f, ensure_ascii=False, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"Ingestion complete:")
    log.info(f"  Written : {written} document(s) → {out_jsonl}")
    log.info(f"  Skipped : {skipped} file(s)")
    log.info(f"  Index   : {metadata_out}")
    log.info(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LegalMind AI — Legal Document Ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input file or directory containing legal documents",
    )
    parser.add_argument(
        "--output", "-o", default="data/processed/",
        help="Output directory for JSONL documents (default: data/processed/)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output file (default: append)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        log.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    # Check PDF parsers
    if not HAS_PYMUPDF and not HAS_PYPDF:
        log.error("Neither PyMuPDF nor pypdf is installed. Cannot process PDFs.")
        log.error("Run: pip install pymupdf pypdf")
        sys.exit(1)

    if input_path.is_file():
        output_path.mkdir(parents=True, exist_ok=True)
        record = ingest_file(input_path)
        if record:
            out_file = output_path / "documents.jsonl"
            with open(out_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.info(f"Saved to {out_file}")
    else:
        ingest_directory(input_path, output_path, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
