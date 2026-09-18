#!/usr/bin/env python3
"""
scripts/07_build_authority_table.py
===================================
Build the authority table the citation verifier checks answers against.

This is the ground truth for "does this case/citation/provision actually
exist in our source data": every distinct judgment in the Supreme Court
corpus (case name, reporter citation, court, decision date, judges) plus
every statutory provision present in the legislation corpus.

It is deliberately independent of the FAISS index — verification must work
even when retrieval returns nothing, and must not depend on a 6-hour
embedding job.

Output: data/verification/authorities.db (SQLite + FTS5)

    .venv/bin/python scripts/07_build_authority_table.py
"""

from __future__ import annotations
import json, logging, re, sqlite3, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SC = ROOT / "data/raw/open-india-law/in_supreme-court_judgments.parquet"
LEG_CHUNKS = ROOT / "data/chunks/legislation/chunks.jsonl"
OUT = ROOT / "data/verification/authorities.db"

log = logging.getLogger("authority")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")


def norm_citation(c: str) -> str:
    """
    Canonical form for comparing reporter citations.

    "[1979] 3 S.C.R. 453", "(1979) 3 SCR 453" and "1979 3 scr 453" are the
    same authority; only the digits, the reporter letters and their order
    carry meaning.
    """
    if not c:
        return ""
    c = c.lower()
    c = re.sub(r"[.\s]", "", c)          # S.C.R. -> scr
    c = re.sub(r"[\[\](){}]", "", c)     # brackets are presentational
    return c


def norm_name(n: str) -> str:
    """Canonical case name: drop honorifics, punctuation, and v./vs variants."""
    if not n:
        return ""
    n = n.lower()
    n = re.sub(r"\b(?:m/s|shri|smt|sri|the state of|state of)\b", " ", n)
    n = re.sub(r"\s+(?:v|vs|versus)\.?\s+", " v ", n)
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    n = re.sub(r"\b(?:ors|anr|others|another|etc|dead|thr|lrs|d)\b", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    conn = sqlite3.connect(str(OUT))
    conn.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE judgments (
            case_id       TEXT PRIMARY KEY,
            title         TEXT NOT NULL,
            title_norm    TEXT NOT NULL,
            citation      TEXT,
            citation_norm TEXT,
            court         TEXT,
            decision_date TEXT,
            reporter_year INTEGER,
            judges        TEXT
        );
        CREATE INDEX idx_j_citation ON judgments(citation_norm);
        CREATE INDEX idx_j_title    ON judgments(title_norm);
        CREATE TABLE provisions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            act       TEXT NOT NULL,
            act_norm  TEXT NOT NULL,
            kind      TEXT NOT NULL,   -- section | article | order | rule
            number    TEXT NOT NULL,
            source    TEXT
        );
        CREATE INDEX idx_p_lookup ON provisions(kind, number, act_norm);
        CREATE VIRTUAL TABLE judgments_fts USING fts5(
            title, citation, content='judgments', content_rowid='rowid');
    """)

    # ── judgments ────────────────────────────────────────────────────────
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(SC)
    seen, rows, t0 = set(), [], time.time()
    cols = ["case_id", "title", "citation", "court", "decision_date", "year", "judges"]
    for batch in pf.iter_batches(columns=cols, batch_size=100_000):
        for r in batch.to_pylist():
            cid = r.get("case_id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            j = r.get("judges")
            if isinstance(j, (list, tuple)):
                j = ", ".join(str(x) for x in j if x)
            title = (r.get("title") or "").strip()
            cit = (r.get("citation") or "").strip()
            if cit.lower() in ("nan", "none", "null"):
                cit = ""
            rows.append((cid, title, norm_name(title), cit or None, norm_citation(cit) or None,
                         (r.get("court") or "").strip() or None,
                         str(r.get("decision_date") or "")[:10] or None,
                         r.get("year"), (j or "").strip() or None))
        if len(rows) >= 50_000:
            conn.executemany("INSERT OR IGNORE INTO judgments VALUES (?,?,?,?,?,?,?,?,?)", rows)
            conn.commit(); rows = []
    if rows:
        conn.executemany("INSERT OR IGNORE INTO judgments VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    n_j = conn.execute("SELECT COUNT(*) FROM judgments").fetchone()[0]
    n_c = conn.execute("SELECT COUNT(DISTINCT citation_norm) FROM judgments "
                       "WHERE citation_norm IS NOT NULL").fetchone()[0]
    log.info(f"judgments: {n_j:,} distinct ({n_c:,} distinct reporter citations) "
             f"in {time.time()-t0:.0f}s")

    conn.execute("INSERT INTO judgments_fts(rowid, title, citation) "
                 "SELECT rowid, title, citation FROM judgments")
    conn.commit()

    # ── statutory provisions from the legislation corpus ─────────────────
    prov = set()
    pat_sec = re.compile(r"\bSection\s+(\d+[A-Z]{0,2})\b", re.I)
    pat_art = re.compile(r"\bArticle\s+(\d+[A-Z]{0,2})\b", re.I)
    pat_ord = re.compile(r"\bOrder\s+([IVXLC]+|\d+)\b")
    if LEG_CHUNKS.exists():
        with LEG_CHUNKS.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                act = (d.get("title") or d.get("document_id") or "").strip()
                if not act:
                    continue
                txt = (d.get("text") or "")[:4000]
                an = norm_name(act)
                for kind, pat in (("section", pat_sec), ("article", pat_art), ("order", pat_ord)):
                    for m in pat.findall(txt):
                        prov.add((act, an, kind, str(m).upper(), d.get("source")))
                # the chunk's own labelled provision, when present
                for key, kind in (("section", "section"), ("article", "article")):
                    v = d.get(key)
                    if v:
                        prov.add((act, an, kind, str(v).upper(), d.get("source")))
    # ── provisions from the structured legislation parquets ──────────────
    # Far better than regex-scraping chunk text: these carry an explicit
    # section_number column, so the (act, section) pairs are authoritative
    # rather than inferred. 544k rows across 37 files.
    import glob
    leg_files = sorted(glob.glob(str(ROOT / "data/raw/open-india-law/*legislation*.parquet")))
    before = len(prov)
    for lf in leg_files:
        try:
            lpf = pq.ParquetFile(lf)
            names = set(lpf.schema_arrow.names)
            cols = [c for c in ("title", "section_number", "section_title",
                                "instrument_type", "jurisdiction") if c in names]
            if "title" not in cols or "section_number" not in cols:
                continue
            for batch in lpf.iter_batches(columns=cols, batch_size=100_000):
                for r in batch.to_pylist():
                    act = (r.get("title") or "").strip()
                    num = str(r.get("section_number") or "").strip()
                    if not act or not num:
                        continue
                    # section_number arrives as "5", "5A", "Article 21", "Order IV"
                    m = re.match(r"^(?:(section|article|order|rule|regulation|clause)\s+)?"
                                 r"([0-9]+[A-Za-z]{0,3}|[IVXLC]+)$", num, re.I)
                    if not m:
                        continue
                    kind = (m.group(1) or "").lower() or (
                        "article" if "constitution" in act.lower() else "section")
                    prov.add((act, norm_name(act), kind, m.group(2).upper(),
                              Path(lf).name))
        except Exception as e:
            log.warning(f"{Path(lf).name}: {e}")
    log.info(f"legislation parquets contributed {len(prov) - before:,} more provisions")

    # ── Constitution of India, from the official Government text ─────────
    # data/raw/huggingface/Indian_constitution is the Ministry of Law and
    # Justice edition ("as on 1st May, 2024"). Only EXPLICIT mentions
    # ("Article 21", "Arts. 32") are taken. A heading pattern like
    # r"(\d+[A-Z]{0,2})\.\s+[A-Z][a-z]" yields ~521 "articles" but sweeps up
    # schedule and clause numbering — inventing articles such as 12AA. A
    # fabricated entry here would let a hallucinated Article pass verification,
    # which is the exact failure this table exists to catch, so the
    # conservative extraction is the correct trade: what we miss is reported
    # as a source gap, never as verified.
    const_path = ROOT / "data/raw/huggingface/Indian_constitution/constitution_dataset.jsonl"
    if const_path.exists():
        blob_parts = []
        with const_path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    blob_parts.append(json.loads(line).get("text") or "")
                except Exception:
                    continue
        blob = "\n".join(blob_parts)
        arts = set()
        for pat in (re.compile(r"\bArticle\s+(\d{1,3}[A-Z]{0,2})\b", re.I),
                    re.compile(r"\bArts?\.\s*(\d{1,3}[A-Z]{0,2})\b", re.I)):
            for m in pat.findall(blob):
                m = m.upper()
                base = re.sub(r"[A-Z]", "", m)
                if base.isdigit() and 1 <= int(base) <= 395:
                    arts.add(m)
        for a in sorted(arts):
            prov.add(("Constitution of India", norm_name("Constitution of India"),
                      "article", a, const_path.name))
        log.info(f"Constitution of India: {len(arts)} distinct articles "
                 f"(explicit mentions only)")

    conn.executemany("INSERT INTO provisions (act, act_norm, kind, number, source) "
                     "VALUES (?,?,?,?,?)", sorted(prov))
    conn.commit()
    n_p = conn.execute("SELECT COUNT(*) FROM provisions").fetchone()[0]
    log.info(f"provisions: {n_p:,} distinct (act, kind, number) triples")

    conn.execute("ANALYZE"); conn.commit(); conn.close()
    log.info(f"written: {OUT.relative_to(ROOT)}")
    return n_j, n_p


if __name__ == "__main__":
    build()
