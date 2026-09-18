#!/usr/bin/env python3
"""
scripts/06_index_judgments.py
=============================
Build a FAISS index over Supreme Court judgment text so the RAG layer can
actually retrieve judgments.

Why this exists: the app was retrieving only `vector_db/legislation/`, and the
pre-existing `vector_db/case_laws/` index contained one chunk per judgment —
the docket header ("Case title: … Parties: … Docket number: …"), no legal
reasoning. Neither can ground a statement about what a court held.

Output (drop-in compatible with scripts/retriever.py):
    vector_db/judgments/index.faiss
    vector_db/judgments/chunk_ids.json
    vector_db/judgments/metadata.jsonl
    data/chunks/judgments/chunks.jsonl      <- BM25 side of the hybrid retriever

Every chunk is tagged with `script` and `text_quality`. The corpus is ~59%
machine-translated regional-language text that is materially damaged (lost word
boundaries, dropped Tamil viramas, Latin glyphs wedged into Gurmukhi words).
That text is indexed — it is legitimately searchable for regional queries — but
the tag lets retrieval and citation-verification prefer the authoritative
English and avoid quoting corrupted text as if it were the judgment.

Resumable: embeddings are flushed to shards, so an interrupted run continues
from the last completed shard rather than restarting six hours of GPU work.

    .venv/bin/python scripts/06_index_judgments.py [--limit N] [--batch 64]
"""

from __future__ import annotations

import argparse, json, logging, os, re, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/raw/open-india-law/in_supreme-court_judgments.parquet"
OUT_DB = ROOT / "vector_db/judgments"
OUT_CHUNKS = ROOT / "data/chunks/judgments"
SHARDS = OUT_DB / "_shards"

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DEVICE = os.environ.get("EMBED_GPU_DEV", "cuda:4")
MAX_CHARS = 2000           # bge-m3 context; judgments chunks are already ~1-2k
SHARD_ROWS = 25_000        # flush cadence — also the resume granularity

log = logging.getLogger("index_judgments")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")

# ── script / quality tagging ────────────────────────────────────────────────
SCRIPTS = {
    "devanagari": re.compile(r"[ऀ-ॿ]"), "gurmukhi": re.compile(r"[਀-੿]"),
    "gujarati":   re.compile(r"[઀-૿]"), "oriya":    re.compile(r"[଀-୿]"),
    "tamil":      re.compile(r"[஀-௿]"), "telugu":   re.compile(r"[ఀ-౿]"),
    "kannada":    re.compile(r"[ಀ-೿]"), "malayalam":re.compile(r"[ഀ-ൿ]"),
    "bengali":    re.compile(r"[ঀ-৿]"),
}
INDIC = re.compile(r"[ऀ-෿]")
# Latin letters inside an Indic word => legacy non-Unicode font, broken glyph map
FONT_CORRUPT = re.compile(r"[ऀ-෿][A-Za-z`=][ऀ-෿]|[ऀ-෿][A-Za-z`=]\s")
MOJIBAKE = re.compile(r"â€|Ã[\x80-\xbf]|Â[\x80-\xbf]|ï»¿|�")
# long runs with no space: word boundaries lost in translation/extraction
NO_BREAKS = re.compile(r"[^\s]{45,}")


def classify(text: str) -> tuple[str, str]:
    """(script, quality) where quality is clean | degraded | corrupt."""
    if not text or not text.strip():
        return "unknown", "corrupt"
    script = "latin"
    best = 0
    for name, pat in SCRIPTS.items():
        n = len(pat.findall(text))
        if n > best:
            best, script = n, name
    if best == 0:
        script = "latin"

    if FONT_CORRUPT.search(text) or MOJIBAKE.search(text):
        return script, "corrupt"
    if INDIC.search(text) and NO_BREAKS.search(text):
        return script, "corrupt"
    if NO_BREAKS.search(text):
        return script, "degraded"
    if script != "latin":
        return script, "degraded"      # translated: usable, not authoritative
    return script, "clean"


def clean_text(t: str) -> str:
    # The corpus prefixes every chunk with a "Case: … Section: …" header that
    # repeats the metadata we already store as columns. Keeping it would let
    # BM25 match the header rather than the reasoning.
    t = re.sub(r"^Case:.*?\nSection:.*?\n+", "", t or "", flags=re.S)
    t = re.sub(r"\[SECTION\]\s*#*\s*", "", t)
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)     # hyphen line-break
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def as_text(v) -> str | None:
    """
    Coerce a parquet cell to text. Several columns (`judges`, and `bench` on
    some courts) are list-typed, so a bare .strip() breaks on them.
    """
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        parts = [as_text(x) for x in v]
        v = ", ".join(p for p in parts if p)
    v = str(v).strip()
    # pandas writes missing values as the literal string "nan" in this corpus
    return None if v.lower() in ("", "nan", "none", "null") else v


def row_to_chunk(r: dict) -> dict | None:
    text = clean_text(r.get("text") or "")
    if len(text) < 60:
        return None
    script, quality = classify(text)
    cid = str(r.get("chunk_id") or f"{r.get('case_id')}_{r.get('chunk_index')}")
    date = str(r.get("decision_date") or "")[:10]
    return {
        "chunk_id": cid,
        "document_id": str(r.get("case_id") or ""),
        "document_type": "case_law",
        "title": as_text(r.get("title")) or "",
        "citation": as_text(r.get("citation")),
        "court": as_text(r.get("court")) or "Supreme Court of India",
        "date": date or None,
        # `year` in this corpus is the REPORTER VOLUME year, not the decision
        # year — they differ by one for ~13% of cases. Keep both, unmerged.
        "reporter_year": r.get("year"),
        "judges": as_text(r.get("judges")),
        "bench": as_text(r.get("bench")),
        "disposition": as_text(r.get("disposition")),
        "petitioner": as_text(r.get("petitioner")),
        "respondent": as_text(r.get("respondent")),
        "section_type": as_text(r.get("section_type")),
        "chunk_index": r.get("chunk_index"),
        "language_code": r.get("language_code"),
        "script": script,
        "text_quality": quality,
        "source": "in_supreme-court_judgments.parquet",
        "text": text[:MAX_CHARS],
        "char_count": len(text),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows (smoke test)")
    ap.add_argument("--batch", type=int, default=64, help="embedding batch size")
    ap.add_argument("--device", default=EMBED_DEVICE)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    OUT_DB.mkdir(parents=True, exist_ok=True)
    OUT_CHUNKS.mkdir(parents=True, exist_ok=True)
    SHARDS.mkdir(parents=True, exist_ok=True)

    done = {int(p.stem.split("_")[1]) for p in SHARDS.glob("shard_*.npy")}
    if done:
        log.info(f"resuming — {len(done)} shard(s) already embedded, skipping those rows")

    from sentence_transformers import SentenceTransformer
    log.info(f"loading {EMBED_MODEL} on {args.device}")
    model = SentenceTransformer(EMBED_MODEL, device=args.device)

    pf = pq.ParquetFile(SRC)
    total = pf.metadata.num_rows if not args.limit else min(args.limit, pf.metadata.num_rows)
    log.info(f"source: {SRC.name} — {total:,} chunk rows")

    cols = [c for c in ["case_id", "chunk_id", "chunk_index", "text", "title", "citation",
                        "court", "decision_date", "year", "judges", "bench", "disposition",
                        "petitioner", "respondent", "section_type", "language_code"]
            if c in set(pf.schema_arrow.names)]

    chunks_path = OUT_CHUNKS / "chunks.jsonl"
    meta_path = OUT_DB / "metadata.jsonl"
    ids_path = OUT_DB / "chunk_ids.json"
    # Rebuilding text artefacts is cheap; only embeddings are expensive, so
    # they are the only thing resumed.
    mode = "w"
    chunks_f = chunks_path.open(mode, encoding="utf-8")
    meta_f = meta_path.open(mode, encoding="utf-8")

    shard_idx, buf_txt, buf_meta, all_ids = 0, [], [], []
    seen = t_start = 0
    t_start = time.time()
    qual = {"clean": 0, "degraded": 0, "corrupt": 0}

    def flush(idx, texts):
        """Embed one shard and persist it. Skipped if already on disk."""
        path = SHARDS / f"shard_{idx:05d}.npy"
        if path.exists():
            return
        vecs = model.encode(texts, batch_size=args.batch, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        np.save(path, vecs)

    for batch in pf.iter_batches(columns=cols, batch_size=10_000):
        for r in batch.to_pylist():
            if args.limit and seen >= args.limit:
                break
            seen += 1
            c = row_to_chunk(r)
            if not c:
                continue
            qual[c["text_quality"]] = qual.get(c["text_quality"], 0) + 1
            chunks_f.write(json.dumps(c, ensure_ascii=False) + "\n")
            meta_f.write(json.dumps({k: v for k, v in c.items() if k != "text"},
                                    ensure_ascii=False) + "\n")
            all_ids.append(c["chunk_id"])
            buf_txt.append(c["text"])

            if len(buf_txt) >= SHARD_ROWS:
                flush(shard_idx, buf_txt)
                el = time.time() - t_start
                rate = seen / max(el, 1e-9)
                log.info(f"shard {shard_idx:>4} | rows {seen:,}/{total:,} "
                         f"({100*seen/total:.1f}%) | {rate:.0f} rows/s | "
                         f"eta {(total-seen)/max(rate,1e-9)/60:.0f} min | {qual}")
                shard_idx += 1
                buf_txt = []
        if args.limit and seen >= args.limit:
            break

    if buf_txt:
        flush(shard_idx, buf_txt)
        shard_idx += 1
    chunks_f.close(); meta_f.close()
    ids_path.write_text(json.dumps(all_ids), encoding="utf-8")

    # ── assemble FAISS ─────────────────────────────────────────────────────
    log.info("stacking shards…")
    mats = [np.load(SHARDS / f"shard_{i:05d}.npy") for i in range(shard_idx)]
    if not mats:
        log.error("no embeddings produced"); return 1
    X = np.vstack(mats)
    log.info(f"embeddings: {X.shape}")

    import faiss
    dim = X.shape[1]
    n = X.shape[0]
    # IVF needs enough training points; fall back to exact search when small.
    if n >= 40_000:
        nlist = max(64, min(4096, int(np.sqrt(n))))
        quant = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        log.info(f"training IVF (nlist={nlist})…")
        index.train(X[np.random.choice(n, min(n, 200_000), replace=False)])
        index.add(X)
        index.nprobe = max(8, nlist // 32)
        itype = f"IndexIVFFlat(nlist={nlist})"
    else:
        index = faiss.IndexFlatIP(dim); index.add(X); itype = "IndexFlatIP"

    faiss.write_index(index, str(OUT_DB / "index.faiss"))
    (OUT_DB / "index_config.json").write_text(json.dumps({
        "index_type": itype, "num_vectors": int(n), "embedding_dim": int(dim),
        "metric": "inner_product (cosine, L2-normalised)", "model_name": EMBED_MODEL,
        "source_parquet": str(SRC), "text_quality_counts": qual,
        "index_file": str(OUT_DB / "index.faiss"),
        "chunk_ids_file": str(ids_path), "metadata_file": str(meta_path),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2), encoding="utf-8")

    log.info(f"DONE — {n:,} vectors in {(time.time()-t_start)/60:.1f} min")
    log.info(f"quality: {qual}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
