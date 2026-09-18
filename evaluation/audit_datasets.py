#!/usr/bin/env python3
"""
evaluation/audit_datasets.py
============================
Dataset correctness audit for LegalMind AI.

Checks, over a sampled but statistically meaningful slice of each corpus:
  1. structural  — null/empty fields, broken rows, encoding damage, date formats
  2. content     — do title / citation / court / date agree with the judgment text
  3. coverage    — date, court and topic distribution
  4. leakage     — train/eval overlap

Writes evaluation/dataset_audit_report.json and prints a summary.
Sampling is deterministic (seeded) so re-runs are comparable.

    .venv/bin/python evaluation/audit_datasets.py [--sample N]
"""

import argparse, collections, glob, hashlib, json, os, random, re, sys, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260913
random.seed(SEED)

# ── encoding-damage signatures ──────────────────────────────────────────────
# Mojibake: UTF-8 read as latin-1 ("â€™", "Ã©"), lone replacement chars, and
# PDF-extraction artefacts (ligature loss, control chars, hyphen-broken words).
MOJIBAKE = re.compile(r"â€|Ã[\x80-\xbf]|Â[\x80-\xbf]|ï»¿|�")
CTRL     = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
BROKEN_W = re.compile(r"\w-\s*\n\s*\w")
SPACED   = re.compile(r"(?:\b\w\s){6,}")          # "S U P R E M E  C O U R T"
NOSPACE  = re.compile(r"[a-z]{25,}")              # lost word boundaries
DATE_PATS = [
    ("ISO   YYYY-MM-DD", re.compile(r"^\d{4}-\d{2}-\d{2}$")),
    ("ISO datetime",     re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}")),
    ("DD-MM-YYYY",       re.compile(r"^\d{2}-\d{2}-\d{4}$")),
    ("DD/MM/YYYY",       re.compile(r"^\d{2}/\d{2}/\d{4}$")),
    ("YYYY only",        re.compile(r"^\d{4}$")),
]
# Indian reporters are routinely written with periods: "[1979] 3 S.C.R. 453",
# "A.I.R. 1973 S.C. 1461". An earlier version of this regex required bare
# letters and therefore flagged 100% of valid citations as malformed.
CITATION_PAT = re.compile(
    r"[\[\(]?\d{4}[\]\)]?\s*(?:\d+\s*)?"
    r"(?:S\.?\s?C\.?\s?C|A\.?\s?I\.?\s?R|S\.?\s?C\.?\s?R|SCC\s*OnLine|INSC|"
    r"ALL\s?ER|Cri\.?\s?L\.?J|Supp)\b", re.I)

# Indic script blocks, for telling translated text from English.
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
GURMUKHI   = re.compile(r"[\u0A00-\u0A7F]")
GUJARATI   = re.compile(r"[\u0A80-\u0AFF]")
TAMIL      = re.compile(r"[\u0B80-\u0BFF]")
TELUGU     = re.compile(r"[\u0C00-\u0C7F]")
MALAYALAM  = re.compile(r"[\u0D00-\u0D7F]")
BENGALI    = re.compile(r"[\u0980-\u09FF]")
INDIC_ANY  = re.compile(r"[\u0900-\u0DFF]")
# Latin letters wedged inside an Indic word: the signature of a legacy
# (non-Unicode) PDF font being extracted with a broken glyph map.
FONT_CORRUPT = re.compile(r"[\u0900-\u0DFF][A-Za-z`=][\u0900-\u0DFF]|"
                          r"[\u0900-\u0DFF][A-Za-z`=]\s")


def dominant_script(t):
    counts = {
        "devanagari": len(DEVANAGARI.findall(t)), "gurmukhi": len(GURMUKHI.findall(t)),
        "gujarati": len(GUJARATI.findall(t)),     "tamil": len(TAMIL.findall(t)),
        "telugu": len(TELUGU.findall(t)),         "malayalam": len(MALAYALAM.findall(t)),
        "bengali": len(BENGALI.findall(t)),
    }
    top = max(counts, key=counts.get)
    return top if counts[top] > 0 else "latin"


def norm(s):
    return unicodedata.normalize("NFKC", s or "").strip()


def text_flags(t):
    """Encoding / extraction problems in a block of text."""
    f = []
    if not t or not t.strip():
        return ["empty"]
    if MOJIBAKE.search(t):  f.append("mojibake")
    if CTRL.search(t):      f.append("control_chars")
    if BROKEN_W.search(t):  f.append("hyphen_linebreak")
    if SPACED.search(t):    f.append("letter_spaced")
    if NOSPACE.search(t):   f.append("lost_word_breaks")
    # Indic-script text is alphabetic too, so a plain alpha ratio flags healthy
    # translated judgments. Only punctuation/digit soup is real extraction noise.
    alpha = sum(c.isalpha() or c.isspace() for c in t)
    if len(t) > 80 and alpha / len(t) < 0.55:
        f.append("symbol_soup")
    if INDIC_ANY.search(t):
        f.append("indic_script")
        if FONT_CORRUPT.search(t):
            f.append("legacy_font_corruption")
    if len(t.strip()) < 40:
        f.append("very_short")
    return f


# ═══════════════════════════════════════════════════════════════════════════
def audit_parquet(path, sample_n, label):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    cols = set(pf.schema_arrow.names)

    want = [c for c in ["case_id", "chunk_id", "chunk_index", "text", "title", "citation",
                        "court", "decision_date", "year", "judges", "bench",
                        "case_number", "disposition", "petitioner", "respondent",
                        "language_code", "source_url"] if c in cols]

    # reservoir-sample rows across the whole file
    reservoir, seen = [], 0
    for batch in pf.iter_batches(columns=want, batch_size=50_000):
        rows = batch.to_pylist()
        for r in rows:
            seen += 1
            if len(reservoir) < sample_n:
                reservoir.append(r)
            else:
                j = random.randrange(seen)
                if j < sample_n:
                    reservoir[j] = r

    res = {
        "file": os.path.basename(path), "label": label,
        "total_chunk_rows": total_rows, "sampled": len(reservoir),
        "columns": sorted(cols),
        "null_or_empty": collections.Counter(),
        "text_problems": collections.Counter(),
        "date_formats": collections.Counter(),
        "content_mismatch": collections.Counter(),
        "worst": [],
    }

    years, courts, dispositions, langs = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    case_ids, text_hashes = set(), collections.Counter()

    for r in reservoir:
        for c in want:
            v = r.get(c)
            if v is None or (isinstance(v, str) and not v.strip()):
                res["null_or_empty"][c] += 1

        t = norm(r.get("text") or "")
        flags = text_flags(t)
        for f in flags:
            res["text_problems"][f] += 1

        d = str(r.get("decision_date") or "").strip()
        if d:
            for name, pat in DATE_PATS:
                if pat.match(d):
                    res["date_formats"][name] += 1
                    break
            else:
                res["date_formats"][f"UNRECOGNISED: {d[:22]}"] += 1
        else:
            res["date_formats"]["(missing)"] += 1

        # ── content cross-checks against the judgment text itself ──────────
        title = norm(r.get("title") or "")
        cit   = norm(r.get("citation") or "")
        court = norm(r.get("court") or "")
        yr    = r.get("year")

        if cit and not CITATION_PAT.search(cit):
            res["content_mismatch"]["citation_not_reporter_shaped"] += 1
        # only meaningful on the first chunk, where the cause-title appears
        if r.get("chunk_index") in (0, "0") and title and t:
            parts = [p for p in re.split(r"\s+v\.?s?\.?\s+|\s+VERSUS\s+", title, flags=re.I) if p]
            if parts:
                lead = re.sub(r"[^a-z ]", "", parts[0].lower()).strip()
                toks = [w for w in lead.split() if len(w) > 3][:3]
                if toks and not any(w in t.lower() for w in toks):
                    res["content_mismatch"]["party_name_absent_from_text"] += 1
        if yr and d and re.match(r"^\d{4}", d) and str(yr) != d[:4]:
            res["content_mismatch"]["year_vs_decision_date_disagree"] += 1
        if court and "supreme" in label.lower() and "supreme" not in court.lower():
            res["content_mismatch"]["court_label_mismatch"] += 1

        if flags and len(res["worst"]) < 12 and "very_short" not in flags:
            res["worst"].append({
                "chunk_id": str(r.get("chunk_id"))[:60], "flags": flags,
                "court": court[:40], "date": d[:20], "citation": cit[:40],
                "text_sample": t[:220],
            })

        if t:
            scr = dominant_script(t)
            res.setdefault("_scripts", collections.Counter())[scr] += 1
            lc = str(r.get("language_code") or "?")
            if lc == "en" and scr != "latin":
                res["content_mismatch"]["lang_en_but_indic_text"] += 1
            elif lc not in ("en", "?") and scr == "latin":
                res["content_mismatch"]["lang_regional_but_latin_text"] += 1
        if yr: years[str(yr)] += 1
        if court: courts[court] += 1
        if r.get("disposition"): dispositions[str(r["disposition"])[:30]] += 1
        if r.get("language_code"): langs[str(r["language_code"])] += 1
        if r.get("case_id"): case_ids.add(r["case_id"])
        if t: text_hashes[hashlib.sha1(re.sub(r"\W+", "", t.lower()).encode()).hexdigest()] += 1

    dupes = sum(c - 1 for c in text_hashes.values() if c > 1)
    res["coverage"] = {
        "distinct_case_ids_in_sample": len(case_ids),
        "year_range": (min(years) if years else None, max(years) if years else None),
        "top_years": years.most_common(8),
        "thin_years": [y for y, c in sorted(years.items()) if c <= 2][:12],
        "courts": courts.most_common(6),
        "dispositions": dispositions.most_common(5),
        "languages": langs.most_common(5),
    }
    res["duplicate_chunk_texts"] = dupes
    res["scripts"] = dict(res.pop("_scripts", collections.Counter()))
    return res


# ═══════════════════════════════════════════════════════════════════════════
def audit_jsonl(path, sample_n, label, text_keys=("text", "content", "answer", "output")):
    p = Path(path)
    if not p.exists():
        return None
    total = sum(1 for _ in p.open("r", encoding="utf-8", errors="replace"))
    take = sorted(random.sample(range(total), min(sample_n, total))) if total else []
    want = set(take)
    res = {"file": str(p.relative_to(ROOT)), "label": label, "total_rows": total,
           "sampled": len(take), "broken_json": 0,
           "null_or_empty": collections.Counter(), "text_problems": collections.Counter(),
           "worst": []}
    hashes = collections.Counter()
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i not in want:
                continue
            try:
                d = json.loads(line)
            except Exception:
                res["broken_json"] += 1
                continue
            if not isinstance(d, dict):
                continue
            for k, v in d.items():
                if v is None or (isinstance(v, str) and not v.strip()):
                    res["null_or_empty"][k] += 1
            blob = ""
            for k in text_keys:
                if isinstance(d.get(k), str):
                    blob += d[k] + "\n"
            if "messages" in d and isinstance(d["messages"], list):
                blob += "\n".join(m.get("content", "") for m in d["messages"]
                                  if isinstance(m, dict))
            if blob.strip():
                for fl in text_flags(blob):
                    res["text_problems"][fl] += 1
                hashes[hashlib.sha1(re.sub(r"\W+", "", blob.lower()).encode()).hexdigest()] += 1
                if MOJIBAKE.search(blob) and len(res["worst"]) < 6:
                    res["worst"].append({"row": i, "sample": blob[:220]})
    res["duplicate_rows_in_sample"] = sum(c - 1 for c in hashes.values() if c > 1)
    return res


def audit_leakage(train_path, eval_path):
    """Exact and near-duplicate overlap between a train and an eval split."""
    def load(p):
        out = []
        pp = Path(p)
        if not pp.exists():
            return out
        with pp.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                txt = ""
                if isinstance(d.get("messages"), list):
                    txt = " ".join(m.get("content", "") for m in d["messages"]
                                   if isinstance(m, dict) and m.get("role") != "system")
                else:
                    txt = " ".join(str(d.get(k, "")) for k in
                                   ("instruction", "input", "question", "prompt", "output", "answer"))
                out.append(txt)
        return out

    tr, ev = load(train_path), load(eval_path)
    if not tr or not ev:
        return {"train_rows": len(tr), "eval_rows": len(ev), "note": "a split was empty/missing"}

    def key(t):
        return hashlib.sha1(re.sub(r"\W+", "", t.lower()).encode()).hexdigest()

    def shingles(t, n=8):
        w = re.sub(r"\W+", " ", t.lower()).split()
        return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}

    tr_keys = {key(t) for t in tr}
    exact = sum(1 for t in ev if key(t) in tr_keys)

    # near-duplicate: Jaccard over 8-grams against a sampled slice of train
    tr_sample = random.sample(tr, min(400, len(tr)))
    tr_sh = [shingles(t) for t in tr_sample]
    near, worst = 0, []
    for t in random.sample(ev, min(200, len(ev))):
        s = shingles(t)
        if not s:
            continue
        best = max((len(s & o) / len(s | o) if (s | o) else 0) for o in tr_sh) if tr_sh else 0
        if best >= 0.8:
            near += 1
            if len(worst) < 3:
                worst.append({"jaccard": round(best, 3), "eval_sample": t[:200]})
    return {
        "train_rows": len(tr), "eval_rows": len(ev),
        "exact_overlap": exact,
        "exact_overlap_pct": round(100 * exact / len(ev), 2),
        "near_dup_checked": min(200, len(ev)),
        "near_dup_found": near,
        "near_dup_pct": round(100 * near / min(200, len(ev)), 2),
        "worst": worst,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=4000)
    args = ap.parse_args()

    report = {"seed": SEED, "sample_per_file": args.sample, "parquet": [], "jsonl": [], "leakage": {}}

    sc = ROOT / "data/raw/open-india-law/in_supreme-court_judgments.parquet"
    if sc.exists():
        print(f"auditing {sc.name} …", flush=True)
        report["parquet"].append(audit_parquet(str(sc), args.sample, "Supreme Court"))

    others = sorted(glob.glob(str(ROOT / "data/raw/open-india-law/*.parquet")))
    others = [f for f in others if "supreme-court" not in f][:3]
    for f in others:
        print(f"auditing {os.path.basename(f)} …", flush=True)
        report["parquet"].append(audit_parquet(f, max(600, args.sample // 6), "High Court"))

    for rel, label in [
        ("data/chunks/case_laws/indexed_chunks.jsonl", "case-law chunks (INDEXED)"),
        ("data/chunks/legislation/chunks.jsonl",        "legislation chunks (INDEXED, used by app)"),
        ("fine_tuning/datasets/legal_sft_train.jsonl",  "SFT train"),
        ("fine_tuning/datasets/legal_sft_eval.jsonl",   "SFT eval"),
    ]:
        p = ROOT / rel
        if p.exists():
            print(f"auditing {rel} …", flush=True)
            r = audit_jsonl(str(p), min(args.sample, 3000), label)
            if r:
                report["jsonl"].append(r)

    print("checking train/eval leakage …", flush=True)
    report["leakage"]["sft_train_vs_eval"] = audit_leakage(
        ROOT / "fine_tuning/datasets/legal_sft_train.jsonl",
        ROOT / "fine_tuning/datasets/legal_sft_eval.jsonl")

    out = ROOT / "evaluation/dataset_audit_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten: {out.relative_to(ROOT)}")
    return report


if __name__ == "__main__":
    main()
