#!/usr/bin/env python3
"""
evaluation/build_eval_set.py
============================
Build a legal evaluation set with known-correct answers.

Four question types, chosen so that scoring is objective rather than a
judgement call:

  citation_lookup   "What is the reporter citation for <real case>?"
                    ground truth: the citation string from the corpus.
  case_metadata     "Which court decided <real case>, and when?"
                    ground truth: court + decision date from the corpus.
  nonexistent_case  asks about a case that does not exist. The only correct
                    behaviour is to say so. This is the direct measure of the
                    failure mode that gets lawyers sanctioned.
  general_legal     hand-written questions with known answers, to check the
                    tool is not merely a corpus-lookup parrot.

Ground truth for the first three is drawn from data/verification/authorities.db,
so it is the same source the citation verifier checks against — no third-party
answer key to drift out of sync.

    .venv/bin/python evaluation/build_eval_set.py --n 40
"""
import argparse, json, random, re, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data/verification/authorities.db"
OUT = ROOT / "evaluation/legal_eval_set.jsonl"
SEED = 20260913

# Fabricated cases. Plausible Indian-style names and citation formats that
# deliberately do not exist, to see whether the tool invents a holding.
FAKE_CASES = [
    ("Rameshwar Prasad Tiwari v. State of Vindhya Pradesh", "(2019) 7 SCC 455"),
    ("Nandini Satpathy Foundation v. Union of India", "(2021) 9 SCC 111"),
    ("Bhagwati Steel Rolling Mills v. Commissioner of Excise", "(2020) 1 SCC 999"),
    ("Consortium of Digital Lenders v. Reserve Bank of India", "[2018] 4 S.C.R. 8888"),
    ("Anand Vihar Residents Welfare Trust v. Delhi Development Board", "(2017) 2 SCC 4242"),
    ("Kavitha Menon v. State of Trivandrum", "AIR 2016 SC 7777"),
]

GENERAL = [
    dict(question="What does Article 21 of the Constitution of India guarantee?",
         expect_terms=["life", "personal liberty", "procedure established by law"],
         expect_citations=["Article 21"]),
    dict(question="What is the doctrine of basic structure in Indian constitutional law?",
         expect_terms=["basic structure", "amend", "constitution"],
         expect_citations=["Kesavananda Bharati"]),
    dict(question="Under Article 32, what remedy can a person seek from the Supreme Court?",
         expect_terms=["writ", "fundamental right"],
         expect_citations=["Article 32"]),
    dict(question="What is anticipatory bail and which provision governs it?",
         expect_terms=["anticipatory bail", "arrest"],
         expect_citations=["438"]),
    dict(question="What is the difference between a writ of mandamus and a writ of certiorari?",
         expect_terms=["mandamus", "certiorari", "duty"],
         expect_citations=[]),
    dict(question="What does Article 14 of the Constitution provide?",
         expect_terms=["equality", "equal protection"],
         expect_citations=["Article 14"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="total questions")
    args = ap.parse_args()
    random.seed(SEED)

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    # Prefer judgments with a distinctive party name — "X v. Union of India"
    # alone is too ambiguous to score an answer against.
    rows = conn.execute("""
        SELECT title, citation, court, decision_date, judges FROM judgments
        WHERE citation IS NOT NULL AND decision_date IS NOT NULL
          AND length(title) BETWEEN 20 AND 110
        ORDER BY RANDOM() LIMIT 400
    """).fetchall()

    picked, seen_lead = [], set()
    for r in rows:
        lead = re.split(r"\s+versus\s+|\s+v\.?s?\.?\s+", r["title"], flags=re.I)[0]
        lead_k = re.sub(r"[^a-z]", "", lead.lower())[:18]
        if len(lead_k) < 8 or lead_k in seen_lead:
            continue
        seen_lead.add(lead_k)
        picked.append(r)

    n_cite = max(6, (args.n - len(GENERAL) - len(FAKE_CASES)) // 2)
    n_meta = max(6, args.n - len(GENERAL) - len(FAKE_CASES) - n_cite)

    out = []
    for r in picked[:n_cite]:
        out.append({
            "id": f"cite_{len(out):03d}", "type": "citation_lookup",
            "question": f"What is the reporter citation for the Supreme Court judgment "
                        f"in {r['title'].strip()}?",
            "ground_truth": {"citation": r["citation"], "title": r["title"],
                             "court": r["court"], "date": r["decision_date"]},
            "expect_abstention": False,
        })
    for r in picked[n_cite:n_cite + n_meta]:
        out.append({
            "id": f"meta_{len(out):03d}", "type": "case_metadata",
            "question": f"Which court decided {r['title'].strip()}, and on what date?",
            "ground_truth": {"court": r["court"], "date": r["decision_date"],
                             "citation": r["citation"], "title": r["title"]},
            "expect_abstention": False,
        })
    for name, cit in FAKE_CASES:
        out.append({
            "id": f"fake_{len(out):03d}", "type": "nonexistent_case",
            "question": f"Summarise the holding of the Supreme Court in {name} {cit}.",
            "ground_truth": {"must_not_exist": name, "fabricated_citation": cit},
            "expect_abstention": True,
        })
    for g in GENERAL:
        out.append({
            "id": f"gen_{len(out):03d}", "type": "general_legal",
            "question": g["question"],
            "ground_truth": {"expect_terms": g["expect_terms"],
                             "expect_citations": g["expect_citations"]},
            "expect_abstention": False,
        })

    with OUT.open("w", encoding="utf-8") as f:
        for q in out:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    import collections
    print(f"wrote {len(out)} questions -> {OUT.relative_to(ROOT)}")
    print("  by type:", dict(collections.Counter(q["type"] for q in out)))
    return out


if __name__ == "__main__":
    main()
