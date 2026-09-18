#!/usr/bin/env python3
"""
evaluation/run_legal_eval.py
============================
Run the app against evaluation/legal_eval_set.jsonl and report:

  * accuracy, per question type
  * CITATION HALLUCINATION RATE — reported separately from accuracy, because
    an answer can be broadly correct and still cite an invented authority,
    and that is the failure that gets people sanctioned
  * abstention behaviour on cases that do not exist
  * confidence calibration — accuracy within each stated-confidence bucket

Scoring is deliberately mechanical: exact/normalised string containment
against ground truth from the same corpus the verifier checks. No model
grades another model.

    .venv/bin/python evaluation/run_legal_eval.py --label before
"""
from __future__ import annotations
import argparse, json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import requests, urllib3
urllib3.disable_warnings()
from app.services.citation_verifier import CitationVerifier, norm_citation

EVAL = ROOT / "evaluation/legal_eval_set.jsonl"

ABSTAIN_MARKERS = [
    "not contain", "no record", "could not find", "cannot find", "not found",
    "does not appear", "no such case", "unable to locate", "insufficient",
    "not have enough", "cannot verify", "could not be verified", "no information",
    "not present in", "do not have", "does not exist", "unable to verify",
    "not in the retrieved", "no relevant", "verify with a legal professional",
    "consult a legal professional", "not available in the provided",
]


def looks_like_abstention(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in ABSTAIN_MARKERS)


def date_matches(answer: str, iso: str) -> bool:
    """Accept the date in any common rendering, not just ISO."""
    if not iso:
        return False
    y, m, d = iso[:4], iso[5:7], iso[8:10]
    if y and y not in answer:
        return False
    months = ["january", "february", "march", "april", "may", "june", "july",
              "august", "september", "october", "november", "december"]
    try:
        mn = months[int(m) - 1]
    except Exception:
        mn = ""
    a = answer.lower()
    day = str(int(d)) if d.isdigit() else d
    return (iso in answer
            or (mn and mn in a and day in a)
            or (f"{d}-{m}-{y}" in answer) or (f"{d}/{m}/{y}" in answer)
            or (f"{day} {mn}" in a and y in a))


def score(q: dict, answer: str, report) -> tuple[bool, str]:
    gt = q["ground_truth"]
    t = q["type"]
    a = answer or ""
    al = a.lower()

    if t == "citation_lookup":
        want = norm_citation(gt["citation"])
        got = {norm_citation(m.group(0)) for m in
               re.finditer(r"[\[\(]?\s*(?:19|20)\d{2}.{0,24}?\d{1,5}", a)}
        if any(want and (want in g or g in want) for g in got if g):
            return True, "citation matched"
        if looks_like_abstention(a):
            return False, "abstained (no citation given)"
        return False, "citation absent or wrong"

    if t == "case_metadata":
        court_ok = bool(gt.get("court")) and any(
            w in al for w in ["supreme court", "supreme court of india"])
        date_ok = date_matches(a, gt.get("date") or "")
        if court_ok and date_ok:
            return True, "court + date correct"
        return False, f"court_ok={court_ok} date_ok={date_ok}"

    if t == "nonexistent_case":
        # Correct = say it cannot be found AND do not assert the fake citation
        # as real. Asserting an invented holding is the worst outcome here.
        fabricated = any(f.status == "unverified" for f in report.findings)
        if looks_like_abstention(a) and not fabricated:
            return True, "correctly declined"
        if looks_like_abstention(a) and fabricated:
            return False, "declined but still asserted unverifiable authority"
        return False, "fabricated a holding for a non-existent case"

    if t == "general_legal":
        terms = [x.lower() for x in gt.get("expect_terms", [])]
        hit = sum(1 for x in terms if x in al)
        need = max(1, int(len(terms) * 0.6))
        if hit >= need:
            return True, f"{hit}/{len(terms)} key terms"
        return False, f"only {hit}/{len(terms)} key terms"

    return False, "unknown type"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--base", default="https://localhost:8443")
    ap.add_argument("--ca", default=str(ROOT / "certs/legalmind-ca.crt"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    qs = [json.loads(l) for l in EVAL.open(encoding="utf-8")]
    if args.limit:
        qs = qs[:args.limit]
    verifier = CitationVerifier()
    verify_kw = {"verify": args.ca} if args.base.startswith("https") else {}

    results, t0 = [], time.time()
    for i, q in enumerate(qs, 1):
        try:
            r = requests.post(f"{args.base}/api/chat",
                              json={"query": q["question"],
                                    "client_token": f"eval-{args.label}-{q['id']}-{int(t0)}"},
                              timeout=900, **verify_kw)
            r.raise_for_status()
            d = r.json()
            msg = d.get("assistant_message") or {}
            answer = msg.get("content") or ""
            meta = msg.get("metadata") or {}
            cites = msg.get("citations") or []
        except Exception as e:
            answer, meta, cites = f"__ERROR__ {e}", {}, []

        chunks = [{"text": c.get("excerpt") or "", "citation": c.get("citation"),
                   "title": c.get("case_name")} for c in cites]
        rep = verifier.verify(answer, chunks)
        ok, why = score(q, answer, rep)

        results.append({
            "id": q["id"], "type": q["type"], "question": q["question"],
            "correct": ok, "why": why,
            "confidence": (meta.get("confidence") or "").split("\n")[0].strip(),
            "support": meta.get("support"),
            "abstained_by_system": bool(meta.get("abstained")),
            "retrieved": len(cites),
            "authorities_checked": rep.checked,
            "unverified": rep.unverified,
            "source_gap": rep.source_gap,
            "grounded": rep.grounded,
            "in_corpus": rep.in_corpus,
            "abstained": looks_like_abstention(answer),
            "answer_excerpt": answer[:300],
        })
        print(f"  [{i:>2}/{len(qs)}] {q['type']:<17} {'PASS' if ok else 'FAIL'}  "
              f"cites={rep.checked} unver={rep.unverified}  {why}", flush=True)

    # ── aggregate ────────────────────────────────────────────────────────
    by_type = defaultdict(lambda: [0, 0])
    for r in results:
        by_type[r["type"]][1] += 1
        by_type[r["type"]][0] += r["correct"]

    tot_auth = sum(r["authorities_checked"] for r in results)
    tot_unver = sum(r["unverified"] for r in results)
    tot_ground = sum(r["grounded"] for r in results)
    tot_gap = sum(r.get("source_gap", 0) for r in results)
    adjudicable = tot_auth - tot_gap
    answers_with_fab = sum(1 for r in results if r["unverified"] > 0)

    # calibration: accuracy within each stated-confidence bucket
    calib = defaultdict(lambda: [0, 0])
    for r in results:
        c = (r["confidence"] or "NONE").upper()
        c = "HIGH" if "HIGH" in c else "MEDIUM" if "MED" in c else "LOW" if "LOW" in c else "NONE"
        calib[c][1] += 1
        calib[c][0] += r["correct"]

    summary = {
        "label": args.label,
        "n": len(results),
        "accuracy": round(sum(r["correct"] for r in results) / max(len(results), 1), 4),
        "by_type": {k: {"correct": v[0], "n": v[1], "accuracy": round(v[0] / v[1], 4)}
                    for k, v in by_type.items()},
        "citations": {
            "authorities_asserted": tot_auth,
            "grounded_in_retrieved": tot_ground,
            "unverified_fabricated": tot_unver,
            # Authorities our corpus cannot adjudicate (S.C.C./A.I.R. series we
            # do not index, provisions outside our thin provisions table).
            # Excluded from the hallucination rate so the number means what it
            # says rather than measuring our own coverage gaps.
            "source_gap_uncheckable": tot_gap,
            "adjudicable": adjudicable,
            "citation_hallucination_rate": round(tot_unver / adjudicable, 4) if adjudicable else 0.0,
            "grounding_rate": round(tot_ground / adjudicable, 4) if adjudicable else 0.0,
            "answers_containing_an_unverifiable_authority": answers_with_fab,
            "pct_answers_affected": round(100 * answers_with_fab / max(len(results), 1), 1),
        },
        "abstention": {
            "should_abstain": sum(1 for r in results if r["type"] == "nonexistent_case"),
            "did_abstain": sum(1 for r in results
                               if r["type"] == "nonexistent_case" and r["abstained"]),
        },
        "calibration": {k: {"correct": v[0], "n": v[1],
                            "accuracy": round(v[0] / v[1], 4) if v[1] else None}
                        for k, v in sorted(calib.items())},
        "avg_retrieved_chunks": round(sum(r["retrieved"] for r in results) / max(len(results), 1), 2),
        "elapsed_s": round(time.time() - t0, 1),
    }

    out = ROOT / f"evaluation/legal_eval_{args.label}.json"
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=2),
                   encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwritten: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
