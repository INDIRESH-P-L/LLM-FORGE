#!/usr/bin/env python3
"""
evaluation/calibrate_confidence.py
==================================
Fit the confidence thresholds in app/services/answer_grounding.py against
measured correctness, instead of choosing numbers that look reasonable.

A confidence label on a legal tool is only worth showing if high-confidence
answers are actually right more often than low-confidence ones. This script
answers that question directly, and reports the standard calibration
diagnostics rather than a single number:

    separation    mean support on correct answers minus mean on wrong ones.
                  If this is ~0 the score carries no signal and no choice of
                  threshold can rescue it — that must be said, not papered over.
    AUC           probability a random correct answer scores above a random
                  wrong one. 0.5 = coin flip.
    ECE           expected calibration error across support bins.
    best split    thresholds maximising bucket separation, searched over a grid.

Abstentions are excluded: for those the "correct" outcome is the refusal, so
including them measures refusal quality, not answer reliability.

    .venv/bin/python evaluation/calibrate_confidence.py evaluation/legal_eval_after_fixed.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def auc(correct: list[float], wrong: list[float]) -> float:
    """Mann-Whitney U / |pos||neg| — ties count a half."""
    if not correct or not wrong:
        return float("nan")
    wins = sum((1.0 if c > w else 0.5 if c == w else 0.0)
               for c in correct for w in wrong)
    return wins / (len(correct) * len(wrong))


def ece(rows: list[tuple[float, bool]], bins: int = 5) -> float:
    """Expected calibration error, treating support as a predicted probability."""
    if not rows:
        return float("nan")
    total, n = 0.0, len(rows)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(s, c) for s, c in rows if (lo <= s < hi or (b == bins - 1 and s == 1.0))]
        if not sel:
            continue
        conf = sum(s for s, _ in sel) / len(sel)
        acc = sum(1 for _, c in sel if c) / len(sel)
        total += (len(sel) / n) * abs(acc - conf)
    return total


def grid_search(rows: list[tuple[float, bool]]) -> tuple[float, float, float]:
    """
    Pick (medium, high) cutoffs maximising monotone separation:
    accuracy(HIGH) >= accuracy(MEDIUM) >= accuracy(LOW), scored by the spread
    between the top and bottom bucket, with a minimum bucket size so a single
    lucky answer cannot define a threshold.
    """
    best, best_score = (0.40, 0.68), -1.0
    grid = [i / 50 for i in range(1, 50)]
    n = len(rows)
    min_n = max(2, n // 10)
    for med in grid:
        for high in grid:
            if high <= med:
                continue
            lo = [c for s, c in rows if s < med]
            mid = [c for s, c in rows if med <= s < high]
            hi = [c for s, c in rows if s >= high]
            if min(len(lo), len(mid), len(hi)) < min_n:
                continue
            a_lo = sum(lo) / len(lo); a_mid = sum(mid) / len(mid); a_hi = sum(hi) / len(hi)
            if not (a_hi >= a_mid >= a_lo):
                continue
            score = a_hi - a_lo
            if score > best_score:
                best, best_score = (med, high), score
    return best[0], best[1], best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?",
                    default=str(ROOT / "evaluation/legal_eval_after_fixed.json"))
    ap.add_argument("--include-abstentions", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.report).read_text(encoding="utf-8"))
    results = data["results"]
    rows = [(float(r["support"]), bool(r["correct"])) for r in results
            if r.get("support") is not None
            and (args.include_abstentions or not r.get("abstained_by_system"))]

    if len(rows) < 8:
        print(f"only {len(rows)} scorable answers — too few to fit thresholds.")
        return 1

    cor = [s for s, c in rows if c]
    wrong = [s for s, c in rows if not c]
    sep = (sum(cor) / len(cor) - sum(wrong) / len(wrong)) if cor and wrong else float("nan")
    a = auc(cor, wrong)
    e = ece(rows)

    print(f"scorable answers: {len(rows)}  (correct={len(cor)}, wrong={len(wrong)})")
    print(f"  mean support | correct : {sum(cor)/len(cor):.3f}" if cor else "  no correct answers")
    print(f"  mean support | wrong   : {sum(wrong)/len(wrong):.3f}" if wrong else "  no wrong answers")
    print(f"  separation             : {sep:+.3f}")
    print(f"  AUC                    : {a:.3f}   (0.5 = no signal)")
    print(f"  ECE                    : {e:.3f}")

    if not (a == a) or a < 0.60 or abs(sep) < 0.05:
        print("\n  VERDICT: the support score does not separate correct from wrong answers.")
        print("  No threshold choice can fix that — the inputs to the score must improve")
        print("  first (chiefly: retrieval that can actually reach the source judgments).")
        print("  Showing a confidence label now would be the fake-confidence failure the")
        print("  score exists to prevent. Recommend leaving it suppressed until AUC > 0.6.")
        return 2

    med, high, score = grid_search(rows)
    print(f"\n  fitted thresholds: MEDIUM >= {med:.2f}, HIGH >= {high:.2f}  (spread {score:.2f})")
    for lbl, sel in (("HIGH", [c for s, c in rows if s >= high]),
                     ("MEDIUM", [c for s, c in rows if med <= s < high]),
                     ("LOW", [c for s, c in rows if s < med])):
        if sel:
            print(f"    {lbl:<7} n={len(sel):<3} accuracy={100*sum(sel)/len(sel):5.1f}%")
    print("\n  apply by editing THRESHOLDS in app/services/answer_grounding.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
