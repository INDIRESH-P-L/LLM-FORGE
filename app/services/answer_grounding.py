#!/usr/bin/env python3
"""
app/services/answer_grounding.py
================================
Decide how well an answer is actually supported, and whether it should be
shown at all.

The confidence label this produces is derived only from things that can be
measured about a specific answer:

    retrieval strength   how well the reranker scored the passages we found
    verification         what share of the authorities the answer asserts are
                         grounded in those passages, in the corpus, or nowhere
    provision coverage   if the question named Article/Section N, did the
                         retrieved text actually contain it
    source quality       ~59% of the judgment corpus is damaged machine
                         translation; passages tagged `corrupt` are not
                         treated as support

It replaces a label that was produced by pattern-matching the model's own
prose — i.e. by how confident the answer *sounded*. On a legal tool a
confidently-worded wrong answer is the failure mode, so the number must come
from evidence, not from tone.

Thresholds live in THRESHOLDS and are fitted against evaluation/legal_eval_*.json
(see evaluation/calibrate_confidence.py). They are not guesses tuned by eye.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict

# Support score below this and the tool declines to answer rather than guess.
# Set to 0.0 to disable abstain entirely — the system now always produces a useful answer
# and uses knowledge_mode to communicate evidence quality instead of refusing.
ABSTAIN_FLOOR = float(os.environ.get("LEGALMIND_ABSTAIN_FLOOR", "0.0"))

# knowledge_mode thresholds: how well the answer is grounded determines which mode to use.
# evidence_based   — sufficient retrieved evidence; cite sources directly.
# partial_evidence — some evidence found; mix retrieved + general knowledge.
# general_knowledge — no usable retrieved evidence; answer from model knowledge with disclaimer.
KNOWLEDGE_MODE_THRESHOLDS = {
    "evidence_based":    float(os.environ.get("LEGALMIND_KM_EVIDENCE",    "0.40")),
    "partial_evidence":  float(os.environ.get("LEGALMIND_KM_PARTIAL",     "0.15")),
    # below partial_evidence → general_knowledge
}

# How hard each unverifiable authority reduces support. This is the single
# most consequential safety knob in the project, so it is named rather than
# inlined. Measured across the 160 stored evaluation answers:
#
#     penalty   abstained   answers SHOWN containing a fabricated authority
#      0.15         6                        12
#      0.30         8                        10
#      0.45        12                         6
#
# Lower values surface more answers (useful when the verifier has false
# positives) at the cost of showing more unverifiable citations. Note that
# answers which are shown still carry the ⚠ citation-check annotation, so a
# fabrication is flagged to the reader either way — the penalty decides
# whether it is flagged or withheld.
FABRICATION_PENALTY = float(os.environ.get("LEGALMIND_FABRICATION_PENALTY", "0.15"))

# support-score cutoffs -> label. Fitted, not eyeballed.
THRESHOLDS = {"HIGH": 0.68, "MEDIUM": 0.40}

# Whether the support score has been shown to separate correct answers from
# wrong ones on the eval set. Measured by evaluation/calibrate_confidence.py:
# it currently reports AUC 0.63 with a separation of +0.015, i.e. effectively
# no signal — because 70% of the eval fails for a reason the score cannot see
# (the judgment corpus is not yet retrievable).
#
# While this is False the API reports the underlying evidence counts and does
# NOT publish a HIGH/MEDIUM/LOW label. A confidence word that does not track
# reliability is worse than none on a legal tool: it manufactures exactly the
# false assurance this layer exists to prevent. Flip to True (or set
# LEGALMIND_CONFIDENCE_CALIBRATED=1) once the fitter reports real separation.
CALIBRATED = os.environ.get("LEGALMIND_CONFIDENCE_CALIBRATED", "0").lower() in ("1", "true", "yes")

# Kept for backward-compatibility imports only — no longer shown to users.
ABSTAIN_MESSAGE = (
    "## Insufficient source material\n\n"
    "I don't have enough source material to answer this confidently.\n\n"
    "The retrieval step did not find judgments or statutory provisions in our corpus "
    "that substantively address this question, so any answer I gave would rest on the "
    "model's recollection rather than on a verifiable source — which is exactly how "
    "fabricated citations arise.\n\n"
    "**Please verify with a legal professional**, or rephrase with a specific case "
    "name, citation, statute or section so the search has something concrete to match."
)

# Notice prepended to answers produced from general legal knowledge (no usable retrieved evidence).
GENERAL_KNOWLEDGE_NOTICE = (
    "> ⚠️ **Source Verification Unavailable** — No relevant documents were retrieved from "
    "the legal corpus for this query. The answer below is based on general Indian legal "
    "knowledge. Specific citations, case names, and section numbers have **not** been "
    "verified against retrieved sources. Please consult a qualified lawyer before acting "
    "on this information.\n\n"
)

# Notice prepended to answers with only partial retrieved support.
PARTIAL_EVIDENCE_NOTICE = (
    "> ℹ️ **Partial Source Coverage** — Only limited retrieved evidence was found for this "
    "query. Sections marked `[VERIFIED SOURCE]` are drawn from retrieved documents; "
    "sections marked `[GENERAL LEGAL KNOWLEDGE]` rely on the model's legal training. "
    "Verify unconfirmed details with a qualified lawyer.\n\n"
)


@dataclass
class Grounding:
    support: float               # 0..1
    confidence: str              # HIGH | MEDIUM | LOW  (internal, always computed)
    calibrated: bool             # is `confidence` fit to show a user?
    should_abstain: bool         # always False now; kept for API compatibility
    knowledge_mode: str          # evidence_based | partial_evidence | general_knowledge
    retrieval_score: float
    verified_ratio: float
    unverified_count: int
    provision_covered: bool | None
    usable_sources: int
    reasons: list[str]

    def to_dict(self):
        d = asdict(self)
        d["support"] = round(self.support, 4)
        d["retrieval_score"] = round(self.retrieval_score, 4)
        d["verified_ratio"] = round(self.verified_ratio, 4)
        return d


def _rerank_strength(chunks: list[dict]) -> float:
    """
    Normalised strength of the retrieved set.

    bge-reranker logits are roughly -11..+11; map through a logistic so the
    score is comparable across questions. Falls back to RRF score, then to a
    flat prior when the reranker is unavailable.
    """
    if not chunks:
        return 0.0
    scores = [c.get("rerank_score") for c in chunks if c.get("rerank_score") is not None]
    if scores:
        import math
        top = max(scores)
        return 1.0 / (1.0 + math.exp(-float(top)))
    rrf = [c.get("rrf_score") for c in chunks if c.get("rrf_score") is not None]
    if rrf:
        return min(1.0, float(max(rrf)) * 20.0)
    return 0.35


def _chunk_text(c: dict) -> str:
    """
    Pull the passage text whatever the caller named the field.

    run_rag() emits retrieved_documents with the body under `text_preview`;
    the retriever's own chunks use `text`; the stored citation shape uses
    `excerpt`. Checking only the first two silently discarded every passage
    and drove the support score to zero for every question.
    """
    for key in ("text", "text_preview", "excerpt"):
        v = c.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _usable(chunks: list[dict]) -> list[dict]:
    """Passages fit to support a legal proposition."""
    out = []
    for c in chunks:
        if (c.get("text_quality") or "clean") == "corrupt":
            continue                      # damaged translation: not authority
        if len(_chunk_text(c)) < 80:
            continue
        out.append(c)
    return out


def _provision_covered(query: str, chunks: list[dict]) -> bool | None:
    """If the question named Article/Section N, was N actually retrieved?"""
    nums = re.findall(r"\b(?:article|section|sec\.?|art\.?)\s*(\d+[A-Za-z]{0,2})\b",
                      query or "", re.I)
    if not nums:
        return None
    blob = " ".join(_chunk_text(c) for c in chunks)
    return all(re.search(rf"\b{re.escape(n)}\b", blob, re.I) for n in nums)


def assess(query: str, chunks: list[dict], verification=None) -> Grounding:
    chunks = chunks or []
    usable = _usable(chunks)
    reasons: list[str] = []

    retrieval = _rerank_strength(usable)
    if not usable:
        reasons.append("no usable passages were retrieved")
    elif len(usable) < len(chunks):
        reasons.append(f"{len(chunks) - len(usable)} retrieved passage(s) discarded as "
                       "corrupt or too short")

    checked = getattr(verification, "checked", 0) or 0
    grounded = getattr(verification, "grounded", 0) or 0
    in_corpus = getattr(verification, "in_corpus", 0) or 0
    unverified = getattr(verification, "unverified", 0) or 0
    # in_corpus counts for less than grounded: the authority is real, but the
    # model was not looking at it when it made the claim.
    verified_ratio = ((grounded + 0.45 * in_corpus) / checked) if checked else 0.0
    if unverified:
        reasons.append(f"{unverified} asserted authority(ies) could not be verified")
    if checked and grounded == 0:
        reasons.append("no asserted authority appears in the retrieved passages")

    covered = _provision_covered(query, usable)
    if covered is False:
        reasons.append("the provision named in the question was not found in the sources")

    # ── combine ──────────────────────────────────────────────────────────
    support = 0.55 * retrieval + 0.45 * (verified_ratio if checked else retrieval)
    if checked:
        # Each unverifiable authority reduces support; see FABRICATION_PENALTY
        # for the measured trade-off between recall and showing bad citations.
        support *= max(0.0, 1.0 - FABRICATION_PENALTY * unverified)
    if covered is False:
        support *= 0.5
    if not usable:
        support = 0.0
    support = max(0.0, min(1.0, support))

    if support >= THRESHOLDS["HIGH"]:
        label = "HIGH"
    elif support >= THRESHOLDS["MEDIUM"]:
        label = "MEDIUM"
    else:
        label = "LOW"

    # Determine knowledge_mode from evidence strength.
    # Never refuse to answer; always return a useful response.
    # Use the final support score (after all penalties applied) rather than
    # checking usable chunk presence: if fabrication penalties drove support to
    # zero the answer has no reliable grounding regardless of chunk count.
    if support >= KNOWLEDGE_MODE_THRESHOLDS["evidence_based"]:
        km = "evidence_based"
    elif support >= KNOWLEDGE_MODE_THRESHOLDS["partial_evidence"]:
        km = "partial_evidence"
    else:
        km = "general_knowledge"

    return Grounding(
        support=support, confidence=label, calibrated=CALIBRATED,
        should_abstain=False,   # never refuse; knowledge_mode handles transparency
        knowledge_mode=km,
        retrieval_score=retrieval, verified_ratio=verified_ratio,
        unverified_count=unverified, provision_covered=covered,
        usable_sources=len(usable),
        reasons=reasons or ["answer is supported by the retrieved passages"],
    )
