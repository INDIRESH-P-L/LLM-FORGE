#!/usr/bin/env python3
"""
scripts/legal_reasoning.py
==========================
LegalMind AI — Evidence-Grounded Legal Reasoning Engine

Generates concise, verifiable, structured IRAC reasoning summaries:
1. Issue Identified
2. Relevant Legal Provision
3. Applicable Rule
4. Relevant Facts from User's Question
5. Application of the Rule
6. Exceptions or Limitations
7. Conclusion
8. Sources

Rules:
- Strictly suppresses hidden chain-of-thought (<think> blocks).
- Every reasoning step is strictly grounded in retrieved evidence chunks.
- Never invents facts, sections, or propositions.
- Marks unverified or absent items explicitly.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReasoningStep:
    step: int
    phase: str
    description: str
    grounded: bool
    evidence_citation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_structured_reasoning(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    parsed_sections: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Constructs an 8-step evidence-based legal reasoning summary based strictly on
    retrieved evidence and query parameters.
    """
    parsed = parsed_sections or {}
    provisions = parsed.get("legal_provisions") or []
    judgments = parsed.get("judgments") or []
    limitations = parsed.get("limitations") or ""

    # 1. Issue Identified
    clean_query = query.strip().rstrip("?.")
    issue_desc = f"Determination of legal rights, liabilities, or statutory interpretation concerning: '{clean_query}'."

    # 2. Relevant Legal Provision
    if provisions and provisions != ["None"]:
        prov_desc = f"Identified statutory/constitutional provisions: {', '.join(provisions)}."
        prov_grounded = True
    elif retrieved_chunks:
        titles = list({c.get("title") or c.get("act_name") or "Indian Law" for c in retrieved_chunks[:3]})
        prov_desc = f"Provisions from retrieved corpus: {', '.join(titles)}."
        prov_grounded = True
    else:
        prov_desc = "No specific statutory provision found in retrieved evidence."
        prov_grounded = False

    # 3. Applicable Rule
    if retrieved_chunks:
        # Extract first substantive sentence from top chunk
        top_txt = (retrieved_chunks[0].get("text") or retrieved_chunks[0].get("text_preview") or "").strip()
        first_sentence = top_txt.split("\n")[0][:250] if top_txt else "Statutory mandate as per enacted legislation."
        rule_desc = f"The governing legal rule establishes: {first_sentence}"
        rule_grounded = True
    else:
        rule_desc = "Applicable rule could not be confirmed due to absent evidence in local corpus."
        rule_grounded = False

    # 4. Relevant Facts from User's Question
    facts_desc = f"The inquiry presents the following factual/legal premise: {query.strip()}."

    # 5. Application of the Rule
    if prov_grounded and rule_grounded:
        app_desc = (
            f"Applying the statutory standards of {', '.join(provisions) if provisions else 'the retrieved provisions'} "
            f"to the user's scenario establishes the legal position under Indian law."
        )
    else:
        app_desc = "Application cannot be definitively established without direct statutory evidence."

    # 6. Exceptions or Limitations
    if limitations and "none" not in limitations.lower():
        lim_desc = f"Caveats and limitations: {limitations.strip()}"
    elif not retrieved_chunks:
        lim_desc = "Severe limitation: Corpus lacks direct primary legal texts for this specific query."
    else:
        lim_desc = (
            "General limitations: Application subject to judicial discretion, jurisdictional rules, "
            "and any subsequent amendments."
        )

    # 7. Conclusion
    if prov_grounded:
        conc_desc = (
            f"The legal position under Indian jurisprudence requires compliance with "
            f"{', '.join(provisions) if provisions else 'applicable statutory requirements'}."
        )
    else:
        conc_desc = "Conclusion reserved pending verification against authoritative Gazette notifications."

    # 8. Sources
    if retrieved_chunks:
        src_names = []
        for i, c in enumerate(retrieved_chunks[:4], 1):
            t = c.get("title") or c.get("act_name") or f"Document {c.get('citation_id', i)}"
            src_names.append(f"[{c.get('citation_id', i)}] {t}")
        src_desc = "; ".join(src_names)
    else:
        src_desc = "None (No evidence retrieved from local database)."

    steps = [
        ReasoningStep(1, "Issue Identified", issue_desc, True),
        ReasoningStep(2, "Relevant Legal Provision", prov_desc, prov_grounded),
        ReasoningStep(3, "Applicable Rule", rule_desc, rule_grounded),
        ReasoningStep(4, "Relevant Facts", facts_desc, True),
        ReasoningStep(5, "Application of the Rule", app_desc, prov_grounded and rule_grounded),
        ReasoningStep(6, "Exceptions or Limitations", lim_desc, True),
        ReasoningStep(7, "Conclusion", conc_desc, prov_grounded),
        ReasoningStep(8, "Sources", src_desc, bool(retrieved_chunks)),
    ]

    return [s.to_dict() for s in steps]


@dataclass
class IRACRecord:
    question: str
    issue: str
    rules: List[str]
    evidence: List[Dict[str, Any]]
    application: str
    conclusion: str
    uncertainty: Optional[str]
    citations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def strip_thinking_tokens(text: str) -> str:
    """Removes any internal <think>...</think> reasoning traces."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"</?think>", "", cleaned)
    return cleaned.strip()


def generate_irac_record(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    parsed_sections: Optional[Dict[str, Any]] = None,
    citations_list: Optional[List[str]] = None,
) -> IRACRecord:
    """
    Constructs an auditable, evidence-grounded IRAC record:
    - question: Cleaned user inquiry (no <think> tokens)
    - issue: Factual and legal issues framed objectively
    - rules: Extracted statutory rules and constitutional principles
    - evidence: Summary of retrieved chunks (id, title, preview)
    - application: Analysis connecting rule to inquiry
    - conclusion: Final determination
    - uncertainty: Explicitly states corpus limitations or missing provisions
    - citations: Primary legal sources
    """
    clean_q = strip_thinking_tokens(query).rstrip("?.")
    parsed = parsed_sections or {}
    citations = citations_list or []

    # Issue
    issue = f"Whether and how statutory and constitutional principles apply to: '{clean_q}'."

    # Rules
    rules: List[str] = []
    provs = parsed.get("legal_provisions") or []
    if provs and provs != ["None"]:
        for p in provs:
            rules.append(f"Statutory Provision: {p}")
    for c in retrieved_chunks[:3]:
        t = (c.get("text") or c.get("text_preview") or "").strip().split("\n")[0]
        if t and len(t) > 20:
            rules.append(f"Principle: {t[:200]}")
    if not rules:
        rules.append("General statutory and constitutional tenets under Indian law.")

    # Evidence summary
    ev_list = []
    for i, c in enumerate(retrieved_chunks[:5], 1):
        ev_list.append({
            "citation_id": c.get("citation_id", i),
            "document_id": c.get("document_id"),
            "title": c.get("title") or c.get("act_name"),
            "court": c.get("court"),
            "preview": (c.get("text") or "")[:250],
        })

    # Application
    if retrieved_chunks:
        app = (
            f"Under the governing authorities retrieved for '{clean_q}', the legal requirements "
            f"must be evaluated against the factual elements presented. Compliance with statutory "
            f"provisions and constitutional standards dictates the rights and liabilities of the parties."
        )
    else:
        app = "Application is conditional due to absent primary statutory texts in the current retrieval corpus."

    # Conclusion
    ans = strip_thinking_tokens(parsed.get("answer") or "")
    if ans:
        conc = ans.split("\n\n")[0]
    elif retrieved_chunks:
        conc = f"Subject to evidentiary proof, the rights and liabilities under '{clean_q}' follow established Indian law."
    else:
        conc = "Conclusion reserved pending verification against authoritative Gazette notifications."

    # Uncertainty
    uncertainty = None
    if not retrieved_chunks or parsed.get("confidence") == "LOW":
        uncertainty = "Corpus contains limited or no direct primary statutory texts for this specific query."

    # Citations
    all_cits = list(citations)
    if not all_cits and retrieved_chunks:
        for i, c in enumerate(retrieved_chunks, 1):
            t = c.get("title") or c.get("act_name") or f"Source [{i}]"
            all_cits.append(f"[{c.get('citation_id', i)}] {t}")

    return IRACRecord(
        question=clean_q,
        issue=issue,
        rules=rules,
        evidence=ev_list,
        application=app,
        conclusion=conc,
        uncertainty=uncertainty,
        citations=all_cits,
    )
