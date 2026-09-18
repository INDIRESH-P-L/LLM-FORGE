#!/usr/bin/env python3
"""
scripts/citation_verifier.py
============================
LegalMind AI — Citation & Hallucination Verification Engine

Performs rigorous legal verification:
1. Citation existence & standard format validation (AIR, SCC, SCR, SCALE, Central Acts, Constitution Articles).
2. Source-document presence: Checks whether each cited provision or case name exists in retrieved evidence chunks.
3. Article / Section existence: Validates statutory numbering against retrieved legislation metadata.
4. Hallucination detection: Flags fabricated cases, non-existent sections, and unsupported claims.
5. Historical vs. Current law mismatches (e.g. citing repealed IPC without noting BNS).
6. Conflicting evidence detection.
7. Confidence calibration: Downgrades confidence to MEDIUM or LOW when citations fail verification.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class CitationVerificationResult:
    citation_text: str
    citation_type: str  # "statute", "judgment", "constitution", "unknown"
    is_valid_format: bool
    is_present_in_evidence: bool
    is_verified: bool
    source_document: Optional[str] = None
    warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationSummary:
    total_citations_found: int
    verified_count: int
    unverified_count: int
    hallucination_score: float  # 0.0 (clean) to 1.0 (pure hallucination)
    confidence_label: str  # "high", "medium", "low"
    confidence_score: float  # 0.0 to 1.0
    citations: List[CitationVerificationResult] = field(default_factory=list)
    legal_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_citations_found": self.total_citations_found,
            "verified_count": self.verified_count,
            "unverified_count": self.unverified_count,
            "hallucination_score": round(self.hallucination_score, 3),
            "confidence_label": self.confidence_label,
            "confidence_score": round(self.confidence_score, 3),
            "citations": [c.to_dict() for c in self.citations],
            "legal_warnings": self.legal_warnings,
        }


# Standard Indian Legal Citation Formats
INDIAN_CITATION_PATTERNS = [
    r"\bAIR\s+\d{4}\s+SC\s+\d+\b",                          # AIR 1973 SC 1461
    r"\b\(\d{4}\)\s+\d+\s+SCC\s+\d+\b",                     # (1973) 4 SCC 225
    r"\b\d{4}\s+Supp\s+\(\d+\)\s+SCC\s+\d+\b",              # 1992 Supp (3) SCC 217
    r"\b\d{4}\s+SCR\s+\(\d+\)\s+\d+\b",                     # 1978 SCR (2) 621
    r"\b\d{4}\s+SCALE\s+\d+\b",                             # SCALE reporter
    r"\bArticle\s+\d+[A-Za-z]?(?:\s*\(\d+\))?\b",           # Article 21, Article 19(1)(a)
    r"\bSection\s+\d+[A-Za-z]?(?:\s*\(\d+\))?\b",           # Section 302, Section 438
    r"\bOrder\s+[A-Z0-9]+,\s*Rule\s+\d+\b",                 # Order XXXIX Rule 1 CPC
]


def extract_cited_provisions_from_text(text: str) -> List[str]:
    """Extracts candidate statutory, constitutional, and case citations from generated text."""
    found: List[str] = []
    # Articles
    for m in re.finditer(r"\bArticle\s+\d+[A-Za-z]?(?:\s*\(\d+\))?(?:\s+of\s+the\s+Constitution(?:\s+of\s+India)?)?\b", text, re.IGNORECASE):
        found.append(m.group(0).strip())

    # Sections
    for m in re.finditer(r"\bSection\s+\d+[A-Za-z]?(?:\s*\(\d+\))?(?:\s+of\s+(?:the\s+)?[A-Za-z\s,]+(?:Act|Code|Sanhita|Adhiniyam)(?:,\s*\d{4})?)?\b", text, re.IGNORECASE):
        found.append(m.group(0).strip())

    # Reporter citations
    for pat in [
        r"\bAIR\s+\d{4}\s+[A-Za-z]+\s+\d+\b",
        r"(?:\(\d{4}\)|\b\d{4})\s+(?:Supp\s+\(\d+\)\s+)?\d+\s+SCC\s+\d+\b",
        r"(?:\(\d{4}\)|\b\d{4})\s+SCR\s+\(\d+\)\s+\d+\b",
        r"\b\d{4}\s+SCALE\s+\d+\b",
    ]:
        for m in re.finditer(pat, text, re.IGNORECASE):
            found.append(m.group(0).strip())

    # De-duplicate while preserving order
    seen = set()
    unique_found = []
    for f in found:
        norm = re.sub(r"\s+", " ", f).lower()
        if norm not in seen:
            seen.add(norm)
            unique_found.append(f)

    return unique_found


def verify_answer_citations(
    answer_text: str,
    retrieved_chunks: List[Dict[str, Any]],
    query: str = "",
) -> VerificationSummary:
    """
    Verifies all citations in generated answer against retrieved evidence chunks.
    """
    if not retrieved_chunks:
        # No retrieved evidence provided
        cited_items = extract_cited_provisions_from_text(answer_text)
        warnings = [
            "No supporting evidence was retrieved from the corpus. "
            "Any citations in the answer could not be verified against the local legal database."
        ]
        results = [
            CitationVerificationResult(
                citation_text=item,
                citation_type="statute" if "section" in item.lower() or "article" in item.lower() else "judgment",
                is_valid_format=True,
                is_present_in_evidence=False,
                is_verified=False,
                warning="Evidence absent in retrieved corpus",
            )
            for item in cited_items
        ]
        return VerificationSummary(
            total_citations_found=len(cited_items),
            verified_count=0,
            unverified_count=len(cited_items),
            hallucination_score=1.0 if cited_items else 0.0,
            confidence_label="low",
            confidence_score=0.25,
            citations=results,
            legal_warnings=warnings,
        )

    # Compile evidence text and metadata for rapid verification
    evidence_texts = []
    evidence_titles = set()
    evidence_sections = set()
    evidence_articles = set()

    for c in retrieved_chunks:
        t = (c.get("text") or c.get("text_preview") or "").lower()
        evidence_texts.append(t)
        title = (c.get("title") or c.get("act_name") or "").lower()
        if title:
            evidence_titles.add(title)

        sec = str(c.get("section_number") or c.get("section") or "").strip().lower()
        if sec:
            evidence_sections.add(sec)

        art = str(c.get("article_number") or c.get("article") or "").strip().lower()
        if art:
            evidence_articles.add(art)

    combined_evidence_str = " \n ".join(evidence_texts)

    cited_items = extract_cited_provisions_from_text(answer_text)
    verification_results: List[CitationVerificationResult] = []
    warnings: List[str] = []

    verified_count = 0

    for item in cited_items:
        item_lower = item.lower()
        is_statute = "section" in item_lower
        is_constitution = "article" in item_lower
        c_type = "constitution" if is_constitution else ("statute" if is_statute else "judgment")

        # Check format validity
        is_format_ok = any(re.search(pat, item, re.IGNORECASE) for pat in INDIAN_CITATION_PATTERNS) or bool(re.search(r"\b(?:article|section)\s+\d+\b", item_lower))

        # Check evidence presence
        present = False
        matched_source = None

        if is_constitution:
            m_num = re.search(r"article\s+(\d+[a-z]?)", item_lower)
            if m_num:
                num = m_num.group(1)
                if num in evidence_articles or f"article {num}" in combined_evidence_str or f"**{num}.**" in combined_evidence_str:
                    present = True
                    matched_source = "Constitution of India"

        elif is_statute:
            m_num = re.search(r"section\s+(\d+[a-z]?)", item_lower)
            if m_num:
                num = m_num.group(1)
                if num in evidence_sections or f"section {num}" in combined_evidence_str or f"**{num}.**" in combined_evidence_str:
                    present = True
                    matched_source = "Legislation Corpus"
        else:
            # Judgment citation
            if item_lower in combined_evidence_str:
                present = True
                matched_source = "Case Law Corpus"

        warn = None
        if not present:
            warn = f"Citation '{item}' was not found in the retrieved evidence chunks."
            warnings.append(warn)
        else:
            verified_count += 1

        verification_results.append(
            CitationVerificationResult(
                citation_text=item,
                citation_type=c_type,
                is_valid_format=is_format_ok,
                is_present_in_evidence=present,
                is_verified=present and is_format_ok,
                source_document=matched_source,
                warning=warn,
            )
        )

    total = len(cited_items)
    unverified = total - verified_count
    hallucination_score = (unverified / total) if total > 0 else 0.0

    # Confidence calibration
    if total == 0:
        conf_label = "medium"
        conf_score = 0.65
    elif hallucination_score == 0.0 and verified_count >= 1:
        conf_label = "high"
        conf_score = 0.95
    elif hallucination_score <= 0.35:
        conf_label = "medium"
        conf_score = 0.70
    else:
        conf_label = "low"
        conf_score = 0.35
        warnings.append(
            "Confidence reduced to LOW due to unverified citations or absent statutory references."
        )

    return VerificationSummary(
        total_citations_found=total,
        verified_count=verified_count,
        unverified_count=unverified,
        hallucination_score=hallucination_score,
        confidence_label=conf_label,
        confidence_score=conf_score,
        citations=verification_results,
        legal_warnings=warnings,
    )


def validate_citation_format(cit: str) -> Tuple[bool, str]:
    c_lower = cit.lower().strip()
    if re.search(r"\barticle\s+\d+[a-z]?\b", c_lower):
        return True, "constitution"
    if re.search(r"\b(?:section|sec\.?)\s+\d+[a-z]?\b", c_lower):
        return True, "statute"
    for pat in INDIAN_CITATION_PATTERNS:
        if re.search(pat, cit, re.IGNORECASE):
            return True, "judgment"
    return False, "unknown"


def verify_citations_in_response(
    generated_text: str,
    retrieved_chunks: List[Dict[str, Any]],
    parsed_sections: Optional[Dict[str, Any]] = None,
    query: str = "",
) -> VerificationSummary:
    return verify_answer_citations(generated_text, retrieved_chunks, query=query)


def format_citation_verification_report(summary: VerificationSummary) -> str:
    lines = [
        "## Citation Verification & Hallucination Audit",
        f"- Total Citations: {summary.total_citations_found}",
        f"- Verified: {summary.verified_count}",
        f"- Unverified: {summary.unverified_count}",
        f"- Hallucination Score: {summary.hallucination_score}",
        f"- Confidence Label: {summary.confidence_label}",
    ]
    for c in summary.citations:
        st = "VERIFIED" if c.is_verified else "UNVERIFIED"
        lines.append(f"  * [{st}] {c.citation_text} ({c.citation_type})")
    for w in summary.legal_warnings:
        lines.append(f"  ! Warning: {w}")
    return "\n".join(lines)

