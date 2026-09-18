"""
app/services/legal_extraction_service.py
========================================
Structured Legal Information and Entity Extraction Service.

Features:
- Precision extraction of Indian legal entities without hallucination.
- Strict null / empty list defaulting (no guessing).
- Document classification: Judgment, Legal Notice, Agreement/Contract, FIR, Statute.
- Extraction of parties, bench/judges, courts, case numbers, statutory sections, acts, citations, deadlines.
- Field-level uncertainty tracking and source grounding.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("legalmind.extraction")


@dataclass
class ExtractedEntity:
    value: str
    source_snippet: str
    confidence: float


@dataclass
class LegalExtractionResult:
    document_type: str = "legal_document"
    case_name: Optional[str] = None
    court: Optional[str] = None
    case_number: Optional[str] = None
    date: Optional[str] = None
    parties: List[str] = field(default_factory=list)
    judges: List[str] = field(default_factory=list)
    sections: List[str] = field(default_factory=list)
    acts: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    legal_issues: List[str] = field(default_factory=list)
    deadlines: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    confidence: str = "low"
    uncertain_fields: List[str] = field(default_factory=list)
    source_references: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "case_name": self.case_name,
            "court": self.court,
            "case_number": self.case_number,
            "date": self.date,
            "parties": self.parties,
            "judges": self.judges,
            "sections": self.sections,
            "acts": self.acts,
            "citations": self.citations,
            "legal_issues": self.legal_issues,
            "deadlines": self.deadlines,
            "locations": self.locations,
            "confidence": self.confidence,
            "uncertain_fields": self.uncertain_fields,
            "source_references": self.source_references,
        }


class LegalExtractionService:
    """Extracts structured Indian legal data using deterministic entity patterns."""

    @classmethod
    def classify_document(cls, text: str, filename: str = "") -> str:
        """Classifies document genre based on statutory and judicial anchors."""
        t_low = text.lower()
        f_low = filename.lower()

        if any(w in t_low for w in ["supreme court of india", "in the high court of", "writ petition", "special leave petition", "criminal appeal", "civil appeal"]):
            return "judgment"
        if any(w in t_low for w in ["legal notice", "statutory notice", "demand notice", "advocate notice", "under section 138"]):
            return "legal_notice"
        if any(w in t_low for w in ["first information report", "police station", "fir no"]):
            return "fir"
        if any(w in t_low for w in ["agreement", "contract", "memorandum of understanding", "between the parties", "whereas the client"]):
            return "contract"
        if any(w in t_low for w in ["be it enacted by parliament", "short title and commencement", "ministry of law and justice"]):
            return "statute"
        if "notice" in f_low:
            return "legal_notice"
        if "order" in f_low or "judgment" in f_low:
            return "judgment"
        return "legal_document"

    @classmethod
    def extract_entities(cls, text: str, filename: str = "") -> LegalExtractionResult:
        """Performs entity extraction with evidence references and uncertainty tracking."""
        res = LegalExtractionResult()
        res.document_type = cls.classify_document(text, filename)

        if not text or len(text.strip()) < 20:
            res.confidence = "low"
            res.uncertain_fields = ["case_name", "court", "case_number", "date", "parties"]
            return res

        confidence_score = 0.0
        uncertain_fields: List[str] = []
        source_refs: Dict[str, str] = {}

        # 1. Court Name
        court_match = re.search(r"\b(Supreme\s+Court\s+of\s+India|High\s+Court\s+of\s+[A-Za-z\s]+?)(?=\n|\r|,|\.|\s{2,})", text, re.I)
        if court_match:
            c_val = court_match.group(1).strip()
            if c_val.upper() == "SUPREME COURT OF INDIA":
                res.court = "Supreme Court of India"
            elif "HIGH COURT" in c_val.upper():
                res.court = c_val.title()
            else:
                res.court = c_val
            source_refs["court"] = court_match.group(0).strip()
            confidence_score += 0.25
        else:
            uncertain_fields.append("court")

        # 2. Case Number
        case_no_match = re.search(
            r"\b((?:Writ\s+Petition|Civil\s+Appeal|Criminal\s+Appeal|Special\s+Leave\s+Petition|SLP|FIR)\s*(?:\(C\))?\s*(?:No\.?|Number)?\s*[\d/]+(?:\s*(?:of|/)\s*\d{4})?)",
            text, re.I
        )
        if case_no_match:
            res.case_number = case_no_match.group(1).strip()
            source_refs["case_number"] = case_no_match.group(0).strip()
            confidence_score += 0.2
        else:
            uncertain_fields.append("case_number")

        # 3. Case Name / Parties
        vs_match = re.search(r"([A-Z][A-Za-z0-9 .,&()-]{2,60}?)\s+(?:versus|vs\.?|v\.)\s+([A-Z][A-Za-z0-9 .,&()-]{2,60})", text, re.I)
        if vs_match:
            party1 = vs_match.group(1).strip().split("\n")[-1].strip()
            party2 = vs_match.group(2).strip().split("\n")[0].strip()
            # Strip common suffixes
            party1 = re.sub(r"(?:^|\s+)(?:Appellant|Petitioner)s?\.?$", "", party1, flags=re.I).strip()
            party2 = re.sub(r"(?:^|\s+)(?:Respondent|Defendant)s?\.?$", "", party2, flags=re.I).strip()
            res.case_name = f"{party1} v. {party2}"
            res.parties = [party1, party2]
            source_refs["case_name"] = f"{party1} v. {party2}"
            confidence_score += 0.25
        else:
            uncertain_fields.append("case_name")
            uncertain_fields.append("parties")

        # 4. Bench / Judges
        judge_matches = re.findall(
            r"\b(?:HON(?:'BLE|\.?)?\s+(?:MR\.|MS\.|JUSTICE|C\.J\.)\s+([A-Z][A-Za-z\s.]+?)|CORAM\s*:\s*([A-Za-z\s,.]+))\b",
            text, re.I
        )
        judges_found = []
        for jm in judge_matches:
            val = (jm[0] or jm[1]).strip()
            if len(val) > 3 and val not in judges_found:
                judges_found.append(val)
        if judges_found:
            res.judges = judges_found[:4]
            source_refs["judges"] = ", ".join(res.judges)
            confidence_score += 0.15
        else:
            uncertain_fields.append("judges")

        # 5. Date
        date_match = re.search(
            r"\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December|[A-Za-z]{3})\s*,?\s*\d{4}|\d{1,2}[-/.](?:0[1-9]|1[0-2]|\d{1,2})[-/.]\d{2,4})\b",
            text, re.I
        )
        if date_match:
            res.date = date_match.group(1).strip()
            source_refs["date"] = date_match.group(0).strip()
            confidence_score += 0.15
        else:
            uncertain_fields.append("date")

        # 6. Statutory Acts
        act_patterns = [
            r"\bConstitution\s+of\s+India\b",
            r"\b[A-Z][A-Za-z\s]+Act,\s*\d{4}\b",
            r"\bIndian\s+Penal\s+Code\b",
            r"\bCode\s+of\s+Criminal\s+Procedure\b",
            r"\bCode\s+of\s+Civil\s+Procedure\b",
            r"\bBharatiya\s+Nyaya\s+Sanhita\b",
            r"\bBharatiya\s+Nagarik\s+Suraksha\s+Sanhita\b",
            r"\bBharatiya\s+Sakshya\s+Adhiniyam\b",
            r"\bPrevention\s+of\s+Money\s+Laundering\s+Act\b",
            r"\bNegotiable\s+Instruments\s+Act\b",
        ]
        found_acts = []
        for p in act_patterns:
            for m in re.findall(p, text, re.I):
                clean_act = m.strip()
                if clean_act not in found_acts:
                    found_acts.append(clean_act)
        res.acts = found_acts[:8]
        if res.acts:
            confidence_score += 0.1

        # 7. Statutory Sections & Articles
        sec_matches = re.findall(r"\b(?:Section|Sec\.|Article|Art\.)\s*(\d+[A-Za-z]*(?:\(\d+[A-Za-z]*\))?)\b", text, re.I)
        found_secs = []
        for s in sec_matches:
            val = f"Section {s}" if "Section" not in s else s
            if val not in found_secs:
                found_secs.append(val)
        res.sections = found_secs[:12]

        # 8. Citations (AIR, SCC, SCR)
        cit_matches = re.findall(r"\b(\d{4}\s+(?:AIR|SCC|SCR|Scale|INSC)\s+\d+(?:\s+SC)?)\b", text, re.I)
        res.citations = list(dict.fromkeys(cit_matches))[:6]

        # 9. Deadlines & Legal Notice Demands
        deadline_matches = re.findall(r"\b(within\s+\d+\s+(?:days|hours|weeks)|on\s+or\s+before\s+[\d/A-Za-z\s,]+)\b", text, re.I)
        res.deadlines = [d.strip() for d in list(dict.fromkeys(deadline_matches))[:4]]

        # Confidence label
        if confidence_score >= 0.70 and res.case_number and res.court:
            res.confidence = "high"
        elif confidence_score >= 0.35:
            res.confidence = "medium"
        else:
            res.confidence = "low"

        res.uncertain_fields = uncertain_fields
        res.source_references = source_refs
        return res
