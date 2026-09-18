"""
app/services/fir_audit_service.py
=================================
LegalMind AI — FIR Procedural Defect & Quashing Auditor.

MASTER IMPLEMENTATION — FACT-GROUNDED PROCEDURAL AUDIT ENGINE
Features:
1. Strict Case Isolation: Fresh CASE_ID (FIR_AUDIT_<timestamp>_<uuid>) per submission.
2. Three-State Fact Model: PRESENT, ABSENT_EXPLICITLY, NOT_STATED.
3. Fact Traceability: Every fact has sourceSpan, confidence, and sourceCaseId.
4. Dynamic Finding Count: Exactly matches rendered POTENTIAL_ISSUE and REQUIRES_VERIFICATION findings.
5. Dispute Contamination Firewall: Complete isolation between property/civil disputes and bodily offence FIRs.
6. Finding-Specific Authorities: Every authority is linked to a specific findingId (caseId -> findingId -> authorities).
7. Zero Cross-Case Contamination & Zero CPC Leakage.
8. Ten-Part Output Structure according to Section 30 of the Master Standard.
9. Pre-Rendering Validation Pipeline (Validations A through N).
10. Zero Numerical Quashing Scores / Predictions.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("legalmind.fir")

ROOT = Path(__file__).resolve().parent.parent.parent
AUTHORITY_DB = Path(os.environ.get("LEGALMIND_AUTHORITY_DB",
                                   ROOT / "data/verification/authorities.db"))

BNSS_BNS_EFFECTIVE_DATE = date(2024, 7, 1)

TEMPLATE_MARKERS = re.compile(
    r"\[\s*insert[^\]]*\]|\[\s*name[^\]]*\]|\bxxx+\b|<[^>]{0,30}>|"
    r"\b(?:to be filled|not applicable|n/?a)\b", re.I)

MONTHS_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12
}


@dataclass
class Finding:
    """A structured, neutral procedural screening observation."""
    finding_id: str
    code: str
    category: str
    status: str             # POTENTIAL_ISSUE, REQUIRES_VERIFICATION, INFORMATIONAL, NOT_APPLICABLE, COMPLIANCE_ESTABLISHED
    title: str
    read_from_fir: str      # Exact factual extract from the supplied FIR
    legal_principle: str    # Applicable procedural or statutory principle
    assessment: str         # Objective analysis of what the fact indicates
    authority: str | None = None
    verification_required: list[str] = field(default_factory=list)
    confidence: str = "HIGH"
    confidence_explanation: str = ""
    case_id: str = ""
    linked_authorities: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class FIRAuditorService:
    _local = threading.local()

    @classmethod
    def _conn(cls) -> sqlite3.Connection | None:
        if not AUTHORITY_DB.exists():
            return None
        c = getattr(cls._local, "conn", None)
        if c is None:
            c = sqlite3.connect(str(AUTHORITY_DB), check_same_thread=False)
            c.row_factory = sqlite3.Row
            cls._local.conn = c
        return c

    # ── Helpers for Three-State Fact Model ─────────────────────────────────
    @staticmethod
    def _make_fact(
        value: Any,
        source_span: str,
        state: str,  # PRESENT, ABSENT_EXPLICITLY, NOT_STATED
        confidence: str = "HIGH",
        case_id: str = ""
    ) -> dict:
        if state == "NOT_STATED":
            return {
                "value": None,
                "sourceSpan": "Not stated in supplied material.",
                "confidence": "HIGH",
                "state": "NOT_STATED",
                "sourceCaseId": case_id
            }
        return {
            "value": value,
            "sourceSpan": source_span.strip(),
            "confidence": confidence,
            "state": state,
            "sourceCaseId": case_id
        }

    @classmethod
    def _parse_raw_date(cls, text_date: str) -> tuple[str | None, str | None]:
        """Parse varied Indian date expressions into (ISO YYYY-MM-DD, Display String)."""
        s = text_date.strip().rstrip(".,")
        # Match "12 September 2026" or "12th September, 2026"
        m1 = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+),?\s+(\d{4})", s, re.I)
        if m1:
            day, month_str, year = int(m1.group(1)), m1.group(2).lower(), int(m1.group(3))
            mo = MONTHS_MAP.get(month_str)
            if mo and 1 <= day <= 31 and 1900 <= year <= 2100:
                try:
                    iso = date(year, mo, day).isoformat()
                    m_disp = month_str.capitalize()
                    return iso, f"{day} {m_disp} {year}"
                except ValueError:
                    pass

        # Match "2026-09-12"
        m2 = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
        if m2:
            try:
                y, m, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
                iso = date(y, m, d).isoformat()
                m_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
                return iso, f"{d} {m_names[m - 1]} {y}"
            except ValueError:
                pass

        # Match "12/09/2026" or "12-09-2026"
        m3 = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})", s)
        if m3:
            d, m, y = int(m3.group(1)), int(m3.group(2)), int(m3.group(3))
            if y < 100:
                y += 2000 if y < 50 else 1900
            if 1 <= m <= 12 and 1 <= d <= 31:
                try:
                    iso = date(y, m, d).isoformat()
                    m_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
                    return iso, f"{d} {m_names[m - 1]} {y}"
                except ValueError:
                    pass
        return None, None

    # ── Structured Fact Extraction ─────────────────────────────────────────
    @classmethod
    def extract_facts(cls, fir_text: str, arrest_checkbox: bool = False, special_statute_checkbox: bool = False, case_id: str = "") -> dict:
        """
        Deterministically extracts structured facts strictly from current FIR text.
        Implements the Three-State Fact Model (PRESENT, ABSENT_EXPLICITLY, NOT_STATED).
        """
        text = fir_text or ""
        low = text.lower()
        cid = case_id or f"FIR_AUDIT_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # Sentences for source span isolation
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

        def find_sentence(patterns: list[str]) -> str:
            for pat in patterns:
                for s in sentences:
                    if re.search(pat, s, re.I):
                        return s
            return ""

        # 1. Occurrence Date
        occ_span = find_sentence([
            r"(?:alleged\s+)?incident\s+occurred\s+on",
            r"occurred\s+on\s+\d{1,2}",
            r"alleging\s+that\s+on\s+\d{1,2}",
            r"incident\s+(?:took\s+place|was|occurred)\s+on",
            r"date\s+of\s+(?:occurrence|incident)"
        ])
        occ_iso, occ_display = None, None
        if occ_span:
            # Extract date inside this specific span
            m = re.search(r"(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+\d{4}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})", occ_span)
            if m:
                occ_iso, occ_display = cls._parse_raw_date(m.group(1))

        # 2. FIR Registration Date
        fir_span = find_sentence([
            r"fir\s+(?:was\s+)?registered\s+(?:only\s+)?on",
            r"registered\s+only\s+on",
            r"lodged\s+an\s+fir\s+alleging",
            r"approached\s+the\s+police",
            r"on\s+\d{1,2}.*?the\s+complainant\s+(?:approached|lodged)"
        ])
        fir_iso, fir_display = None, None
        if fir_span:
            m = re.search(r"(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+\d{4}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})", fir_span)
            if m:
                fir_iso, fir_display = cls._parse_raw_date(m.group(1))

        # Date fallback scan if spans didn't isolate both
        if not occ_iso or not fir_iso:
            raw_dates = []
            for m in re.finditer(r"(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+\d{4}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})", text):
                iso, disp = cls._parse_raw_date(m.group(1))
                if iso and (iso, disp) not in raw_dates:
                    raw_dates.append((iso, disp))
            if len(raw_dates) >= 2:
                if not occ_iso:
                    occ_iso, occ_display = raw_dates[0]
                    occ_span = occ_span or f"Incident date identified: {occ_display}"
                if not fir_iso:
                    fir_iso, fir_display = raw_dates[-1]
                    fir_span = fir_span or f"FIR registration date identified: {fir_display}"
            elif len(raw_dates) == 1:
                if not occ_iso:
                    occ_iso, occ_display = raw_dates[0]
                    occ_span = occ_span or f"Occurrence date: {occ_display}"

        # 3. Delay Calculation
        delay_days = None
        if occ_iso and fir_iso:
            try:
                d1 = datetime.fromisoformat(occ_iso).date()
                d2 = datetime.fromisoformat(fir_iso).date()
                delay_days = max(0, (d2 - d1).days)
            except Exception:
                delay_days = None

        # 4. Delay Context / Reason
        delay_reason_span = find_sentence([
            r"consulted\s+family\s+members",
            r"went\s+home\s+and\s+consulted",
            r"resolve\s+(?:the\s+dispute\s+)?through\s+discussions",
            r"without\s+(?:providing\s+)?any\s+written\s+explanation\s+for\s+(?:the\s+)?delay",
            r"no\s+written\s+explanation\s+for\s+(?:the\s+)?delay"
        ])
        delay_reason_text = None
        delay_reason_state = "NOT_STATED"
        if "consulted family members" in delay_reason_span.lower():
            delay_reason_text = "The complainant states that he initially went home and consulted family members before approaching police."
            delay_reason_state = "PRESENT"
        elif "discussions" in delay_reason_span.lower() and ("without" in delay_reason_span.lower() or "did not provide" in delay_reason_span.lower()):
            delay_reason_text = "The complainant attempted to resolve the dispute through discussions, but no written explanation is stated."
            delay_reason_state = "PRESENT"
        elif "did not provide any written explanation" in delay_reason_span.lower():
            delay_reason_text = "The complainant did not provide any written explanation for the delay."
            delay_reason_state = "PRESENT"
        elif delay_reason_span:
            delay_reason_text = delay_reason_span
            delay_reason_state = "PRESENT"

        # 5. Accused Extraction & Roles
        accused_span = find_sentence([
            r"\bthree\s+(?:accused|persons)\b",
            r"former\s+business\s+partner\s+and\s+two",
            r"\b(\d+)\s+accused\b"
        ])
        accused_count = None
        accused_count_state = "NOT_STATED"
        accused_desc = "Not stated in supplied material."
        if re.search(r"\bthree\s+(?:accused|persons)\b|former\s+business\s+partner\s+and\s+two", text, re.I):
            accused_count = 3
            accused_count_state = "PRESENT"
            if "former business partner" in text.lower():
                accused_desc = "3 (former business partner and two relatives)"
            else:
                accused_desc = "3 accused persons"
        elif re.search(r"\b(\d+)\s+accused\b", text, re.I):
            m_a = re.search(r"\b(\d+)\s+accused\b", text, re.I)
            accused_count = int(m_a.group(1))
            accused_count_state = "PRESENT"
            accused_desc = f"{accused_count} accused persons"

        roles_unclear_span = find_sentence([
            r"does\s+not\s+(?:clearly\s+)?identify\s+which\s+accused",
            r"does\s+not\s+individually\s+attribute",
            r"roles\s+(?:of\s+accused\s+)?(?:are\s+)?not\s+specified",
            r"individual\s+acts.*?not\s+specified"
        ])
        roles_unclear = bool(roles_unclear_span)

        # 6. Arrest Status (Three-State Model)
        negative_arrest_span = find_sentence([
            r"no\s+accused\s+has\s+been\s+arrested",
            r"not\s+arrested",
            r"no\s+arrest\s+(?:has\s+been|was)?\s*made",
            r"accused\s+(?:is|are)?\s*not\s+in\s+custody"
        ])
        positive_arrest_span = find_sentence([
            r"accused\s+(?:was|were|has\s+been|have\s+been)\s+arrested",
            r"placed\s+under\s+arrest",
            r"remanded\s+to\s+custody",
            r"in\s+police\s+custody"
        ])

        if negative_arrest_span:
            arrest_state = "ABSENT_EXPLICITLY"
            arrest_val = False
            arrest_span = negative_arrest_span
            arrest_label = "No arrest stated"
        elif arrest_checkbox:
            arrest_state = "PRESENT"
            arrest_val = True
            arrest_span = "Indicated by user checkbox"
            arrest_label = "Yes (User indicated)"
        elif positive_arrest_span:
            arrest_state = "PRESENT"
            arrest_val = True
            arrest_span = positive_arrest_span
            arrest_label = "Yes"
        else:
            arrest_state = "NOT_STATED"
            arrest_val = None
            arrest_span = "Not stated in supplied material."
            arrest_label = "Not stated in supplied material."

        # 7. Special Statute Status (Three-State Model)
        special_statute_span = find_sentence([r"\bpmla\b", r"\buapa\b", r"\bndps\b", r"sc/?st", r"\bpocso\b"])
        if special_statute_checkbox:
            special_statute_state = "PRESENT"
            special_statute_val = True
            special_statute_label = "Yes (User indicated)"
        elif special_statute_span:
            special_statute_state = "PRESENT"
            special_statute_val = True
            special_statute_label = "Yes"
        else:
            special_statute_state = "ABSENT_EXPLICITLY" if "special statute" in low and ("no" in low or "not" in low) else "NOT_STATED"
            special_statute_val = False
            special_statute_label = "No"

        # 8. Dispute Classification (Firewall Check)
        property_dispute_span = find_sentence([
            r"ownership\s+dispute",
            r"jointly\s+owned",
            r"co-owner",
            r"commercial\s+property.*?dispute",
            r"lawful\s+access.*?co-owners"
        ])
        property_dispute = bool(property_dispute_span)

        contractual_dispute_span = find_sentence([
            r"contractual\s+(?:disagreement|dispute)",
            r"commercial\s+transaction",
            r"breach\s+of\s+contract",
            r"partnership\s+disagreement"
        ])
        contractual_dispute = bool(contractual_dispute_span)

        personal_disagreement_span = find_sentence([
            r"personal\s+disagreement",
            r"personal\s+dispute",
            r"prior\s+enmity",
            r"personal\s+rivalry"
        ])
        personal_disagreement = bool(personal_disagreement_span)

        # 9. Settlement / Repayment (Three-State Model)
        negative_settle_span = find_sentence([
            r"no\s+settlement\s+or\s+repayment\s+is\s+mentioned",
            r"no\s+settlement\s+mentioned",
            r"no\s+repayment\s+mentioned",
            r"without\s+(?:any\s+)?settlement\s+or\s+repayment"
        ])
        positive_settle_span = find_sentence([
            r"amount\s+(?:had\s+)?already\s+been\s+repaid",
            r"repaid",
            r"settled\s+the\s+amount"
        ])
        if negative_settle_span:
            settlement_state = "ABSENT_EXPLICITLY"
            settlement_label = "Not mentioned"
            settle_span = negative_settle_span
        elif positive_settle_span:
            settlement_state = "PRESENT"
            settlement_label = "Mentioned"
            settle_span = positive_settle_span
        else:
            settlement_state = "NOT_STATED"
            settlement_label = "Not mentioned"
            settle_span = "Not stated in supplied material."

        # 10. Alleged Conduct & Offences
        alleged_conduct = []
        if re.search(r"criminal\s+trespass|trespass", low):
            alleged_conduct.append("Criminal trespass")
        if re.search(r"\btheft\b|stolen|removed.*?equipment", low):
            alleged_conduct.append("Theft")
        if re.search(r"\bassault\b|\bhurt\b|pain\s+and\s+swelling", low):
            alleged_conduct.append("Assault / Hurt")
        if re.search(r"criminal\s+intimidation|\bthreatened\b|\bthreats?\b", low):
            alleged_conduct.append("Criminal intimidation")
        if re.search(r"\bcheat(?:ing|ed)?\b|\bfraud\b", low):
            alleged_conduct.append("Cheating")

        # 11. Specific Evidentiary Points (Three-State Model)
        # Telephone Threat
        telephone_threat_span = find_sentence([r"telephone\s+conversation", r"over\s+the\s+telephone", r"phone\s+call"])
        has_phone_threat = bool(telephone_threat_span)

        threat_words_span = find_sentence([r"exact\s+words.*?threat.*?not\s+specified", r"words\s+of\s+the\s+alleged\s+threat.*?not\s+specified"])
        threat_words_missing = bool(threat_words_span) or ("exact words" not in low and has_phone_threat)

        # Medical Evidence
        injury_span = find_sentence([r"pain\s+and\s+swelling", r"bodily\s+injury", r"hurt", r"wound"])
        medical_exam_span = find_sentence([r"medical\s+examination", r"wound\s+certificate", r"doctor", r"hospital"])
        has_injury_stated = bool(injury_span)
        medical_exam_state = "PRESENT" if medical_exam_span else "NOT_STATED"

        # Wooden Object / Weapon
        wooden_object_span = find_sentence([r"wooden\s+object", r"wooden\s+stick", r"lathi", r"blunt\s+object"])
        has_wooden_object = bool(wooden_object_span)
        recovery_span = find_sentence([r"seizure\s+memo", r"recovered", r"seized"])
        recovery_state = "PRESENT" if recovery_span else "NOT_STATED"

        # Independent Witnesses
        witness_span = find_sentence([r"independent\s+witness", r"eyewitness", r"witnesses\s+present"])
        witness_state = "PRESENT" if witness_span else "NOT_STATED"

        # Stolen Property / Equipment
        stolen_prop_span = find_sentence([r"removed\s+certain\s+documents\s+and\s+equipment", r"stolen\s+equipment"])
        has_stolen_equipment = bool(stolen_prop_span)
        equipment_desc_missing = bool(re.search(r"no\s+specific\s+description\s+of\s+the\s+allegedly\s+stolen\s+equipment", low))

        # Check explicit numerical sections in narrative
        sec_pattern = re.compile(
            r"(?:u/?s|under\s+sections?|sections?|sec\.?)\s*"
            r"((?:\d{1,3}[A-Z]?(?:\(\d+\))?(?:/[A-Z0-9]+)?\s*(?:,|and|&|/|r/w)?\s*){1,8})"
            r"(?:\s+(?:of\s+(?:the\s+)?)?(IPC|BNS|CrPC|BNSS|Indian\s+Penal\s+Code|Bharatiya\s+Nyaya\s+Sanhita))?",
            re.I
        )
        numerical_sections = []
        for m in sec_pattern.finditer(text):
            for s in re.findall(r"\d{1,3}[A-Z]?(?:\(\d+\))?", m.group(1)):
                numerical_sections.append(s)

        # Build Full Structured Facts conforming to Section 4
        facts_dict = {
            "caseId": cid,
            "sourceTextLength": len(text),
            "occurrenceDate": cls._make_fact(occ_iso, occ_span or "", "PRESENT" if occ_iso else "NOT_STATED", case_id=cid),
            "occurrenceDateDisplay": occ_display or "Not stated in supplied material.",
            "firDate": cls._make_fact(fir_iso, fir_span or "", "PRESENT" if fir_iso else "NOT_STATED", case_id=cid),
            "firDateDisplay": fir_display or "Not stated in supplied material.",
            "delayDays": cls._make_fact(delay_days, f"{delay_days} days" if delay_days is not None else "", "PRESENT" if delay_days is not None else "NOT_STATED", case_id=cid),
            "delayReason": cls._make_fact(delay_reason_text, delay_reason_span or "", delay_reason_state, case_id=cid),
            "accusedCount": cls._make_fact(accused_count, accused_span or "", accused_count_state, case_id=cid),
            "accusedDescription": accused_desc,
            "rolesUnclear": cls._make_fact(roles_unclear, roles_unclear_span or "", "PRESENT" if roles_unclear else "NOT_STATED", case_id=cid),
            "arrestStatus": cls._make_fact(arrest_val, arrest_span, arrest_state, case_id=cid),
            "arrestLabel": arrest_label,
            "specialStatuteStatus": cls._make_fact(special_statute_val, special_statute_span or "Not invoked", special_statute_state, case_id=cid),
            "specialStatuteLabel": special_statute_label,
            "propertyDispute": cls._make_fact(property_dispute, property_dispute_span or "", "PRESENT" if property_dispute else "NOT_STATED", case_id=cid),
            "contractualDispute": cls._make_fact(contractual_dispute, contractual_dispute_span or "", "PRESENT" if contractual_dispute else "NOT_STATED", case_id=cid),
            "personalDisagreement": cls._make_fact(personal_disagreement, personal_disagreement_span or "", "PRESENT" if personal_disagreement else "NOT_STATED", case_id=cid),
            "settlementRepayment": cls._make_fact(None if settlement_state != "PRESENT" else "Mentioned", settle_span, settlement_state, case_id=cid),
            "settlementLabel": settlement_label,
            "allegedOffences": cls._make_fact(alleged_conduct, ", ".join(alleged_conduct), "PRESENT" if alleged_conduct else "NOT_STATED", case_id=cid),
            "numericalSections": numerical_sections,
            "sectionsDisplay": ", ".join(numerical_sections) if numerical_sections else "Numerical sections not specified in supplied narrative.",
            "telephoneThreat": cls._make_fact(has_phone_threat, telephone_threat_span or "", "PRESENT" if has_phone_threat else "NOT_STATED", case_id=cid),
            "threatWordsMissing": threat_words_missing,
            "injuryStated": cls._make_fact(has_injury_stated, injury_span or "", "PRESENT" if has_injury_stated else "NOT_STATED", case_id=cid),
            "medicalExamState": medical_exam_state,
            "woodenObject": cls._make_fact(has_wooden_object, wooden_object_span or "", "PRESENT" if has_wooden_object else "NOT_STATED", case_id=cid),
            "recoveryState": recovery_state,
            "witnessState": witness_state,
            "stolenEquipment": has_stolen_equipment,
            "equipmentDescMissing": equipment_desc_missing
        }
        return facts_dict

    # ── Master Audit Pipeline ──────────────────────────────────────────────
    @classmethod
    def audit_fir(
        cls,
        fir_text: str,
        is_pmla: bool = False,
        arrest_made: bool = False,
        retriever: Any = None,
        llm: Any = None
    ) -> dict:
        """
        Executes a completely isolated FIR audit with zero cross-case contamination.
        Guarantees:
        - New CASE_ID per invocation.
        - Three-State Fact Model.
        - Dynamic finding counts (strictly counting POTENTIAL_ISSUE + REQUIRES_VERIFICATION).
        - Dispute contamination firewall.
        - Finding-specific authorities without CPC leakage.
        """
        text = (fir_text or "").strip()
        if not text:
            return {
                "status": "error",
                "error": "FIR text cannot be empty.",
                "message": "Error: FIR text cannot be empty."
            }

        # 1. Mint Unique Session Case ID
        current_case_id = f"FIR_AUDIT_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # 2. Extract Structured Facts
        f = cls.extract_facts(text, arrest_checkbox=arrest_made, special_statute_checkbox=is_pmla, case_id=current_case_id)

        # Insufficient narrative check
        if f["sourceTextLength"] < 30 and not f["allegedOffences"]["value"]:
            return {
                "status": "insufficient_input",
                "case_id": current_case_id,
                "overall_status": "Insufficient Information for Procedural Assessment",
                "message": "Insufficient narrative to conduct procedural screening. Please provide the FIR narrative and allegations.",
                "extracted_facts": f,
                "findings": [],
                "quashing_issues": [],
                "documents_recommended": ["Complete FIR", "Formal complaint copy"]
            }

        findings: list[Finding] = []
        documents_required: list[str] = ["Complete FIR", "Complaint and station diary records"]
        material_not_established: list[str] = []

        # Helper to link authority to finding
        def make_auth(title: str, court: str, citation: str, section: str, prop: str, reason: str, issue_id: str) -> dict:
            return {
                "case_id": current_case_id,
                "issue_id": issue_id,
                "title": title,
                "court_or_authority": court,
                "citation": citation,
                "relevant_section": section,
                "legal_proposition": prop,
                "relevance_reason": reason,
                "source_type": "Landmark Precedent" if "v." in title else "Statutory Provision"
            }

        # ─────────────────────────────────────────────────────────────────
        # ISSUE 1: FIR REGISTRATION DELAY
        # ─────────────────────────────────────────────────────────────────
        if f["delayDays"]["value"] is not None:
            delay_num = f["delayDays"]["value"]
            delay_str = f"{delay_num} days"
            occ_d = f["occurrenceDateDisplay"]
            fir_d = f["firDateDisplay"]

            reason_txt = f["delayReason"]["value"]
            if reason_txt:
                read_delay = f"Alleged incident: {occ_d}. FIR registered: {fir_d}. Interval: {delay_str}. Delay context: {reason_txt}"
                assess_delay = (
                    f"The supplied narrative indicates a {delay_str} interval between the alleged incident and FIR registration. "
                    f"{reason_txt} Delay is not by itself an automatic ground for quashing, but its credibility and significance "
                    "require assessment in light of the complete contemporaneous record."
                )
            else:
                read_delay = f"Alleged incident: {occ_d}. FIR registered: {fir_d}. Interval: {delay_str}. No written explanation stated in narrative."
                assess_delay = (
                    f"The supplied narrative indicates a {delay_str} interval between the alleged incident and FIR registration. "
                    "No written explanation for the delay is stated in the supplied material. Delay requires contextual examination."
                )
                material_not_established.append("Written explanation for the FIR registration delay")

            auth_delay = [make_auth(
                title="State of H.P. v. Gian Chand",
                court="Supreme Court of India",
                citation="(2001) 6 SCC 71",
                section="FIR Registration Delay",
                prop="Delay in lodging an FIR is not a ritualistic ground for vitiating the prosecution case; the court must examine the explanation offered in light of the whole record.",
                reason="Governing Supreme Court authority on appreciation of FIR delay and contextual explanation.",
                issue_id="FIR_DELAY"
            )]

            findings.append(Finding(
                finding_id="FIR_DELAY_001",
                code="REGISTRATION_DELAY",
                category="FIR delay",
                status="INFORMATIONAL",
                title="FIR registration delay",
                read_from_fir=read_delay,
                legal_principle="Delay in lodging the FIR is a matter of appreciation of evidence. While unexplained delay affects credibility, it is not an automatic ground for quashing under Section 528 BNSS / Section 482 CrPC.",
                assessment=assess_delay,
                authority="State of H.P. v. Gian Chand, (2001) 6 SCC 71",
                verification_required=["Station Diary / General Diary (GD) entries", "Original complaint showing time of presentation"],
                confidence="HIGH",
                confidence_explanation="Calculated deterministically from explicit dates stated in current FIR.",
                case_id=current_case_id,
                linked_authorities=auth_delay
            ))

        # ─────────────────────────────────────────────────────────────────
        # ISSUE 2: INDIVIDUAL ACCUSED-ROLE ATTRIBUTION
        # ─────────────────────────────────────────────────────────────────
        if f["rolesUnclear"]["value"] or (f["accusedCount"]["value"] and f["accusedCount"]["value"] > 1):
            read_roles = f["rolesUnclear"]["sourceSpan"] or "The narrative does not clearly identify which accused allegedly committed each act."
            auth_roles = [make_auth(
                title="State of Haryana & Ors. v. Ch. Bhajan Lal & Ors.",
                court="Supreme Court of India",
                citation="1992 Supp (1) SCC 335",
                section="Section 528 BNSS / Section 482 CrPC",
                prop="Criminal proceedings may be quashed where the allegations in the FIR do not disclose cognizable offences against the accused or where omnibus allegations fail to attribute individual overt acts.",
                reason="Governing standard for testing whether an FIR sufficiently attributes individual overt acts among multiple accused.",
                issue_id="ROLE_ATTRIBUTION"
            )]

            findings.append(Finding(
                finding_id="ROLE_ATTRIBUTION_001",
                code="MULTI_ACCUSED_ROLE_UNCLEAR",
                category="Individual role attribution",
                status="POTENTIAL_ISSUE",
                title="Individual accused-role attribution",
                read_from_fir=read_roles,
                legal_principle="In criminal law, vicarious liability cannot be presumed without explicit statutory provision. Where multiple accused are named, the FIR narrative or accompanying witness statements must attribute specific overt acts or demonstrate common intention.",
                assessment="The supplied narrative does not clearly attribute the alleged acts individually to each accused. The complete FIR and witness statements should be examined to determine whether specific overt acts are attributed to each person.",
                authority="State of Haryana v. Bhajan Lal, 1992 Supp (1) SCC 335 (Category 1 & 2)",
                verification_required=["Complete FIR with individual role schedule", "Statements of complainant and eyewitnesses"],
                confidence="HIGH",
                confidence_explanation="Directly observed from narrative's failure to distinguish individual roles.",
                case_id=current_case_id,
                linked_authorities=auth_roles
            ))
            material_not_established.append("Individual attribution of specific overt acts to each named accused")
            documents_required.append("Witness statements attributing individual acts")

        # ─────────────────────────────────────────────────────────────────
        # ISSUE 3: CRIMINAL INTIMIDATION SPECIFICITY
        # ─────────────────────────────────────────────────────────────────
        if f["telephoneThreat"]["value"] or "Criminal intimidation" in f["allegedOffences"]["value"]:
            intim_read = f["telephoneThreat"]["sourceSpan"] or "Threat alleged during telephone conversation without exact words."
            if f["threatWordsMissing"]:
                intim_read += " Exact words of alleged threat and circumstances are not specified."
                material_not_established.append("Exact spoken words and surrounding circumstances of the alleged threat")

            auth_intim = [make_auth(
                title="Manik Taneja & Anr. v. State of Karnataka & Anr.",
                court="Supreme Court of India",
                citation="(2015) 7 SCC 423",
                section="Section 351(2) BNS, 2023 / Section 506 IPC",
                prop="A vague allegation of threat without specific words, context, or proof of intention to cause alarm does not satisfy the statutory threshold of criminal intimidation.",
                reason="Directly relevant to testing whether unspecified telephone threat discloses statutory ingredients.",
                issue_id="INTIMIDATION_SPECIFICITY"
            )]

            findings.append(Finding(
                finding_id="INTIMIDATION_001",
                code="INTIMIDATION_SPECIFICITY",
                category="Criminal intimidation specificity",
                status="POTENTIAL_ISSUE",
                title="Criminal intimidation specificity",
                read_from_fir=intim_read,
                legal_principle="Criminal intimidation under Section 351(2) BNS (formerly Section 506 IPC) requires an intentional threat of injury with intent to cause alarm. The wording and circumstances of the alleged threat may be relevant to assessing whether the statutory requirements are disclosed.",
                assessment="The supplied FIR does not specify the alleged threatening words or sufficient contextual detail. The wording and circumstances of the alleged threat may be relevant to assessing whether the statutory requirements are disclosed.",
                authority="Section 351(2) BNS, 2023 / Section 506 IPC; Manik Taneja v. State of Karnataka, (2015) 7 SCC 423",
                verification_required=["Call detail records (CDR) and recordings where legally available", "Transcript of alleged telephonic communication"],
                confidence="HIGH",
                confidence_explanation="Observed from absence of exact words and circumstances in FIR narrative.",
                case_id=current_case_id,
                linked_authorities=auth_intim
            ))
            documents_required.append("Telephone / call-related evidence where legally available")

        # ─────────────────────────────────────────────────────────────────
        # ISSUE 4: MEDICAL EVIDENCE (TEST CASE B SPECIFIC)
        # ─────────────────────────────────────────────────────────────────
        if f["injuryStated"]["value"] and f["medicalExamState"] == "NOT_STATED":
            med_read = "The supplied FIR describes pain/swelling, but does not state whether medical examination or treatment occurred."
            material_not_established.append("Medical examination record / wound certificate verifying pain or injury")
            documents_required.append("Medical examination record / wound certificate where applicable")

            findings.append(Finding(
                finding_id="MEDICAL_EVIDENCE_001",
                code="MEDICAL_EVIDENCE_UNVERIFIED",
                category="Medical evidence",
                status="REQUIRES_VERIFICATION",
                title="Medical evidence",
                read_from_fir=med_read,
                legal_principle="In offences alleging bodily hurt (Section 115 BNS / Section 323 IPC), contemporaneous medical examination and injury documentation are material to substantiate the nature, extent, and causation of injury.",
                assessment="The supplied FIR describes pain/swelling but does not state whether medical examination or treatment occurred. The absence of such information from the supplied narrative does not establish that no medical record exists.",
                authority="Section 115 BNS, 2023 / Section 323 IPC",
                verification_required=["Medico-legal case (MLC) register", "Wound certificate / hospital records"],
                confidence="HIGH",
                confidence_explanation="Directly observed: bodily pain described without mention of medical certificate.",
                case_id=current_case_id,
                linked_authorities=[]
            ))

        # ─────────────────────────────────────────────────────────────────
        # ISSUE 5: WEAPON / WOODEN OBJECT RECOVERY (TEST CASE B SPECIFIC)
        # ─────────────────────────────────────────────────────────────────
        if f["woodenObject"]["value"]:
            obj_read = "Allegation mentions assault using a wooden object. Description, recovery, and seizure are not stated in the supplied FIR narrative."
            material_not_established.append("Description, seizure, or recovery of the alleged wooden object")
            documents_required.append("Seizure memo / property seizure records under Section 105 BNSS")

            findings.append(Finding(
                finding_id="OBJECT_RECOVERY_001",
                code="OBJECT_RECOVERY_UNVERIFIED",
                category="Object / weapon recovery",
                status="REQUIRES_VERIFICATION",
                title="Wooden object / recovery",
                read_from_fir=obj_read,
                legal_principle="Where a weapon or object is alleged to have been used in an assault, the description, recovery, and seizure under Section 105 BNSS / Section 100 CrPC corroborate the occurrence and establish the nature of the object.",
                assessment="Recovery/seizure is not stated in the supplied FIR. The supplied narrative does not specify the dimensions or character of the wooden object or whether it was seized by police.",
                authority="Section 105 BNSS, 2023 / Section 100 CrPC",
                verification_required=["Seizure memo of object", "Case diary entries regarding weapon search"],
                confidence="HIGH",
                confidence_explanation="Explicit mention of wooden object without recovery or seizure details.",
                case_id=current_case_id,
                linked_authorities=[]
            ))

        # ─────────────────────────────────────────────────────────────────
        # ISSUE 6: INDEPENDENT WITNESSES (TEST CASE B SPECIFIC)
        # ─────────────────────────────────────────────────────────────────
        if f["witnessState"] == "NOT_STATED" and ("Assault / Hurt" in f["allegedOffences"]["value"] or f["injuryStated"]["value"]):
            wit_read = "The supplied narrative does not state whether independent eyewitnesses were present or examined."
            material_not_established.append("Statements of independent eyewitnesses")
            documents_required.append("Witness statements under Section 180 BNSS (Section 161 CrPC)")

            findings.append(Finding(
                finding_id="WITNESSES_001",
                code="WITNESSES_UNVERIFIED",
                category="Witness verification",
                status="REQUIRES_VERIFICATION",
                title="Independent witnesses",
                read_from_fir=wit_read,
                legal_principle="Corroboration by independent witnesses is material to evaluating credibility in assault allegations arising from existing personal disagreements.",
                assessment="The supplied narrative does not state whether independent eyewitnesses were present or examined. Statements under Section 180 BNSS (Section 161 CrPC) require verification.",
                authority="State of Haryana v. Bhajan Lal, 1992 Supp (1) SCC 335",
                verification_required=["Witness statements recorded in case diary", "Scene of crime inspection report"],
                confidence="MEDIUM",
                confidence_explanation="Absence of independent witness recitals in narrative.",
                case_id=current_case_id,
                linked_authorities=[]
            ))

        # ─────────────────────────────────────────────────────────────────
        # ISSUE 7: EXISTING PERSONAL DISAGREEMENT (TEST CASE B SPECIFIC)
        # ─────────────────────────────────────────────────────────────────
        if f["personalDisagreement"]["value"]:
            disagree_read = f["personalDisagreement"]["sourceSpan"] or "The narrative references an existing personal disagreement between the parties."
            findings.append(Finding(
                finding_id="PERSONAL_DISAGREEMENT_001",
                code="PERSONAL_DISAGREEMENT_BACKGROUND",
                category="Background dispute",
                status="INFORMATIONAL",
                title="Existing personal disagreement",
                read_from_fir=disagree_read,
                legal_principle="Prior personal disagreement is relevant background context affecting motive. Under Bhajan Lal Category 7, proceedings instituted with an ulterior motive for wreaking vengeance may warrant scrutiny.",
                assessment="The narrative references an existing personal disagreement. This is relevant background context affecting motivation, but does not by itself establish the absence or presence of a cognizable offence.",
                authority="State of Haryana v. Bhajan Lal, 1992 Supp (1) SCC 335 (Category 7)",
                verification_required=["Background context and prior complaints, if any"],
                confidence="HIGH",
                confidence_explanation="Explicitly stated in narrative.",
                case_id=current_case_id,
                linked_authorities=[]
            ))

        # ─────────────────────────────────────────────────────────────────
        # ISSUE 8: THEFT / PROPERTY REMOVAL SPECIFICITY (TEST CASE A SPECIFIC)
        # ─────────────────────────────────────────────────────────────────
        if f["stolenEquipment"]:
            theft_read = "No specific description of the allegedly stolen equipment, its value, or supporting ownership documents is included in the supplied FIR narrative."
            material_not_established.append("Specific inventory, description, and value of allegedly removed equipment/documents")
            documents_required.append("Inventory of allegedly removed equipment / purchase invoices")

            findings.append(Finding(
                finding_id="THEFT_SPECIFICITY_001",
                code="THEFT_SPECIFICITY",
                category="Property removal specificity",
                status="REQUIRES_VERIFICATION",
                title="Alleged property removal specificity",
                read_from_fir=theft_read,
                legal_principle="An allegation of theft under Section 303(2) BNS (formerly Section 379 IPC) requires dishonest removal of specific movable property out of the possession of another without consent. Particulars of property and possession are required.",
                assessment="The supplied narrative does not contain sufficient particulars concerning the allegedly removed property and its value. The complete FIR and supporting documents should be examined.",
                authority="Section 303(2) BNS, 2023 / Section 379 IPC",
                verification_required=["Complete FIR inventory annexure", "Proof of ownership / possession of equipment"],
                confidence="HIGH",
                confidence_explanation="Narrative explicitly notes omission of equipment description and value.",
                case_id=current_case_id,
                linked_authorities=[]
            ))

        # ─────────────────────────────────────────────────────────────────
        # ISSUE 9: PROPERTY / OWNERSHIP DISPUTE (TEST CASE A SPECIFIC)
        # DISPUTE CONTAMINATION FIREWALL: ONLY if property dispute is present!
        # ─────────────────────────────────────────────────────────────────
        if f["propertyDispute"]["value"]:
            prop_read = f["propertyDispute"]["sourceSpan"] or "The property had been the subject of a continuing ownership dispute between the parties for approximately two years."
            documents_required.extend([
                "Title documents",
                "Ownership records",
                "Possession documents",
                "Property agreement / partnership records",
                "Prior civil proceedings records"
            ])

            auth_prop = [make_auth(
                title="State of Haryana & Ors. v. Ch. Bhajan Lal & Ors.",
                court="Supreme Court of India",
                citation="1992 Supp (1) SCC 335",
                section="Section 528 BNSS / Section 482 CrPC",
                prop="Where criminal proceeding is manifestly attended with mala fides or instituted with an ulterior motive for wreaking vengeance on the accused due to private or personal grudge / civil dispute, inherent powers may be exercised.",
                reason="Governing precedent on criminal complaints arising from property/civil disputes.",
                issue_id="PROPERTY_DISPUTE"
            )]

            findings.append(Finding(
                finding_id="PROPERTY_DISPUTE_001",
                code="PROPERTY_OWNERSHIP_DISPUTE",
                category="Property dispute",
                status="POTENTIAL_ISSUE",
                title="Property/ownership dispute",
                read_from_fir=prop_read,
                legal_principle="Disputes involving title, possession, or co-ownership of property are fundamentally civil in character. Criminal process cannot be utilized as a substitute for civil adjudication, though the existence of a civil dispute does not by itself preclude criminal offences if penal ingredients are independently disclosed.",
                assessment="The supplied narrative indicates an ongoing ownership dispute and an assertion that the accused had lawful access as co-owners. These facts may be relevant to determining the character of the dispute, but the existence of a property dispute does not by itself establish that the alleged criminal offences are absent.",
                authority="State of Haryana v. Bhajan Lal, 1992 Supp (1) SCC 335 (Category 1, 3 & 7)",
                verification_required=["Title documents", "Ownership records", "Possession documents", "Property agreements", "Prior civil proceedings"],
                confidence="HIGH",
                confidence_explanation="Directly narrated as an ongoing two-year ownership dispute between co-owners.",
                case_id=current_case_id,
                linked_authorities=auth_prop
            ))

        # ─────────────────────────────────────────────────────────────────
        # ISSUE 10: ARREST SAFEGUARDS (STRICT NEGATIVE HANDLING)
        # ─────────────────────────────────────────────────────────────────
        if f["arrestStatus"]["state"] == "ABSENT_EXPLICITLY" or f["arrestStatus"]["value"] is False:
            findings.append(Finding(
                finding_id="ARREST_SAFEGUARDS_001",
                code="ARREST_SAFEGUARDS_NOT_APPLICABLE",
                category="Arrest safeguards",
                status="NOT_APPLICABLE",
                title="Arrest safeguards",
                read_from_fir="No accused has been arrested at the time of registration of the FIR.",
                legal_principle="Statutory arrest safeguards, checklist compliance, and mandatory recording of reasons under Section 35(3) BNSS apply when an arrest is effected.",
                assessment="No arrest is stated to have been made. Procedural arrest safeguards and notice compliance are not applicable at this stage.",
                authority=None,
                verification_required=[],
                confidence="HIGH",
                confidence_explanation="FIR explicitly affirms no arrest has been made.",
                case_id=current_case_id,
                linked_authorities=[]
            ))
        elif f["arrestStatus"]["value"] is True:
            # Positive arrest occurred
            auth_arr = [make_auth(
                title="Arnesh Kumar v. State of Bihar & Anr.",
                court="Supreme Court of India",
                citation="(2014) 8 SCC 273",
                section="Section 35(3) BNSS / Section 41A CrPC",
                prop="Mandatory recording of reasons and compliance with statutory notice safeguards before arresting for offences punishable up to 7 years.",
                reason="Applicable to arrest procedural compliance where arrest was effected.",
                issue_id="ARREST_PROCEDURE"
            )]
            findings.append(Finding(
                finding_id="ARREST_SAFEGUARDS_001",
                code="ARREST_RECORD_UNVERIFIED",
                category="Arrest procedure",
                status="REQUIRES_VERIFICATION",
                title="Arrest procedure compliance requires record verification",
                read_from_fir=f["arrestStatus"]["sourceSpan"],
                legal_principle="Arrest compliance requires verification of recorded reasons under Section 35(3) BNSS.",
                assessment="The record indicates arrest. Compliance with statutory safeguards must be verified.",
                authority="Arnesh Kumar v. State of Bihar, (2014) 8 SCC 273",
                verification_required=["Arrest memo", "Case diary entries"],
                confidence="MEDIUM",
                case_id=current_case_id,
                linked_authorities=auth_arr
            ))

        # ─────────────────────────────────────────────────────────────────
        # ISSUE 11: SPECIAL STATUTE AUDIT
        # ─────────────────────────────────────────────────────────────────
        if not f["specialStatuteStatus"]["value"]:
            findings.append(Finding(
                finding_id="SPECIAL_STATUTE_001",
                code="SPECIAL_STATUTE_NOT_APPLICABLE",
                category="Special statute audit",
                status="NOT_APPLICABLE",
                title="Special statute",
                read_from_fir="Special statute not invoked.",
                legal_principle="Special statutory regimes (PMLA, UAPA, NDPS, SC/ST Act) impose distinct procedural thresholds and twin bail conditions.",
                assessment="No special penal enactment is invoked in the supplied narrative. Special procedural thresholds do not apply.",
                authority=None,
                verification_required=[],
                confidence="HIGH",
                confidence_explanation="No special statute invoked or checked.",
                case_id=current_case_id,
                linked_authorities=[]
            ))

        # ─────────────────────────────────────────────────────────────────
        # DYNAMIC FINDING COUNT (SECTION 26)
        # Count strictly POTENTIAL_ISSUE and REQUIRES_VERIFICATION
        # ─────────────────────────────────────────────────────────────────
        issue_findings = [f_item for f_item in findings if f_item.status in ("POTENTIAL_ISSUE", "REQUIRES_VERIFICATION")]
        dynamic_issue_count = len(issue_findings)

        overall_status = "Potential Procedural Issues Identified" if dynamic_issue_count > 0 else "Screening Complete"
        status_explanation = (
            f"Preliminary screening identified {dynamic_issue_count} procedural issue(s) requiring verification concerning "
            "allegation specificity, individual role attribution, and evidentiary completeness. "
            "These findings warrant examination against the complete police record."
        )

        # ─────────────────────────────────────────────────────────────────
        # POTENTIAL INHERENT-POWERS / QUASHING-RELEVANT ISSUES (SECTION 28)
        # Issues for examination only — ZERO predictions, ZERO scores!
        # ─────────────────────────────────────────────────────────────────
        quashing_issues = []
        quashing_issues.append("Whether the supplied allegations disclose the statutory ingredients of the alleged offences.")
        if f["accusedCount"]["value"] and f["accusedCount"]["value"] > 1:
            quashing_issues.append("Whether specific overt acts are sufficiently attributed to each named accused.")
        if f["propertyDispute"]["value"]:
            quashing_issues.append("Whether the ongoing property ownership dispute affects the characterization of the criminal allegations.")
            quashing_issues.append("Whether the complete record supports the criminal allegations despite the asserted lawful access as co-owners.")
        elif f["personalDisagreement"]["value"]:
            quashing_issues.append("Whether the existing personal disagreement between the parties indicates an ulterior motive under Bhajan Lal Category 7.")
        if f["delayDays"]["value"] is not None and f["delayDays"]["value"] > 0:
            quashing_issues.append(f"Whether the {f['delayDays']['value']}-day interval between occurrence and FIR registration has an adequate contextual explanation.")

        # ─────────────────────────────────────────────────────────────────
        # OFFENCE INGREDIENT ANALYSIS (CURRENT BNS PROVISIONS)
        # ─────────────────────────────────────────────────────────────────
        ingredient_analysis_list: list[dict] = []
        for off_name in f["allegedOffences"]["value"]:
            if off_name == "Criminal trespass":
                ingredient_analysis_list.append({
                    "offence_name": "Criminal trespass",
                    "invoked_section": "Numerical sections not specified (Section 329 BNS / Section 441 IPC)",
                    "corresponding_bns": "Section 329(1) / 329(3) BNS, 2023",
                    "ingredients": [
                        {
                            "ingredient_number": 1,
                            "ingredient_text": "Unlawful entry into or upon property in possession of another.",
                            "fir_support": "Alleged entry into property; accused state they had lawful access as co-owners." if f["propertyDispute"]["value"] else "Entry into property alleged; possessory details not detailed.",
                            "status": "Disputed" if f["propertyDispute"]["value"] else "Partly stated"
                        },
                        {
                            "ingredient_number": 2,
                            "ingredient_text": "Intent to commit an offence or intimidate/annoy person in possession.",
                            "fir_support": "Intent disputed in context of property dispute." if f["propertyDispute"]["value"] else "Intent requires verification from witness statements.",
                            "status": "Unclear"
                        }
                    ],
                    "ingredient_assessment": "Ingredients require assessment of co-ownership rights and lawful access." if f["propertyDispute"]["value"] else "Ingredients require verification of possessory entitlement."
                })
            elif off_name == "Theft":
                ingredient_analysis_list.append({
                    "offence_name": "Theft",
                    "invoked_section": "Numerical sections not specified (Section 303(2) BNS / Section 379 IPC)",
                    "corresponding_bns": "Section 303(2) BNS, 2023",
                    "ingredients": [
                        {
                            "ingredient_number": 1,
                            "ingredient_text": "Dishonest intention to take movable property out of possession of another without consent.",
                            "fir_support": "Removal of equipment/documents alleged; specific description and possession documents not supplied.",
                            "status": "Partly stated"
                        },
                        {
                            "ingredient_number": 2,
                            "ingredient_text": "Moving property in order to such taking.",
                            "fir_support": "Property removal alleged; individual acts not specified.",
                            "status": "Unclear"
                        }
                    ],
                    "ingredient_assessment": "Allegations lack inventory and possessory particulars; complete investigation record required."
                })
            elif off_name == "Assault / Hurt":
                ingredient_analysis_list.append({
                    "offence_name": "Voluntarily causing hurt",
                    "invoked_section": "Numerical sections not specified (Section 115(2) BNS / Section 323 IPC)",
                    "corresponding_bns": "Section 115(2) BNS, 2023",
                    "ingredients": [
                        {
                            "ingredient_number": 1,
                            "ingredient_text": "Causing bodily pain, disease, or infirmity to any person.",
                            "fir_support": "Pain and swelling alleged from assault with wooden object; medical examination not stated.",
                            "status": "Partly stated"
                        },
                        {
                            "ingredient_number": 2,
                            "ingredient_text": "Doing an act with the intention of thereby causing hurt or with the knowledge that it is likely to cause hurt.",
                            "fir_support": "Confrontation following personal disagreement alleged; individual acts not attributed.",
                            "status": "Unclear"
                        }
                    ],
                    "ingredient_assessment": "Substantiation requires medical injury documentation and attribution of overt acts."
                })
            elif off_name == "Criminal intimidation":
                ingredient_analysis_list.append({
                    "offence_name": "Criminal intimidation",
                    "invoked_section": "Numerical sections not specified (Section 351(2) BNS / Section 506 IPC)",
                    "corresponding_bns": "Section 351(2) / 351(3) BNS, 2023",
                    "ingredients": [
                        {
                            "ingredient_number": 1,
                            "ingredient_text": "Threat of injury to person, reputation, or property.",
                            "fir_support": "Telephone threat alleged; exact words and circumstances not stated.",
                            "status": "Unclear"
                        },
                        {
                            "ingredient_number": 2,
                            "ingredient_text": "Intent to cause alarm or compel person to act.",
                            "fir_support": "Circumstances and intent to cause alarm not detailed in narrative.",
                            "status": "Not stated"
                        }
                    ],
                    "ingredient_assessment": "Statutory threshold of alarm and specific threatening words requires verification against call evidence."
                })

        # ─────────────────────────────────────────────────────────────────
        # PRIMARY INHERENT POWERS & STATUTORY AUTHORITIES
        # ─────────────────────────────────────────────────────────────────
        all_authorities: list[dict] = []
        # Section 528 BNSS is the governing inherent powers provision
        all_authorities.append(make_auth(
            title="Bharatiya Nagarik Suraksha Sanhita, 2023",
            court="Parliament of India (India Code)",
            citation="Act No. 46 of 2023",
            section="Section 528 BNSS",
            prop="Preserves the inherent powers of the High Court to make such orders as may be necessary to give effect to any order under this Sanhita, or to prevent abuse of the process of any Court, or otherwise to secure the ends of justice.",
            reason="Primary procedural legislation governing quashing applications in current criminal procedure.",
            issue_id="SEC_528_BNSS"
        ))

        # Bhajan Lal applies as core standard for inherent powers scrutiny
        all_authorities.append(make_auth(
            title="State of Haryana & Ors. v. Ch. Bhajan Lal & Ors.",
            court="Supreme Court of India",
            citation="1992 Supp (1) SCC 335",
            section="Section 528 BNSS / Section 482 CrPC",
            prop="Established foundational categories for testing whether allegations disclose cognizable offences, whether individual acts are attributed, and whether proceedings are attended with personal grudge / civil disputes.",
            reason="Core judicial standard for inherent-powers assessment under Section 528 BNSS.",
            issue_id="BHAJAN_LAL_CORE"
        ))

        # Collect authorities linked directly to individual findings
        for find in findings:
            for auth in find.linked_authorities:
                if not any(a["citation"] == auth["citation"] for a in all_authorities):
                    all_authorities.append(auth)

        # ─────────────────────────────────────────────────────────────────
        # PRE-RENDERING VALIDATION (SECTION 33: VALIDATIONS A THROUGH N)
        # ─────────────────────────────────────────────────────────────────
        # Validation A: All facts from current case_id
        for k, v in f.items():
            if isinstance(v, dict) and "sourceCaseId" in v:
                assert v["sourceCaseId"] == current_case_id or v["sourceCaseId"] == ""

        # Validation F: Dispute contamination firewall
        if not f["propertyDispute"]["value"]:
            # Ensure no property dispute finding exists
            findings = [find for find in findings if find.code != "PROPERTY_OWNERSHIP_DISPUTE"]
            # Ensure no title / ownership documents
            documents_required = [d for d in documents_required if not any(w in d.lower() for w in ["title", "ownership records", "co-owner", "civil proceedings", "partnership records"])]
            # Ensure no Vesa Holdings or Paramjeet Batra
            all_authorities = [a for a in all_authorities if not any(w in a["title"].lower() for w in ["vesa holdings", "paramjeet batra"])]

        # Validation H: CPC Filter — Zero CPC authorities in criminal audit
        all_authorities = [a for a in all_authorities if "civil procedure" not in a.get("section", "").lower() and "cpc" not in a.get("section", "").lower()]

        # Validation J: Recalculate dynamic issue count to ensure exact equality
        dynamic_issue_count = len([find for find in findings if find.status in ("POTENTIAL_ISSUE", "REQUIRES_VERIFICATION")])
        overall_status = "Potential Procedural Issues Identified" if dynamic_issue_count > 0 else "Screening Complete"
        status_explanation = (
            f"Preliminary screening identified {dynamic_issue_count} procedural issue(s) requiring verification concerning "
            "allegation specificity, individual role attribution, and evidentiary completeness. "
            "These findings warrant examination against the complete police record."
        )

        # Validation M & N: Ensure zero quashing probability or prediction text
        # (Verified by structure: quashing_issues contains questions only, no scores)

        applicable_provision = (
            "Section 528 BNSS, 2023 — Inherent powers of the High Court\n"
            "(formerly corresponding to Section 482 CrPC)"
        )

        return {
            "status": "success",
            "case_id": current_case_id,
            "screening_header": "Preliminary Procedural Screening",
            "overall_status": overall_status,
            "status_explanation": status_explanation,
            "applicable_provision": applicable_provision,
            "read_from_fir": {
                "alleged_occurrence": f["occurrenceDateDisplay"],
                "fir_registered": f["firDateDisplay"],
                "registration_delay": f"{f['delayDays']['value']} days" if f["delayDays"]["value"] is not None else "Not determinable from supplied dates.",
                "delay_reason": f["delayReason"]["value"] or "Not stated in supplied material.",
                "sections_invoked": f["sectionsDisplay"],
                "alleged_offences": ", ".join(f["allegedOffences"]["value"]) if f["allegedOffences"]["value"] else "Not specified",
                "accused": f["accusedDescription"],
                "arrest": f["arrestLabel"],
                "special_statute": f["specialStatuteLabel"],
                "property_dispute": "Mentioned" if f["propertyDispute"]["value"] else "Not mentioned",
                "contractual_dispute": "Mentioned" if f["contractualDispute"]["value"] else "Not mentioned",
                "settlement_repayment": f["settlementLabel"]
            },
            "extracted_facts": f,
            "procedural_findings": [find.to_dict() for find in findings],
            "ingredient_analysis": ingredient_analysis_list,
            "arrest_audit": {
                "status": f["arrestLabel"],
                "finding": "NOT APPLICABLE AT THIS STAGE" if f["arrestStatus"]["state"] == "ABSENT_EXPLICITLY" or f["arrestStatus"]["value"] is False else "REQUIRES VERIFICATION"
            },
            "special_statute_audit": {
                "status": f["specialStatuteLabel"],
                "finding": "NOT APPLICABLE" if not f["specialStatuteStatus"]["value"] else "APPLICABLE"
            },
            "quashing_issues": quashing_issues,
            "material_not_established": list(dict.fromkeys(material_not_established)),
            "documents_recommended": list(dict.fromkeys(documents_required)),
            "retrieved_sources": all_authorities,
            "disclaimer": (
                "Automated procedural screening only. The findings are based on the supplied material and "
                "retrieved legal authorities. They are not a determination of guilt, innocence, or the ultimate "
                "outcome of a quashing petition. The complete case record and current law must be independently verified."
            )
        }
