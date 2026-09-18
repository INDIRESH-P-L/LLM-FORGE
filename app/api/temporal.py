"""
app/api/temporal.py
===================
LegalMind AI — Criminal Law Temporal Bridge API
Converts offense sections between IPC/CrPC/IEA and BNS/BNSS/BSA
based on the date of the alleged offense (critical for post-July 1, 2024 cases).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Add scripts dir so temporal_law module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

router = APIRouter(prefix="/api/temporal", tags=["Temporal Law Bridge"])

CUTOFF_DATE = date(2024, 7, 1)   # BNS/BNSS/BSA in force

# Section mapping database (IPC → BNS, CrPC → BNSS, IEA → BSA)
IPC_TO_BNS: Dict[str, Dict] = {
    "302": {"bns": "103", "title": "Murder", "change": "Punishment same; new mandatory community service schedule added"},
    "307": {"bns": "109", "title": "Attempt to murder", "change": "Punishment unchanged"},
    "376": {"bns": "64", "title": "Rape", "change": "Enhanced minimum sentence; new aggravated categories added"},
    "376a": {"bns": "66", "title": "Rape causing death or persistent vegetative state", "change": "Punishment same"},
    "376ab": {"bns": "65", "title": "Rape of child below 12 years", "change": "New aggravated category"},
    "420": {"bns": "318(4)", "title": "Cheating and dishonestly inducing delivery of property", "change": "Punishment same; consolidated under cheating provisions"},
    "406": {"bns": "316(2)", "title": "Criminal breach of trust", "change": "Punishment same"},
    "120b": {"bns": "61(2)", "title": "Criminal conspiracy", "change": "Cognisable offence where punishment > 6 months"},
    "124a": {"bns": "152", "title": "Acts endangering sovereignty — sedition replaced", "change": "Sedition omitted; replaced by broader 'endangering sovereignty' offence"},
    "304b": {"bns": "80", "title": "Dowry death", "change": "Punishment same; mandatory 7-year minimum"},
    "498a": {"bns": "85", "title": "Cruelty by husband or relatives", "change": "Cognisable; same punishment"},
    "379": {"bns": "303(2)", "title": "Theft", "change": "Punishment unchanged"},
    "380": {"bns": "305", "title": "Theft in dwelling house", "change": "Punishment same"},
    "392": {"bns": "309(1)", "title": "Robbery", "change": "Punishment same"},
    "395": {"bns": "310(1)", "title": "Dacoity", "change": "Punishment same"},
    "323": {"bns": "115(2)", "title": "Voluntarily causing hurt", "change": "Punishment same; community service option added"},
    "325": {"bns": "117(2)", "title": "Voluntarily causing grievous hurt", "change": "Punishment same"},
    "34":  {"bns": "3(5)", "title": "Common intention", "change": "Substantively same"},
    "149": {"bns": "190", "title": "Common object (unlawful assembly)", "change": "Substantively same"},
    "465": {"bns": "336(2)", "title": "Forgery", "change": "Punishment same"},
    "468": {"bns": "336(3)", "title": "Forgery for purpose of cheating", "change": "Punishment same"},
    "471": {"bns": "340(2)", "title": "Using forged document", "change": "Punishment same"},
    "497": {"bns": "OMITTED", "title": "Adultery — STRUCK DOWN & OMITTED", "change": "S.497 IPC struck down in Joseph Shine (2018) and omitted from BNS"},
    "506": {"bns": "351(2)", "title": "Criminal intimidation", "change": "Punishment same"},
    "509": {"bns": "79", "title": "Word / gesture to insult modesty of woman", "change": "Punishment enhanced"},
    "153a": {"bns": "196", "title": "Promoting enmity between groups", "change": "Punishment same"},
    "295a": {"bns": "302", "title": "Deliberate acts to outrage religious feelings", "change": "Punishment same"},
}

CRPC_TO_BNSS: Dict[str, Dict] = {
    "154": {"bnss": "173", "title": "FIR (Information in cognizable cases)", "change": "Mandatory e-FIR within 3 days; copy to informant immediately"},
    "161": {"bnss": "180", "title": "Examination of witnesses by police", "change": "Audio-video recording now mandatory"},
    "164": {"bnss": "183", "title": "Recording of confessions and statements", "change": "Electronic recording mandatory"},
    "167": {"bnss": "187", "title": "Remand / 24-hour rule", "change": "Includes electronic remand; 60/90 day default bail preserved"},
    "173": {"bnss": "193", "title": "Chargesheet / Police report", "change": "Preliminary inquiry mandatory for offences 3–7 years (S.173 BNSS, 14-day window)"},
    "190": {"bnss": "210", "title": "Cognizance of offences by Magistrates", "change": "Substantively same"},
    "200": {"bnss": "223", "title": "Examination of complainant", "change": "Same"},
    "313": {"bnss": "351", "title": "Power to examine the accused", "change": "Same"},
    "41a": {"bnss": "35(3)", "title": "Notice before arrest (Arnesh Kumar compliance)", "change": "Now mandatory for all offences punishable ≤ 7 years; violation = illegal arrest"},
    "437": {"bnss": "480", "title": "Bail in non-bailable offences (Sessions / Magistrate)", "change": "Same twin conditions; PMLA/NDPS special provisions remain"},
    "438": {"bnss": "482", "title": "Anticipatory bail", "change": "No time limit on anticipatory bail protection (overrules earlier view)"},
    "439": {"bnss": "483", "title": "Special powers of HC / Sessions Court re bail", "change": "Same"},
    "482": {"bnss": "528", "title": "Inherent powers of High Court", "change": "Same; FIR quashing jurisdiction preserved"},
    "144": {"bnss": "163", "title": "Section 144 prohibitory orders", "change": "Same; extended to online/cyber spaces"},
    "125": {"bnss": "144", "title": "Maintenance of wife, children, parents", "change": "Interim maintenance within 60 days mandatory"},
    "320": {"bnss": "359", "title": "Compoundable offences", "change": "Same list; some additions"},
    "362": {"bnss": "424", "title": "Alteration of judgment", "change": "Same"},
    "374": {"bnss": "415", "title": "Appeals from conviction", "change": "Same"},
    "397": {"bnss": "438", "title": "Revision", "change": "Same"},
    "482": {"bnss": "528", "title": "Inherent powers of High Court", "change": "Same"},
}

IEA_TO_BSA: Dict[str, Dict] = {
    "24": {"bsa": "22", "title": "Confession caused by inducement", "change": "Same"},
    "25": {"bsa": "23(1)", "title": "Confession to police — inadmissible", "change": "Same"},
    "26": {"bsa": "23(2)", "title": "Confession by accused in custody", "change": "Same"},
    "27": {"bsa": "23(2) proviso", "title": "Discovery statement", "change": "Same; now includes digital discovery"},
    "32": {"bsa": "26", "title": "Dying declaration", "change": "Explicit provision for electronic dying declarations"},
    "45": {"bsa": "39", "title": "Expert opinion", "change": "Expert evidence expanded to include digital forensics experts"},
    "65b": {"bsa": "63", "title": "Electronic evidence admissibility", "change": "Certificate requirement simplified; no specific officer needed"},
    "101": {"bsa": "104", "title": "Burden of proof", "change": "Same"},
    "113b": {"bsa": "118", "title": "Presumption as to dowry death", "change": "Same; extended to BSA"},
    "114a": {"bsa": "119", "title": "Presumption as to absence of consent in rape cases", "change": "Substantially same; broadened"},
    "134": {"bsa": "139", "title": "Number of witnesses", "change": "Same — no minimum number required"},
}

NEW_PROVISIONS: List[Dict] = [
    {"section": "S. 105 BNSS", "title": "Mandatory audio-video recording of search and seizure", "note": "Mandatory in all premises searches; failure = suppression risk"},
    {"section": "S. 173 BNSS proviso", "title": "14-day preliminary inquiry before registering FIR", "note": "For offences punishable 3–7 years; magistrate must direct inquiry"},
    {"section": "S. 479 BNSS", "title": "Half-period undertrial bail (first-time offenders)", "note": "New: undertrials who have served ½ maximum sentence entitled to bail if first offence"},
    {"section": "S. 23 BNS", "title": "Community service as punishment", "note": "New category of punishment for minor offences (no jail option)"},
    {"section": "S. 63 BSA", "title": "Simplified electronic record certification", "note": "Removes rigid Section 65B IEA certificate requirement"},
    {"section": "S. 35(3) BNSS", "title": "Mandatory notice before arrest (all offences ≤ 7 yrs)", "note": "Arnesh Kumar compliance now statutory; no discretion"},
]


class TemporalConvertRequest(BaseModel):
    offense_date: str = Field(..., description="Date of alleged offense (YYYY-MM-DD)")
    sections: List[str] = Field(default=[], description="List of section numbers (e.g. ['302', '34'])")
    act: str = Field(default="ipc", description="Source act: ipc | crpc | iea | auto")
    incident_description: Optional[str] = Field(default=None, description="Brief description for regime advice")


class SectionMapping(BaseModel):
    original_section: str
    original_act: str
    new_section: str
    new_act: str
    title: str
    change_summary: str
    status: str  # "mapped" | "omitted" | "no_change_needed"


class TemporalConvertResponse(BaseModel):
    offense_date: str
    regime: str              # "OLD (IPC/CrPC/IEA)" | "NEW (BNS/BNSS/BSA)"
    regime_rationale: str
    applicable_cutoff: str
    section_mappings: List[SectionMapping]
    new_provisions_applicable: List[Dict]
    temporal_warning: Optional[str]
    art20_note: str


def _resolve_act(act: str, section: str) -> str:
    """Auto-detect act from section prefix hints."""
    act = act.lower()
    if act == "auto":
        s = section.lower()
        if s.startswith("s.") or any(k in s for k in ["ipc", "bns", "bnss", "crpc", "iea", "bsa"]):
            return "ipc"  # default
        return "ipc"
    return act


def _parse_date(date_str: str) -> date:
    from datetime import datetime
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date format: {date_str!r}. Use YYYY-MM-DD.")


@router.post("/convert", response_model=TemporalConvertResponse)
def convert_sections(req: TemporalConvertRequest):
    """
    Converts IPC/CrPC/IEA sections to BNS/BNSS/BSA equivalents (or confirms
    old law applies) based on the date of the alleged offense.

    Constitutional basis: Article 20(1) — no retrospective criminal law.
    Offenses BEFORE July 1, 2024 → IPC/CrPC/IEA regime.
    Offenses ON OR AFTER July 1, 2024 → BNS/BNSS/BSA regime.
    """
    try:
        odate = _parse_date(req.offense_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    is_new_regime = odate >= CUTOFF_DATE
    regime = "NEW (BNS/BNSS/BSA)" if is_new_regime else "OLD (IPC/CrPC/IEA)"
    regime_rationale = (
        "Offense date is ON OR AFTER 01-July-2024. Bharatiya Nyaya Sanhita (BNS), "
        "Bharatiya Nagarik Suraksha Sanhita (BNSS) and Bharatiya Sakshya Adhiniyam (BSA) apply."
        if is_new_regime else
        "Offense date is BEFORE 01-July-2024. Indian Penal Code (IPC), Code of Criminal Procedure (CrPC) "
        "and Indian Evidence Act (IEA) continue to apply under Article 20(1) of the Constitution."
    )

    act = req.act.lower()
    if act == "auto":
        act = "ipc"

    mappings = []
    for raw_sec in req.sections:
        s = raw_sec.strip().lower().replace("s.", "").replace("section", "").strip()
        if act in ("ipc", "bns"):
            db = IPC_TO_BNS
            old_act = "IPC, 1860"
            new_act = "BNS, 2023"
        elif act in ("crpc", "bnss"):
            db = CRPC_TO_BNSS
            old_act = "CrPC, 1973"
            new_act = "BNSS, 2023"
        elif act in ("iea", "bsa"):
            db = IEA_TO_BSA
            old_act = "IEA, 1872"
            new_act = "BSA, 2023"
        else:
            db = IPC_TO_BNS
            old_act = "IPC, 1860"
            new_act = "BNS, 2023"

        entry = db.get(s)
        if entry:
            new_sec_val = entry.get("bns") or entry.get("bnss") or entry.get("bsa") or "See notification"
            status = "omitted" if "OMITTED" in str(new_sec_val).upper() else "mapped"
            mappings.append(SectionMapping(
                original_section=f"S. {raw_sec.upper()} {old_act}",
                original_act=old_act,
                new_section=f"S. {new_sec_val} {new_act}" if status != "omitted" else "OMITTED / STRUCK DOWN",
                new_act=new_act if status != "omitted" else "N/A",
                title=entry.get("title", ""),
                change_summary=entry.get("change", "No substantive change"),
                status=status,
            ))
        else:
            mappings.append(SectionMapping(
                original_section=f"S. {raw_sec.upper()} {old_act}",
                original_act=old_act,
                new_section="Not in mapping database — consult notification",
                new_act=new_act,
                title="Unknown section",
                change_summary="Section not in our mapping database. Verify manually from the official Gazette.",
                status="no_change_needed",
            ))

    # Add new procedural provisions if new regime
    applicable_new = NEW_PROVISIONS if is_new_regime else []

    art20_note = (
        "Article 20(1) of the Constitution prohibits conviction under a law not in force at the time of the offence. "
        "All pending cases registered BEFORE 01-July-2024 will continue to be tried under IPC/CrPC/IEA even after "
        "the new codes came into force. Only offences committed ON OR AFTER 01-July-2024 attract BNS/BNSS/BSA."
    )

    temporal_warning = None
    if is_new_regime:
        temporal_warning = (
            "⚠️ This offense falls under the NEW criminal code regime (BNS/BNSS/BSA). "
            "Ensure all pleadings, FIRs, and chargesheets cite BNS/BNSS/BSA provisions. "
            "Citing IPC/CrPC sections in court documents post-July 1, 2024 is legally incorrect "
            "and may affect maintainability."
        )

    return TemporalConvertResponse(
        offense_date=req.offense_date,
        regime=regime,
        regime_rationale=regime_rationale,
        applicable_cutoff="01-July-2024",
        section_mappings=mappings,
        new_provisions_applicable=applicable_new,
        temporal_warning=temporal_warning,
        art20_note=art20_note,
    )


@router.get("/acts")
def list_acts():
    """Returns the list of replaced acts and their replacement details."""
    try:
        import temporal_law
        return {"replaced_acts": list(temporal_law.REPLACED_ACTS.keys()),
                "current_acts": list(temporal_law.CURRENT_ACTS.keys())}
    except Exception:
        return {
            "replaced_acts": ["indian penal code", "code of criminal procedure", "indian evidence act"],
            "current_acts": ["bharatiya nyaya sanhita", "bharatiya nagarik suraksha sanhita", "bharatiya sakshya adhiniyam"],
        }


@router.get("/new-provisions")
def list_new_provisions():
    """Returns landmark new procedural provisions introduced by BNSS/BNS/BSA."""
    return {"new_provisions": NEW_PROVISIONS}
