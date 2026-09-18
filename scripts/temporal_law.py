#!/usr/bin/env python3
"""
scripts/temporal_law.py
=======================
LegalMind AI — Temporal Law & Statutory Evolution Engine

Supports:
1. Current law vs Historical / Repealed law detection.
2. Major Indian statutory transitions:
   - Indian Penal Code, 1860 (IPC) -> Bharatiya Nyaya Sanhita, 2023 (BNS) [In force: 2024-07-01]
   - Code of Criminal Procedure, 1973 (CrPC) -> Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS) [In force: 2024-07-01]
   - Indian Evidence Act, 1872 (IEA) -> Bharatiya Sakshya Adhiniyam, 2023 (BSA) [In force: 2024-07-01]
   - Companies Act, 1956 -> Companies Act, 2013
   - Consumer Protection Act, 1986 -> Consumer Protection Act, 2019
   - FERA, 1973 -> FEMA, 1999
   - MRTP Act, 1969 -> Competition Act, 2002
   - Arbitration Act, 1940 -> Arbitration and Conciliation Act, 1996
3. Key Section Mappings between historical criminal codes and modern Sanhitas.
4. Constitutional amendment timeline metadata (42nd, 44th, 86th, 101st, 103rd, 106th).
5. Comprehensive temporal metadata enrichment and hallucination warnings.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TemporalStatus:
    status: str  # "current" | "historical" | "repealed" | "amended" | "unknown"
    in_force: bool
    effective_date: Optional[str] = None
    repeal_date: Optional[str] = None
    repealed_by: Optional[str] = None
    modern_counterpart: Optional[str] = None
    modern_section: Optional[str] = None
    amendment_info: Optional[str] = None
    notes: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Indian Statutory Replacement Registry
# ---------------------------------------------------------------------------

REPLACED_ACTS: Dict[str, Dict[str, Any]] = {
    "indian penal code": {
        "full_name": "Indian Penal Code, 1860",
        "short_name": "IPC",
        "enacted_year": 1860,
        "repealed_date": "2024-07-01",
        "repealed_by": "Bharatiya Nyaya Sanhita, 2023",
        "repealed_by_short": "BNS",
        "status": "repealed",
        "in_force": False,
        "note": "Repealed and replaced by Bharatiya Nyaya Sanhita (BNS), 2023 with effect from 1 July 2024.",
        "section_map": {
            "302": {"new_sec": "103", "title": "Punishment for murder"},
            "307": {"new_sec": "109", "title": "Attempt to murder"},
            "420": {"new_sec": "318(4)", "title": "Cheating and dishonestly inducing delivery of property"},
            "376": {"new_sec": "64", "title": "Punishment for rape"},
            "124a": {"new_sec": "152", "title": "Acts endangering sovereignty, unity and integrity of India"},
            "304b": {"new_sec": "80", "title": "Dowry death"},
            "498a": {"new_sec": "85/86", "title": "Cruelty by husband or relatives"},
            "34": {"new_sec": "3(5)", "title": "Common intention"},
            "149": {"new_sec": "190", "title": "Common object"},
            "323": {"new_sec": "115(2)", "title": "Voluntarily causing hurt"},
            "325": {"new_sec": "117(2)", "title": "Voluntarily causing grievous hurt"},
            "379": {"new_sec": "303(2)", "title": "Punishment for theft"},
            "380": {"new_sec": "305", "title": "Theft in dwelling house"},
            "406": {"new_sec": "316(2)", "title": "Criminal breach of trust"},
            "465": {"new_sec": "336(2)", "title": "Punishment for forgery"},
            "497": {"new_sec": "repealed/omitted", "title": "Adultery (struck down in Joseph Shine, omitted in BNS)"},
            "506": {"new_sec": "351(2)", "title": "Criminal intimidation"},
        },
    },
    "code of criminal procedure": {
        "full_name": "Code of Criminal Procedure, 1973",
        "short_name": "CrPC",
        "enacted_year": 1973,
        "repealed_date": "2024-07-01",
        "repealed_by": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "repealed_by_short": "BNSS",
        "status": "repealed",
        "in_force": False,
        "note": "Repealed and replaced by Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023 with effect from 1 July 2024.",
        "section_map": {
            "154": {"new_sec": "173", "title": "Information in cognizable cases (FIR)"},
            "161": {"new_sec": "180", "title": "Examination of witnesses by police"},
            "164": {"new_sec": "183", "title": "Recording of confessions and statements"},
            "167": {"new_sec": "187", "title": "Procedure when investigation cannot be completed in 24 hours (Remand)"},
            "173": {"new_sec": "193", "title": "Report of police officer on completion of investigation (Chargesheet)"},
            "190": {"new_sec": "210", "title": "Cognizance of offences by Magistrates"},
            "200": {"new_sec": "223", "title": "Examination of complainant"},
            "313": {"new_sec": "351", "title": "Power to examine the accused"},
            "437": {"new_sec": "480", "title": "When bail may be taken in case of non-bailable offence"},
            "438": {"new_sec": "482", "title": "Direction for grant of bail to person apprehending arrest (Anticipatory Bail)"},
            "439": {"new_sec": "483", "title": "Special powers of High Court or Court of Session regarding bail"},
            "482": {"new_sec": "528", "title": "Saving of inherent powers of High Court"},
            "144": {"new_sec": "163", "title": "Power to issue order in urgent cases of nuisance or apprehended danger"},
            "125": {"new_sec": "144", "title": "Order for maintenance of wives, children and parents"},
        },
    },
    "indian evidence act": {
        "full_name": "Indian Evidence Act, 1872",
        "short_name": "IEA",
        "enacted_year": 1872,
        "repealed_date": "2024-07-01",
        "repealed_by": "Bharatiya Sakshya Adhiniyam, 2023",
        "repealed_by_short": "BSA",
        "status": "repealed",
        "in_force": False,
        "note": "Repealed and replaced by Bharatiya Sakshya Adhiniyam (BSA), 2023 with effect from 1 July 2024.",
        "section_map": {
            "24": {"new_sec": "22", "title": "Confession caused by inducement, threat or promise"},
            "25": {"new_sec": "23(1)", "title": "Confession to police officer not to be proved"},
            "26": {"new_sec": "23(2)", "title": "Confession by accused while in custody of police"},
            "27": {"new_sec": "23(2) proviso", "title": "How much of information received from accused may be proved (Discovery)"},
            "32": {"new_sec": "26", "title": "Cases in which statement of relevant fact by person who is dead is relevant (Dying declaration)"},
            "65b": {"new_sec": "63", "title": "Admissibility of electronic records"},
            "113b": {"new_sec": "118", "title": "Presumption as to dowry death"},
            "114": {"new_sec": "119", "title": "Court may presume existence of certain facts"},
            "134": {"new_sec": "139", "title": "Number of witnesses"},
        },
    },
    "companies act, 1956": {
        "full_name": "Companies Act, 1956",
        "short_name": "Companies Act 1956",
        "enacted_year": 1956,
        "repealed_date": "2013-09-12",
        "repealed_by": "Companies Act, 2013",
        "repealed_by_short": "Companies Act 2013",
        "status": "repealed",
        "in_force": False,
        "note": "Repealed and replaced by Companies Act, 2013.",
        "section_map": {},
    },
    "consumer protection act, 1986": {
        "full_name": "Consumer Protection Act, 1986",
        "short_name": "COPRA 1986",
        "enacted_year": 1986,
        "repealed_date": "2020-07-20",
        "repealed_by": "Consumer Protection Act, 2019",
        "repealed_by_short": "COPRA 2019",
        "status": "repealed",
        "in_force": False,
        "note": "Repealed and replaced by Consumer Protection Act, 2019 with effect from 20 July 2020.",
        "section_map": {},
    },
    "foreign exchange regulation act": {
        "full_name": "Foreign Exchange Regulation Act, 1973",
        "short_name": "FERA",
        "enacted_year": 1973,
        "repealed_date": "2000-06-01",
        "repealed_by": "Foreign Exchange Management Act, 1999",
        "repealed_by_short": "FEMA",
        "status": "repealed",
        "in_force": False,
        "note": "Repealed and replaced by Foreign Exchange Management Act (FEMA), 1999.",
        "section_map": {},
    },
    "monopolies and restrictive trade practices act": {
        "full_name": "Monopolies and Restrictive Trade Practices Act, 1969",
        "short_name": "MRTP Act",
        "enacted_year": 1969,
        "repealed_date": "2009-09-01",
        "repealed_by": "Competition Act, 2002",
        "repealed_by_short": "Competition Act",
        "status": "repealed",
        "in_force": False,
        "note": "Repealed and replaced by Competition Act, 2002.",
        "section_map": {},
    },
}

# Key In-Force Central Acts
CURRENT_ACTS: Dict[str, Dict[str, Any]] = {
    "constitution of india": {
        "full_name": "Constitution of India, 1950",
        "effective_date": "1950-01-26",
        "status": "current",
        "in_force": True,
        "jurisdiction": "India",
        "amendment_count": 106,
    },
    "bharatiya nyaya sanhita": {
        "full_name": "Bharatiya Nyaya Sanhita, 2023",
        "effective_date": "2024-07-01",
        "status": "current",
        "in_force": True,
        "replaces": "Indian Penal Code, 1860",
    },
    "bharatiya nagarik suraksha sanhita": {
        "full_name": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "effective_date": "2024-07-01",
        "status": "current",
        "in_force": True,
        "replaces": "Code of Criminal Procedure, 1973",
    },
    "bharatiya sakshya adhiniyam": {
        "full_name": "Bharatiya Sakshya Adhiniyam, 2023",
        "effective_date": "2024-07-01",
        "status": "current",
        "in_force": True,
        "replaces": "Indian Evidence Act, 1872",
    },
    "prevention of money laundering act": {
        "full_name": "Prevention of Money Laundering Act, 2002",
        "effective_date": "2005-07-01",
        "status": "current",
        "in_force": True,
        "major_amendment": "Finance Act, 2019 (Twin conditions in Section 45 restored/clarified post-Nikesh Tarachand Shah)",
    },
    "insolvency and bankruptcy code": {
        "full_name": "Insolvency and Bankruptcy Code, 2016",
        "effective_date": "2016-05-28",
        "status": "current",
        "in_force": True,
    },
}

CONSTITUTIONAL_AMENDMENT_MILESTONES: List[Dict[str, Any]] = [
    {
        "amendment": "42nd Constitutional Amendment Act, 1976",
        "year": 1976,
        "changes": "Inserted 'Secular', 'Socialist', and 'Integrity' into Preamble; added Part IV-A (Fundamental Duties, Art 51A).",
    },
    {
        "amendment": "44th Constitutional Amendment Act, 1978",
        "year": 1978,
        "changes": "Removed Right to Property from Fundamental Rights (Art 19(1)(f), Art 31); made it a constitutional right under Art 300A. Safeguards for National Emergency (Art 352).",
    },
    {
        "amendment": "86th Constitutional Amendment Act, 2002",
        "year": 2002,
        "changes": "Inserted Article 21A (Right to Free and Compulsory Education for children 6-14 years).",
    },
    {
        "amendment": "101st Constitutional Amendment Act, 2016",
        "year": 2016,
        "changes": "Introduced Goods and Services Tax (GST) framework; inserted Articles 246A, 269A, 279A.",
    },
    {
        "amendment": "103rd Constitutional Amendment Act, 2019",
        "year": 2019,
        "changes": "Introduced 10% reservation for Economically Weaker Sections (EWS); amended Articles 15(6) and 16(6). Upheld in Janhit Abhiyan (2022).",
    },
    {
        "amendment": "106th Constitutional Amendment Act, 2023",
        "year": 2023,
        "changes": "Nari Shakti Vandan Adhiniyam — One-third reservation for women in Lok Sabha and State Legislative Assemblies (Arts 330A, 332A).",
    },
]


# ---------------------------------------------------------------------------
# Core Analysis API
# ---------------------------------------------------------------------------

def detect_temporal_context(query: str) -> Dict[str, Any]:
    """
    Identifies whether the query mentions historical, repealed, or modern provisions,
    and returns statutory transition warnings.
    """
    q_lower = query.lower()
    warnings: List[str] = []
    matched_repealed: List[Dict[str, Any]] = []
    matched_current: List[Dict[str, Any]] = []
    suggested_modern_counterparts: List[Dict[str, str]] = []

    # Check for historical / repealed statutes
    for key, info in REPLACED_ACTS.items():
        aliases = [key, info["short_name"].lower()]
        if any(re.search(rf"\b{re.escape(alias)}\b", q_lower) for alias in aliases):
            matched_repealed.append(info)
            warn_msg = (
                f"Notice on Legal Status: '{info['full_name']}' was repealed and replaced by "
                f"'{info['repealed_by']}' on {info['repealed_date']}. "
                f"Offences committed on or after 1 July 2024 are governed by modern Sanhitas."
            )
            warnings.append(warn_msg)

            # Check for specific section numbers mapped in query
            for old_sec, sec_info in info.get("section_map", {}).items():
                sec_pat = rf"\b(?:sec(?:tion)?\.?|s\.)\s*{re.escape(old_sec)}\b|\b{re.escape(old_sec)}\s*{re.escape(info['short_name'].lower())}\b"
                if re.search(sec_pat, q_lower):
                    suggested_modern_counterparts.append({
                        "historical_act": info["short_name"],
                        "historical_section": old_sec,
                        "modern_act": info["repealed_by_short"],
                        "modern_section": sec_info["new_sec"],
                        "title": sec_info["title"],
                    })
                    warnings.append(
                        f"Section Transition: Section {old_sec} of {info['short_name']} corresponds to "
                        f"Section {sec_info['new_sec']} of {info['repealed_by_short']} ({sec_info['title']})."
                    )

    # Check for modern current statutes
    for key, info in CURRENT_ACTS.items():
        if key in q_lower or (info.get("full_name") and info["full_name"].lower() in q_lower):
            matched_current.append(info)

    status_str = "current"
    if matched_repealed and not matched_current:
        status_str = "repealed"
    elif matched_repealed and matched_current:
        status_str = "mixed_historical_and_current"
    elif not matched_repealed and not matched_current:
        # Check if query references general constitution or unknown
        if "constitution" in q_lower or "article" in q_lower:
            status_str = "current"
        else:
            status_str = "current"

    return {
        "temporal_status": status_str,
        "is_repealed_law_cited": len(matched_repealed) > 0,
        "repealed_acts": [a["full_name"] for a in matched_repealed],
        "current_acts": [a["full_name"] for a in matched_current],
        "suggested_transitions": suggested_modern_counterparts,
        "warnings": warnings,
        "jurisdiction": "India",
    }


def enrich_chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enriches chunk metadata with temporal law fields:
    - act_status: "in_force" | "repealed" | "current"
    - in_force: bool
    - temporal_status: "current" | "historical" | "unknown"
    - effective_date: str or None
    - modern_equivalent: str or None
    """
    act_title = (chunk.get("act_name") or chunk.get("title") or "").lower()
    doc_type = chunk.get("document_type", "statute")

    enriched = dict(chunk)
    enriched.setdefault("jurisdiction", "India")
    enriched.setdefault("state", "Central")
    enriched.setdefault("amendment_count", 0)

    # Default statuses
    enriched["act_status"] = "in_force"
    enriched["in_force"] = True
    enriched["temporal_status"] = "current"
    enriched["effective_date"] = None
    enriched["modern_equivalent"] = None

    for rep_key, rep_info in REPLACED_ACTS.items():
        if rep_key in act_title or rep_info["short_name"].lower() in act_title:
            enriched["act_status"] = "repealed"
            enriched["in_force"] = False
            enriched["temporal_status"] = "historical"
            enriched["repealed_by"] = rep_info["repealed_by"]
            enriched["repeal_date"] = rep_info["repealed_date"]
            enriched["modern_equivalent"] = rep_info["repealed_by"]

            # Map section if available
            sec = str(chunk.get("section") or "").strip()
            if sec and sec in rep_info.get("section_map", {}):
                mapped = rep_info["section_map"][sec]
                enriched["modern_section"] = mapped["new_sec"]
                enriched["modern_section_title"] = mapped["title"]
            break

    if "constitution" in act_title:
        enriched["act_status"] = "in_force"
        enriched["in_force"] = True
        enriched["temporal_status"] = "current"
        enriched["effective_date"] = "1950-01-26"
        enriched["amendment_count"] = 106

    return enriched


def get_modern_section(act_name: str, section: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    act_lower = act_name.lower().strip()
    sec_clean = str(section).lower().strip().lstrip("s.").strip()
    for key, info in REPLACED_ACTS.items():
        if key in act_lower or info["short_name"].lower() == act_lower or act_lower in key:
            sec_map = info.get("section_map", {})
            if sec_clean in sec_map:
                sec_info = sec_map[sec_clean]
                return sec_info["new_sec"], info["repealed_by"], sec_info.get("title", "")
    return None, None, None


def is_act_repealed(act_name: str) -> bool:
    act_lower = act_name.lower().strip()
    for key, info in REPLACED_ACTS.items():
        if key in act_lower or info["short_name"].lower() == act_lower or act_lower in key:
            return not info.get("in_force", True)
    return False


def check_query_temporal_status(query: str) -> Dict[str, Any]:
    ctx = detect_temporal_context(query)
    is_repealed = ctx.get("is_repealed_law_cited", False)
    repealed_acts = ctx.get("repealed_acts", [])
    warnings = []
    for t in ctx.get("suggested_transitions", []):
        warnings.append(f"Section {t['modern_section']} {t['modern_act']} replaces Section {t['historical_section']} {t['historical_act']} ({t['title']}).")
    for w in ctx.get("warnings", []):
        warnings.append(w)

    rep_by = None
    if repealed_acts:
        for info in REPLACED_ACTS.values():
            if info["full_name"] in repealed_acts:
                rep_by = info["repealed_by"]
                break

    return {
        "has_temporal_context": is_repealed,
        "status": "repealed" if is_repealed else "current",
        "repealed_by": rep_by,
        "warnings": warnings,
        "transitions": ctx.get("suggested_transitions", []),
    }


def format_temporal_warning(query: str) -> str:
    res = check_query_temporal_status(query)
    return "\n".join(res.get("warnings", []))

