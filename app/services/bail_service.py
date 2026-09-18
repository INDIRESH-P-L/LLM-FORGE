"""
app/services/bail_service.py
============================
LegalMind AI — Bail Feasibility & Judicial Sentencing Matrix

Rule-based bail feasibility score built from the inputs that actually drive a
bail court's reasoning:

  1. Offence category      → base score from gravity and maximum punishment
  2. Special Act invoked   → twin conditions (PMLA S.45 / NDPS S.37 /
                              UAPA S.43D(5)) pull the score down and cap it
  3. Custody duration      → Article 21 / prolonged under-trial incarceration,
                              S.436A CrPC / S.479 BNSS half-sentence release
  4. Chargesheet filed     → investigation complete vs. pending
  5. Default bail          → S.167(2) CrPC / S.187(3) BNSS, which overrides
                              the rest once it has accrued

Every point added or removed is recorded in `score_breakdown`, and the factor
text and precedents are chosen from the rules that actually fired.

This is decision support, not a prediction of any court's order.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ── Offence categories ──────────────────────────────────────────────────────
# Matched by keyword so free-text categories from the CLI ("Cheating / Financial
# Crime") land in the right profile. Order matters: the first match wins.
CATEGORY_PROFILES: List[Dict[str, Any]] = [
    {
        "key": "heinous",
        "keywords": ("heinous", "rape", "pocso", "dacoity", "terror", "acid attack"),
        "label": "Heinous offence",
        "max_years": None,          # punishable with death / imprisonment for life
        "base": 20,
        "positive": [],
        "risk": ["Heinous offence: gravity of the accusation, societal impact and the risk of "
                 "witness intimidation weigh most heavily against bail."],
        "precedents": ["Prasanta Kumar Sarkar v. Ashis Chatterjee (2010) 14 SCC 496",
                       "Ram Govind Upadhyay v. Sudarshan Singh (2002) 3 SCC 598"],
        "special_regime": "ndps_uapa",
    },
    {
        "key": "murder",
        "keywords": ("murder", "homicide", "302", "103 bns"),
        "label": "Murder / culpable homicide",
        "max_years": None,
        "base": 25,
        "positive": [],
        "risk": ["Offence punishable with death or imprisonment for life (S.302 IPC / S.103 BNS): "
                 "nature of the evidence and the severity of punishment are primary considerations."],
        "precedents": ["Prahlad Singh Bhati v. NCT of Delhi (2001) 4 SCC 280",
                       "Ram Govind Upadhyay v. Sudarshan Singh (2002) 3 SCC 598"],
        "special_regime": "ndps_uapa",
    },
    {
        "key": "cheating",
        "keywords": ("cheat", "breach of trust", "420", "406"),
        "label": "Cheating / criminal breach of trust",
        "max_years": 7,
        "base": 65,
        "positive": ["Cheating is punishable with up to 7 years (S.420 IPC / S.318(4) BNS): "
                     "the Arnesh Kumar safeguards on arrest and remand apply."],
        "risk": ["Quantum of alleged loss and willingness to restitute will weigh with the court; "
                 "aggravated breach of trust (S.409 IPC / S.316(5) BNS) carries up to life."],
        "precedents": ["Arnesh Kumar v. State of Bihar (2014) 8 SCC 273",
                       "Satender Kumar Antil v. CBI (2022) 10 SCC 51"],
        "special_regime": "pmla",
    },
    {
        "key": "economic",
        "keywords": ("economic", "financial", "fraud", "money", "scam", "embezzle"),
        "label": "Economic offence / financial fraud",
        "max_years": 10,
        "base": 50,
        "positive": ["Gravity of an economic offence is not by itself a ground to refuse bail; "
                     "each case turns on its own facts."],
        "risk": ["Economic offences involving deep-rooted conspiracy and loss of public funds "
                 "are treated as a class apart."],
        "precedents": ["Y.S. Jagan Mohan Reddy v. CBI (2013) 7 SCC 439",
                       "P. Chidambaram v. Directorate of Enforcement (2020) 13 SCC 791"],
        "special_regime": "pmla",
    },
    {
        "key": "cyber",
        "keywords": ("cyber", "it act", "66"),
        "label": "Cyber crime",
        "max_years": 3,
        "base": 68,
        "positive": ["Most IT Act offences (S.66, 66C, 66D) carry up to 3 years and are bailable "
                     "under S.77B of the IT Act."],
        "risk": ["Cyber terrorism (S.66F IT Act) is punishable with imprisonment for life and is "
                 "treated very differently."],
        "precedents": ["Satender Kumar Antil v. CBI (2022) 10 SCC 51"],
        "special_regime": "pmla",
    },
    {
        "key": "general",
        "keywords": ("general", "ipc", "7 years"),
        "label": "General offence (up to 7 years)",
        "max_years": 7,
        "base": 70,
        "positive": ["Offence punishable with up to 7 years: arrest and remand must satisfy "
                     "S.41/41A CrPC (S.35 BNSS) — bail is the rule, jail the exception."],
        "risk": [],
        "precedents": ["Arnesh Kumar v. State of Bihar (2014) 8 SCC 273",
                       "Satender Kumar Antil v. CBI (2022) 10 SCC 51"],
        "special_regime": "pmla",
    },
]
_FALLBACK_KEY = "general"

# ── Special Acts: twin conditions ───────────────────────────────────────────
SPECIAL_REGIMES: Dict[str, Dict[str, Any]] = {
    "pmla": {
        "label": "PMLA S.45",
        "risk": "Twin conditions under S.45 PMLA: the court must be satisfied there are reasonable "
                "grounds to believe the accused is not guilty and is not likely to offend on bail. "
                "The burden effectively shifts to the accused.",
        "objection": "Enforcement Directorate will argue the S.45 twin conditions are not met.",
        "precedents": ["Vijay Madanlal Choudhary v. Union of India, 2022 SCC OnLine SC 929"],
        "long_custody_precedents": ["Manish Sisodia v. Directorate of Enforcement, 2024 SCC OnLine SC 1920",
                                    "V. Senthil Balaji v. Deputy Director, Directorate of Enforcement, 2024 SCC OnLine SC 2626"],
        # PMLA offences carry up to 7 years (10 where NDPS proceeds are involved),
        # so the ordinary CrPC/BNSS periods govern default bail.
        "default_bail_days": None,
    },
    "ndps_uapa": {
        "label": "NDPS S.37 / UAPA S.43D(5)",
        "risk": "Twin conditions under S.37 NDPS / S.43D(5) UAPA: bail is barred unless the court "
                "finds reasonable grounds that the accused is not guilty (NDPS) or that the "
                "accusation is not prima facie true (UAPA).",
        "objection": "Prosecution will argue the statutory bar on bail applies with full force.",
        "precedents": ["Narcotics Control Bureau v. Mohit Aggarwal (2022) 18 SCC 374",
                       "NIA v. Zahoor Ahmad Shah Watali (2019) 5 SCC 1"],
        "long_custody_precedents": ["Union of India v. K.A. Najeeb (2021) 3 SCC 713"],
        # NDPS S.36A(4) and UAPA S.43D(2) allow investigation periods of up to
        # 180 days before default bail accrues.
        "default_bail_days": 180,
    },
}
SPECIAL_ACT_PENALTY = -25
# Twin conditions make bail exceptional; the ceiling relaxes only as prolonged
# incarceration engages Article 21 (K.A. Najeeb / Manish Sisodia line).
SPECIAL_ACT_CAP = [(730, None), (365, 65), (0, 45)]

# ── Custody: Article 21 weight of under-trial incarceration ────────────────
CUSTODY_TIERS: List[Tuple[int, int, str]] = [
    (730, 35, "Over two years in custody: prolonged incarceration without conclusion of trial "
              "strongly engages the Article 21 right to a speedy trial."),
    (365, 28, "Over a year in custody: delay in trial increasingly favours release under Article 21."),
    (180, 20, "Six months or more in custody weighs in favour of bail."),
    (90, 12, "Custody of 90 days or more is a relevant factor in favour of bail."),
    (31, 5, "Over a month in custody."),
]
LONG_CUSTODY_PRECEDENT_DAYS = 365

CHARGESHEET_FILED_POINTS = 12
CHARGESHEET_PENDING_POINTS = -8
HALF_SENTENCE_POINTS = 30
DEFAULT_BAIL_FLOOR = 90
SCORE_MIN, SCORE_MAX = 5, 95   # never present bail as certain either way


def _match_category(offense_category: str) -> Tuple[Dict[str, Any], bool]:
    text = (offense_category or "").lower()
    for profile in CATEGORY_PROFILES:
        if any(k in text for k in profile["keywords"]):
            return profile, True
    return next(p for p in CATEGORY_PROFILES if p["key"] == _FALLBACK_KEY), False


def _add_unique(items: List[str], new: List[str]) -> None:
    for item in new:
        if item not in items:
            items.append(item)


class BailFeasibilityService:
    @staticmethod
    def assess_bail(
        offense_category: str,
        is_special_act: bool,
        custody_days: int,
        chargesheet_filed: bool,
        antecedents: Optional[str] = None,
        co_accused_status: Optional[str] = None,
        max_punishment_years: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Rule-based bail feasibility matrix. Returns the score (5–95), verdict,
        factors, anticipated objections, precedents and a per-rule breakdown.
        """
        custody_days = max(0, int(custody_days or 0))
        profile, recognised = _match_category(offense_category)

        positive: List[str] = []
        risk: List[str] = []
        objections: List[str] = []
        precedents: List[str] = []
        breakdown: List[Dict[str, Any]] = []

        def adjust(points: int, factor: str) -> None:
            breakdown.append({"factor": factor, "points": points})

        # 1. Offence category ────────────────────────────────────────────────
        score = profile["base"]
        adjust(profile["base"], f"Base score for offence category: {profile['label']}")
        if not recognised:
            risk.append(f"Offence category '{offense_category}' not recognised; assessed as a "
                        "general offence punishable with up to 7 years.")
        positive.extend(profile["positive"])
        risk.extend(profile["risk"])
        _add_unique(precedents, profile["precedents"])

        # An explicit maximum punishment (in years) overrides the category default.
        if max_punishment_years is not None:
            max_years = max_punishment_years
            life_or_death = False
            if profile["max_years"] is not None and max_punishment_years > profile["max_years"]:
                score -= 10
                adjust(-10, f"Maximum punishment specified as {max_punishment_years} years")
                risk.append(f"Maximum punishment of {max_punishment_years} years increases gravity.")
        else:
            max_years = profile["max_years"]
            life_or_death = max_years is None   # death / imprisonment for life

        # 2. Special Act: twin conditions ────────────────────────────────────
        regime = SPECIAL_REGIMES[profile["special_regime"]] if is_special_act else None
        if regime:
            score += SPECIAL_ACT_PENALTY
            adjust(SPECIAL_ACT_PENALTY, f"Special Act invoked: twin conditions ({regime['label']})")
            risk.append(regime["risk"])
            objections.append(regime["objection"])
            _add_unique(precedents, regime["precedents"])

        # 3. Custody duration (Article 21) ───────────────────────────────────
        for threshold, points, text in CUSTODY_TIERS:
            if custody_days >= threshold:
                score += points
                adjust(points, f"{custody_days} days in custody")
                positive.append(text)
                break
        if custody_days < 15 and not chargesheet_filed:
            risk.append("Very early stage of custody: the prosecution may seek further custodial interrogation.")
        if regime and custody_days >= LONG_CUSTODY_PRECEDENT_DAYS:
            positive.append("Prolonged incarceration can override statutory restrictions on bail under "
                            "Special Acts where the trial is unlikely to conclude soon.")
            _add_unique(precedents, regime["long_custody_precedents"])

        # S.436A CrPC / S.479 BNSS: half of the maximum sentence served.
        # Not available for offences punishable with death or life imprisonment.
        half_sentence_served = (
            not life_or_death and max_years is not None and max_years > 0
            and custody_days >= max_years * 365 / 2
        )
        if half_sentence_served:
            score += HALF_SENTENCE_POINTS
            adjust(HALF_SENTENCE_POINTS, "Half of maximum sentence already undergone (S.436A CrPC / S.479 BNSS)")
            positive.append(f"{custody_days} days is at least half of the {max_years}-year maximum: "
                            "release on bond is ordinarily mandated under S.436A CrPC / S.479 BNSS.")
            _add_unique(precedents, ["Bhim Singh v. Union of India (2015) 13 SCC 605"])

        # 4. Chargesheet ─────────────────────────────────────────────────────
        if chargesheet_filed:
            score += CHARGESHEET_FILED_POINTS
            adjust(CHARGESHEET_FILED_POINTS, "Chargesheet filed")
            positive.append("Chargesheet filed: investigation is complete and custodial interrogation "
                            "is no longer required.")
            _add_unique(precedents, ["Sanjay Chandra v. CBI (2012) 1 SCC 40"])
        else:
            score += CHARGESHEET_PENDING_POINTS
            adjust(CHARGESHEET_PENDING_POINTS, "Investigation pending (no chargesheet)")
            risk.append("Investigation pending: prosecution will allege risk of tampering with evidence "
                        "or influencing witnesses.")
            objections.append("Prosecution will seek continued custody to complete the investigation.")

        # 5. Antecedents & parity (only when actually supplied) ──────────────
        ante = (antecedents or "").strip().lower()
        if ante and ante not in ("unknown", "not specified", "n/a"):
            if "clean" in ante or ante in ("none", "no"):
                score += 5
                adjust(5, "Clean antecedents")
                positive.append("Clean antecedents: no indication of flight risk or repeat offending.")
            else:
                score -= 15
                adjust(-15, "Criminal antecedents")
                risk.append(f"Criminal antecedents: {antecedents}.")
                objections.append("Prosecution will argue a likelihood of repeating the offence.")
        co = (co_accused_status or "").strip().lower()
        if "granted" in co or "on bail" in co:
            score += 15
            adjust(15, "Parity: co-accused on bail")
            positive.append("Parity: co-accused with a similar role has been granted bail.")
            _add_unique(precedents, ["Ramesh Bhavan Rathod v. Vishanbhai Hirabhai Makwana (2021) 6 SCC 230"])
        elif "rejected" in co:
            score -= 10
            adjust(-10, "Co-accused bail rejected")
            risk.append("Bail of a co-accused has been rejected on merits.")

        # Twin-condition ceiling, relaxed by prolonged custody or half-sentence release
        if regime and not half_sentence_served:
            cap = next(c for days, c in SPECIAL_ACT_CAP if custody_days >= days)
            if cap is not None and score > cap:
                adjust(cap - score, f"Twin-condition ceiling ({cap}) at {custody_days} days' custody")
                score = cap

        # 6. Default bail overrides the merits once accrued ──────────────────
        limit = None
        if not chargesheet_filed:
            limit = regime["default_bail_days"] if regime and regime["default_bail_days"] else (
                90 if life_or_death or (max_years or 0) >= 10 else 60)
            if custody_days > limit:
                if score < DEFAULT_BAIL_FLOOR:
                    adjust(DEFAULT_BAIL_FLOOR - score, f"Default bail accrued ({custody_days} > {limit} days)")
                    score = DEFAULT_BAIL_FLOOR
                positive.insert(0, f"Indefeasible right to default bail: {custody_days} days in custody without "
                                   f"a chargesheet exceeds the {limit}-day limit (S.167(2) CrPC / S.187(3) BNSS). "
                                   "It must be claimed before the chargesheet is filed.")
                objections.append("Prosecution may try to file the chargesheet or seek an extension before the "
                                  "default bail application is heard.")
                _add_unique(precedents, ["Bikramjit Singh v. State of Punjab (2020) 10 SCC 616",
                                         "M. Ravindran v. Intelligence Officer, DRI (2021) 2 SCC 485"])

        clamped = max(SCORE_MIN, min(SCORE_MAX, score))
        if clamped != score:
            adjust(clamped - score, f"Score kept within {SCORE_MIN}–{SCORE_MAX}")
        score = clamped

        if score >= 75:
            verdict = "High Probability"
        elif score >= 45:
            verdict = "Moderate Probability"
        else:
            verdict = "Low Probability"

        return {
            "bail_score": score,
            "verdict": verdict,
            "positive_factors": positive,
            "risk_factors": risk,
            "anticipated_objections": objections,
            "relevant_precedents": precedents,
            # Additive fields: existing clients ignore them
            "offence_profile": profile["label"],
            "special_act_regime": regime["label"] if regime else None,
            "default_bail_limit_days": limit,
            "score_breakdown": breakdown,
        }
