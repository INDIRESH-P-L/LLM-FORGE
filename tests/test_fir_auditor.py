#!/usr/bin/env python3
"""
tests/test_fir_auditor.py
=========================
Tests for the Neutral FIR Procedural Defect & Quashing Auditor.

Verifies:
1. Removal of all numerical feasibility scores and win percentages.
2. Preliminary Procedural Screening classifications and status explanations.
3. Accurate date parsing and delay calculations.
4. Arrest / Arnesh Kumar analysis when reasons for arrest are explicitly unrecorded.
5. Distinction between missing statements vs explicit negative statements.
6. Statutory version verification (IPC cited for post-July 1, 2024 incidents).
7. Offence ingredient analysis and civil/commercial dispute indicators.
8. Special statute exclusion when unchecked.
9. Source deduplication and document verification checklist.
10. End-to-end verification of the user's sample FIR prompt.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.fir_audit_service import FIRAuditorService as S

USER_SAMPLE_FIR = (
    "On 15 March 2026, the complainant approached the police alleging that the accused had "
    "threatened him and committed cheating in relation to a business transaction. The FIR was "
    "registered on 20 March 2026, approximately five days after the alleged incident, without "
    "recording any reason for the delay. The FIR states only that the accused committed cheating "
    "and criminal intimidation but does not describe the specific acts, date, place, amount "
    "involved, or words allegedly constituting the threat. No supporting documents or details of "
    "witnesses were recorded at the time of registration. The accused was arrested immediately "
    "after registration of the FIR despite the allegations being based primarily on documentary "
    "evidence, and no reasons for the necessity of arrest were recorded. The investigating officer "
    "also did not conduct any preliminary verification before arrest. The FIR invokes Sections 420 "
    "and 506 of the IPC, although the alleged transaction occurred in 2026 and the investigating "
    "officer has not stated which corresponding provisions of the applicable new criminal law are "
    "being applied. The accused has no previous criminal record according to the available "
    "information. The complainant subsequently stated that the dispute arose from a contractual "
    "disagreement and that the monetary amount had already been repaid."
)

MURDER_FIR = (
    "FIR No. 214/2024 u/s 103(1) BNS. On 12.03.2024 at 23:00 hrs the accused Ramesh assaulted "
    "the deceased Suresh with an iron rod causing fatal head injuries. Accused arrested on "
    "13.03.2024. Weapon recovered from his house during search. Registered 13.03.2024."
)

TEMPLATE = (
    "FIRST INFORMATION REPORT (Under Section 173 BNSS 2023 / Section 154 CrPC) District: "
    "[Insert District] Police Station: [Insert Police Station Name] Date and Time of Report: "
    "[Insert Date & Time]"
)
GIBBERISH = "asdkjhasd kjahsd kjahsdkjh 12312 !!!!"


class TestNoNumericalScore(unittest.TestCase):
    """Ensure no numerical quashing probability or score is produced."""

    def test_no_score_in_output(self):
        r = S.audit_fir(USER_SAMPLE_FIR, arrest_made=True, is_pmla=False)
        self.assertNotIn("quashing_feasibility_score", r)
        self.assertNotIn("score_breakdown", r)
        self.assertEqual(r["status"], "success")
        self.assertEqual(r["screening_header"], "Preliminary Procedural Screening")
        self.assertIn("Potential Procedural Issues", r["overall_status"])

    def test_status_explanation_provided(self):
        r = S.audit_fir(USER_SAMPLE_FIR, arrest_made=True, is_pmla=False)
        self.assertTrue(r["status_explanation"])
        self.assertIn("Section 528 BNSS", r["applicable_provision"])


class TestUserSampleVerification(unittest.TestCase):
    """End-to-end verification of the exact sample prompt from requirements 19 & 20."""

    def setUp(self):
        self.res = S.audit_fir(USER_SAMPLE_FIR, arrest_made=True, is_pmla=False)

    def test_date_analysis(self):
        rf = self.res["read_from_fir"]
        self.assertEqual(rf["alleged_occurrence"], "2026-03-15")
        self.assertEqual(rf["fir_registered"], "2026-03-20")
        self.assertEqual(rf["registration_delay"], "5 days")
        self.assertIn("Not stated", rf["delay_reason"])

    def test_arnesh_kumar_unrecorded_reasons_flagged(self):
        codes = [f["code"] for f in self.res["procedural_findings"]]
        self.assertIn("ARREST_REASONS_NOT_RECORDED", codes)
        f = next(x for x in self.res["procedural_findings"] if x["code"] == "ARREST_REASONS_NOT_RECORDED")
        self.assertIn("AMBER", f["status"])
        self.assertIn("Arnesh Kumar", f["authority"])
        self.assertIn("no reasons for the necessity of arrest were recorded", f["read_from_fir"])

    def test_statutory_version_check(self):
        codes = [f["code"] for f in self.res["procedural_findings"]]
        self.assertIn("STATUTORY_VERSION_MISMATCH", codes)
        f = next(x for x in self.res["procedural_findings"] if x["code"] == "STATUTORY_VERSION_MISMATCH")
        self.assertIn("AMBER", f["status"])
        self.assertIn("2026-03-15", f["read_from_fir"])
        self.assertIn("318(4)", f["assessment"])
        self.assertIn("351(2)", f["assessment"])

    def test_specificity_flagged(self):
        codes = [f["code"] for f in self.res["procedural_findings"]]
        self.assertIn("ALLEGATION_VAGUENESS", codes)
        f = next(x for x in self.res["procedural_findings"] if x["code"] == "ALLEGATION_VAGUENESS")
        self.assertIn("AMBER", f["status"])
        self.assertIn("Bhajan Lal", f["authority"])

    def test_civil_indicator_flagged(self):
        codes = [f["code"] for f in self.res["procedural_findings"]]
        self.assertIn("CIVIL_COMMERCIAL_INDICATOR", codes)
        f = next(x for x in self.res["procedural_findings"] if x["code"] == "CIVIL_COMMERCIAL_INDICATOR")
        self.assertIn("AMBER", f["status"])
        self.assertIn("contractual disagreement", f["read_from_fir"])
        self.assertIn("repaid", f["read_from_fir"])
        self.assertIn("Vesa Holdings", f["authority"])

    def test_sources_issue_driven_and_deduplicated(self):
        sources = self.res["retrieved_sources"]
        titles = [s["title"] for s in sources]
        self.assertGreater(len(sources), 0)
        # Authoritative cases must be present
        self.assertTrue(any("Arnesh Kumar" in t for t in titles))
        self.assertTrue(any("Bhajan Lal" in t for t in titles))
        self.assertTrue(any("Vesa Holdings" in t for t in titles))
        # Unrelated special statutes must NOT be present
        self.assertFalse(any("Domestic Violence" in t for t in titles))
        self.assertFalse(any("Companies Act" in t for t in titles))
        # No duplicates
        keys = [(s["title"], s["citation"]) for s in sources]
        self.assertEqual(len(keys), len(set(keys)), "Duplicate sources retrieved")

    def test_documents_checklist_generated(self):
        docs = self.res["documents_recommended"]
        self.assertTrue(any("Arrest memo" in d for d in docs))
        self.assertTrue(any("Complete FIR" in d for d in docs))
        self.assertTrue(any("Remand" in d for d in docs))
        self.assertTrue(any("repayment" in d.lower() or "bank" in d.lower() or "contract" in d.lower() for d in docs))


class TestRefusesNonFIR(unittest.TestCase):
    """Ensure non-substantive text is rejected with proper guidance."""

    def test_unfilled_template_rejected(self):
        r = S.audit_fir(TEMPLATE, arrest_made=True)
        self.assertEqual(r["status"], "insufficient_input")
        self.assertIn("Insufficient Information", r["overall_status"])

    def test_gibberish_rejected(self):
        r = S.audit_fir(GIBBERISH, arrest_made=True)
        self.assertEqual(r["status"], "insufficient_input")

    def test_empty_string_rejected(self):
        r = S.audit_fir("", arrest_made=True)
        self.assertEqual(r["status"], "error")
        self.assertIn("Error: FIR text cannot be empty", r["message"])


class TestNoticeDistinction(unittest.TestCase):
    """Ensure distinction between not mentioned and explicitly stated as missing."""

    def test_notice_not_mentioned_is_informational(self):
        fir = "Accused was arrested on 10.01.2024 for theft under Section 379 IPC."
        r = S.audit_fir(fir, arrest_made=True)
        rf = r["read_from_fir"]
        self.assertIn("Not stated", r["procedural_findings"][0]["read_from_fir"])

    def test_notice_explicitly_absent_is_amber(self):
        fir = "Accused was arrested under Section 420 IPC and no notice was issued before arrest."
        r = S.audit_fir(fir, arrest_made=True)
        codes = [f["code"] for f in r["procedural_findings"]]
        self.assertIn("ARREST_NO_NOTICE_ISSUED", codes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
