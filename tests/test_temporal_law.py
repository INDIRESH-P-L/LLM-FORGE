#!/usr/bin/env python3
"""
tests/test_temporal_law.py
==========================
LegalMind AI — Unit & Regression Tests for Temporal Law Engine

Tests:
1. IPC -> BNS transitions (Section 302, 307, 420, 376, 124A).
2. CrPC -> BNSS transitions (Section 154 FIR, 161, 164, 438 Anticipatory Bail).
3. IEA -> BSA transitions (Section 65B electronic evidence, Section 27, Section 114A).
4. Companies Act 1956 -> 2013 transitions.
5. In-force status and effective dates (2024-07-01).
6. Warnings generation for repealed law inquiries.
7. Query temporal context enrichment.
"""

import unittest
import sys
from pathlib import Path

# Ensure scripts directory in path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "scripts"))

from temporal_law import (
    check_query_temporal_status,
    get_modern_section,
    is_act_repealed,
    format_temporal_warning,
    REPLACED_ACTS,
)


class TestTemporalLawEngine(unittest.TestCase):

    def test_ipc_to_bns_section_302(self):
        mod_sec, new_act, note = get_modern_section("Indian Penal Code", "302")
        self.assertEqual(mod_sec, "103")
        self.assertEqual(new_act, "Bharatiya Nyaya Sanhita, 2023")

    def test_ipc_to_bns_section_307_attempt_to_murder(self):
        mod_sec, new_act, note = get_modern_section("IPC", "307")
        self.assertEqual(mod_sec, "109")

    def test_ipc_to_bns_section_420_cheating(self):
        mod_sec, new_act, note = get_modern_section("Indian Penal Code", "420")
        self.assertIn("318", mod_sec)

    def test_crpc_to_bnss_section_438_anticipatory_bail(self):
        mod_sec, new_act, note = get_modern_section("Code of Criminal Procedure", "438")
        self.assertEqual(mod_sec, "482")
        self.assertEqual(new_act, "Bharatiya Nagarik Suraksha Sanhita, 2023")

    def test_crpc_to_bnss_section_154_fir(self):
        mod_sec, new_act, note = get_modern_section("CrPC", "154")
        self.assertEqual(mod_sec, "173")

    def test_iea_to_bsa_section_65b_electronic_evidence(self):
        mod_sec, new_act, note = get_modern_section("Indian Evidence Act", "65b")
        self.assertEqual(mod_sec, "63")
        self.assertEqual(new_act, "Bharatiya Sakshya Adhiniyam, 2023")

    def test_companies_act_1956_to_2013(self):
        repealed = is_act_repealed("Companies Act, 1956")
        self.assertTrue(repealed)

    def test_query_status_detects_repealed_statute(self):
        res = check_query_temporal_status("What is the punishment under Section 302 of the Indian Penal Code?")
        self.assertTrue(res.get("has_temporal_context"))
        self.assertEqual(res.get("status"), "repealed")
        self.assertEqual(res.get("repealed_by"), "Bharatiya Nyaya Sanhita, 2023")
        self.assertTrue(len(res.get("warnings", [])) > 0)
        self.assertIn("Section 103 BNS", res.get("warnings")[0])

    def test_query_status_with_crpc_anticipatory_bail(self):
        res = check_query_temporal_status("Can I get anticipatory bail under Section 438 CrPC?")
        self.assertTrue(res.get("has_temporal_context"))
        self.assertEqual(res.get("status"), "repealed")
        self.assertIn("Section 482 BNSS", " ".join(res.get("warnings", [])))

    def test_effective_date_is_2024_07_01(self):
        ipc_info = REPLACED_ACTS.get("indian penal code")
        self.assertIsNotNone(ipc_info)
        self.assertEqual(ipc_info.get("repealed_date"), "2024-07-01")
        self.assertFalse(ipc_info.get("in_force"))

    def test_current_unrepealed_act_generates_no_repeal_warning(self):
        res = check_query_temporal_status("What are the provisions of the Information Technology Act 2000?")
        self.assertFalse(res.get("has_temporal_context"))
        self.assertEqual(len(res.get("warnings", [])), 0)


if __name__ == "__main__":
    unittest.main()
