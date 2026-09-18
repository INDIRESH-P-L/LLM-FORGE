"""
tests/test_query_typo_tolerance.py
==================================
Misspelled legal terms must still reach the right provision.

"what is the artucle 57" previously reached retrieval verbatim: BM25 matched no
token, and the statutory matcher never saw the word "article", so the Article 57
chunks were never looked up.

Run: .venv/bin/python -m unittest tests.test_query_typo_tolerance -v
"""

import sys
import unittest
from pathlib import Path

# scripts/ holds retriever.py; app.main puts it on the path at runtime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from retriever import correct_legal_typos, extract_statutory_targets, preprocess_query


class TestTypoCorrection(unittest.TestCase):

    def test_reported_query(self):
        self.assertEqual(correct_legal_typos("what is the artucle 57"), "what is the article 57")

    def test_common_legal_typos(self):
        cases = {
            "artucle 21": "article 21",
            "secton 302 of the penal code": "section 302 of the penal code",
            "jugdement of the supreme court": "judgement of the supreme court",
            "the consitution of india": "the constitution of india",
            "petitionar and respondant": "petitioner and respondent",
            "aquittal after trial": "acquittal after trial",
            "negotiabel instruments act": "negotiable instruments act",
            "anticipatry bail": "anticipatory bail",
            "juridiction of the tribunal": "jurisdiction of the tribunal",
            "quashng the fir": "quashing the fir",
        }
        for typo, expected in cases.items():
            self.assertEqual(correct_legal_typos(typo), expected, typo)

    def test_correct_text_is_untouched(self):
        for text in ["what is article 57",
                     "section 138 of the Negotiable Instruments Act",
                     "bail under PMLA section 45",
                     "Kesavananda Bharati v. State of Kerala",
                     "Justice K.S. Puttaswamy (Retd.) v. Union of India",
                     "Maneka Gandhi habeas corpus writ petition",
                     "murder trial evidence and witnesses"]:
            self.assertEqual(correct_legal_typos(text), text, text)

    def test_numbers_and_mixed_tokens_survive(self):
        for text in ["article 356(1)", "section 138A", "2022 SCC OnLine SC 929", "Order 7 Rule 11 CPC"]:
            self.assertEqual(correct_legal_typos(text), text, text)

    def test_unrelated_words_are_not_forced_onto_legal_terms(self):
        for text in ["pollution control board", "railway timetable enquiry",
                     "photosynthesis in plants", "cryptocurrency exchange"]:
            self.assertEqual(correct_legal_typos(text), text, text)

    def test_case_is_preserved(self):
        self.assertEqual(correct_legal_typos("Artucle 21"), "Article 21")
        self.assertEqual(correct_legal_typos("ARTUCLE 21"), "ARTICLE 21")

    def test_short_words_are_left_alone(self):
        # too short to correct safely; must not become "act"/"article"/"bail"
        for text in ["act", "writ", "bail", "tax", "fir"]:
            self.assertEqual(correct_legal_typos(text), text, text)

    def test_correction_runs_inside_preprocess_query(self):
        self.assertIn("article 57", preprocess_query("what is the artucle 57").lower())

    def test_statutory_target_now_extracted(self):
        """The point of the fix: the provision lookup can now find the target."""
        self.assertEqual(extract_statutory_targets("what is the artucle 57"), [])
        targets = extract_statutory_targets(preprocess_query("what is the artucle 57"))
        self.assertTrue(targets, "corrected query should yield a statutory target")
        self.assertEqual(targets[0]["type"], "article")
        self.assertEqual(str(targets[0]["num"]), "57")

    def test_typo_queries_all_resolve_to_targets(self):
        for q, kind, num in [("artucle 21 right to life", "article", "21"),
                             ("secton 302 ipc punishment", "section", "302"),
                             ("artical 14 equality", "article", "14")]:
            targets = extract_statutory_targets(preprocess_query(q))
            self.assertTrue(targets, q)
            self.assertEqual(targets[0]["type"], kind, q)
            self.assertEqual(str(targets[0]["num"]), num, q)

    def test_empty_and_odd_input(self):
        for text in ["", "   ", "???", "57"]:
            correct_legal_typos(text)   # must not raise


if __name__ == "__main__":
    unittest.main()
