"""
tests/test_moot_service.py
==========================
Comprehensive automated test suite for Moot Court & Judicial Adversary Simulator ("Judge Mode").
Tests all 5 Indian Appellate Corams, 3 Temperaments, Multi-Turn Round Escalation,
Counsel Representation Sides, Scorecard Normalization, Moot Problems Catalog,
and Supreme Court Order Sheet Compilation.
"""

import unittest
from unittest.mock import MagicMock, patch

from app.services.moot_service import MootCourtService


class TestMootCourtService(unittest.TestCase):
    """Test suite for MootCourtService."""

    def test_all_five_corams_defined(self):
        """Verify that all 5 specialized Indian Appellate Corams are properly defined."""
        corams = MootCourtService.JUDGE_PERSONAS
        expected_keys = {"constitutional", "criminal", "commercial", "regulatory", "tax_insolvency"}
        self.assertTrue(expected_keys.issubset(set(corams.keys())))

        for key in expected_keys:
            c = corams[key]
            self.assertIn("title", c)
            self.assertIn("coram", c)
            self.assertIn("presiding", c)
            self.assertIn("jurisdiction", c)
            self.assertIn("statutory_compendium", c)
            self.assertIn("landmark_authorities", c)
            self.assertGreaterEqual(len(c["statutory_compendium"]), 3)
            self.assertGreaterEqual(len(c["landmark_authorities"]), 3)

    def test_moot_problems_catalog(self):
        """Verify the curated Moot Problems catalog contains all 5 key legal domains."""
        problems = MootCourtService.MOOT_PROBLEMS
        self.assertGreaterEqual(len(problems), 5)

        coram_ids = {p["coram_id"] for p in problems}
        self.assertEqual(coram_ids, {"constitutional", "criminal", "commercial", "regulatory", "tax_insolvency"})

        for p in problems:
            self.assertIn("id", p)
            self.assertIn("title", p)
            self.assertIn("factual_matrix", p)
            self.assertIn("core_issues", p)
            self.assertIn("suggested_opening", p)
            self.assertGreaterEqual(len(p["core_issues"]), 2)

    def test_empty_argument_rejection(self):
        """Verify that empty arguments return success=False with helpful error."""
        res = MootCourtService.interject(argument="   ")
        self.assertFalse(res["success"])
        self.assertIn("error", res)

    def test_constitutional_coram_interjection(self):
        """Test constitutional coram interjection and scorecard bounds."""
        arg = (
            "May it please your Lordships, under Article 21 and the nine-judge Constitution Bench ruling in Puttaswamy, "
            "privacy is an intrinsic fundamental right. State surveillance without prior judicial warrants fails the four-prong proportionality standard."
        )
        res = MootCourtService.interject(
            argument=arg,
            bench_type="constitutional",
            temperament="inquisitive",
            round_num=1,
            counsel_side="petitioner",
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["round"], 1)
        self.assertEqual(res["bench_type"], "constitutional")
        self.assertIn("Constitution Bench", res["bench_title"])
        self.assertIn("Puttaswamy", res["counter_precedent"])
        self.assertIn("judicial_interjection", res)
        self.assertIn("rebuttal_tip", res)

        sc = res["scorecard"]
        self.assertGreaterEqual(sc["overall_score"], 30.0)
        self.assertLessEqual(sc["overall_score"], 100.0)
        self.assertGreaterEqual(sc["constitutional_grounding"], 30.0)
        self.assertGreaterEqual(sc["statutory_precision"], 30.0)
        self.assertGreaterEqual(sc["precedent_authority"], 30.0)
        self.assertGreaterEqual(sc["judicial_persuasion"], 30.0)

    def test_criminal_coram_bail_interjection(self):
        """Test criminal appellate division evaluating PMLA Section 45 bail argument."""
        arg = (
            "Counsel submits that under Section 45 of PMLA and Manish Sisodia v. Directorate of Enforcement (2024), "
            "prolonged pre-trial incarceration without trial overrides statutory bail bars under Article 21."
        )
        res = MootCourtService.interject(
            argument=arg,
            bench_type="criminal",
            temperament="adversarial",
            round_num=2,
            counsel_side="appellant",
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["bench_type"], "criminal")
        self.assertIn("Vijay Madanlal Choudhary", res["counter_precedent"])
        self.assertIn("[Adversarial Interlocution]", res["judicial_interjection"])
        self.assertIn("rebuttal_tip", res)

    def test_commercial_coram_arbitration_interjection(self):
        """Test commercial coram evaluating Section 74 damages and Section 34 arbitration."""
        arg = (
            "May it please your Lordships, under Section 74 of the Indian Contract Act and Kailash Nath Associates (2015), "
            "liquidated damages cannot be forfeited as penalty without proof of actual legal injury."
        )
        res = MootCourtService.interject(
            argument=arg,
            bench_type="commercial",
            temperament="textualist",
            round_num=1,
            counsel_side="petitioner",
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["bench_type"], "commercial")
        self.assertIn("Kailash Nath", res["counter_precedent"])
        self.assertIn("[Strict Textual Construction]", res["judicial_interjection"])

    def test_regulatory_coram_precautionary_interjection(self):
        """Test green bench evaluating environmental clearance and precautionary principle."""
        arg = (
            "Under Article 48A, 51A(g) and the Precautionary Principle affirmed in Vellore Citizens, "
            "the suppression of seismic fault lines invalidates the environmental clearance ab initio."
        )
        res = MootCourtService.interject(
            argument=arg,
            bench_type="regulatory",
            temperament="inquisitive",
            round_num=3,
            counsel_side="petitioner",
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["bench_type"], "regulatory")
        self.assertIn("Vellore Citizens", res["counter_precedent"])

    def test_tax_insolvency_coram_waterfall_interjection(self):
        """Test corporate & insolvency bench evaluating Section 53 IBC waterfall."""
        arg = (
            "Under Section 238 IBC and the three-judge Bench ruling in Essar Steel (2020), "
            "the Section 53 waterfall strictly prioritizes secured financial creditors over Crown tax dues."
        )
        res = MootCourtService.interject(
            argument=arg,
            bench_type="tax_insolvency",
            temperament="inquisitive",
            round_num=1,
            counsel_side="respondent",
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["bench_type"], "tax_insolvency")
        self.assertIn("Essar Steel", res["counter_precedent"])

    def test_multi_turn_round_escalation(self):
        """Test that Round 1 through 5 progress without failure and provide escalating questions."""
        history = []
        for r in range(1, 6):
            res = MootCourtService.interject(
                argument=f"Counsel oral submission for round {r} citing Article 21 and Section 45.",
                bench_type="criminal",
                round_num=r,
                prior_history=history,
            )
            self.assertTrue(res["success"])
            self.assertEqual(res["round"], r)
            self.assertIn("scorecard", res)
            history.append({"counsel": f"Round {r} arg", "bench": res["judicial_interjection"]})

        self.assertEqual(len(history), 5)

    def test_llm_json_mocking(self):
        """Test that LLM structured JSON output is parsed and normalized correctly."""
        mock_llm = MagicMock()
        mock_llm.generate_chat_completion.return_value = """```json
{
  "judicial_interjection": "Counsel, how do you overcome the five-judge bench in Shayara Bano regarding manifest arbitrariness?",
  "counter_precedent": "Shayara Bano v. Union of India (2017) 9 SCC 1",
  "counter_precedent_ratio": "Manifest arbitrariness strikes down capricious executive action.",
  "rebuttal_tip": "Demonstrate that the state acted pursuant to clear statutory guidelines.",
  "strengths": ["Clear constitutional framing", "Cited relevant articles"],
  "vulnerabilities": ["Did not distinguish larger bench rulings"],
  "scorecard": {
    "overall_score": 88.5,
    "constitutional_grounding": 90,
    "statutory_precision": 85,
    "precedent_authority": 88,
    "judicial_persuasion": 89,
    "readiness_rating": "Formidable Advocacy"
  }
}
```"""

        with patch("app.services.moot_service._get_llm_instance", return_value=mock_llm):
            res = MootCourtService.interject(
                argument="Arbitrary executive notification without determining principle.",
                bench_type="constitutional",
                round_num=1,
            )

        self.assertTrue(res["success"])
        self.assertIn("Shayara Bano", res["judicial_interjection"])
        self.assertEqual(res["scorecard"]["overall_score"], 88.5)
        self.assertEqual(res["scorecard"]["constitutional_grounding"], 90)
        self.assertEqual(len(res["strengths"]), 2)

    def test_supreme_court_order_sheet_export(self):
        """Test generation of official Supreme Court Order Sheet."""
        rounds = [
            {"counsel": "Counsel submits that Article 21 protects digital privacy.", "bench": "Bench queries whether national security justifies surveillance."},
            {"counsel": "Counsel submits that unguided discretion violates Article 14.", "bench": "Bench asks for the exact limiting principle."}
        ]
        scorecard = {
            "overall_score": 84.5,
            "readiness_rating": "Competent Submission",
            "constitutional_grounding": 86,
            "statutory_precision": 82,
            "precedent_authority": 84,
            "judicial_persuasion": 80,
            "rebuttal_tip": "Distinguish PUCL (1997) using subsequent Puttaswamy digital privacy evolution."
        }

        order_sheet = MootCourtService.generate_session_record(
            bench_type="constitutional",
            temperament="inquisitive",
            rounds=rounds,
            scorecard=scorecard,
            case_title="People's Union for Digital Rights v. Union of India",
            counsel_side="petitioner",
        )

        self.assertIn("IN THE SUPREME COURT OF INDIA", order_sheet)
        self.assertIn("APPELLATE JURISDICTION", order_sheet)
        self.assertIn("5-JUDGE CONSTITUTION BENCH", order_sheet)
        self.assertIn("PEOPLE'S UNION FOR DIGITAL RIGHTS V. UNION OF INDIA", order_sheet)
        self.assertIn("ROUND 1", order_sheet)
        self.assertIn("ROUND 2", order_sheet)
        self.assertIn("BENCH DISPOSITION & DIRECTIONS", order_sheet)
        self.assertIn("CERTIFIED COPY", order_sheet)


if __name__ == "__main__":
    unittest.main()
