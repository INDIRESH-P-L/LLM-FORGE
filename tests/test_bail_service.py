"""
tests/test_bail_service.py
==========================
Bail Feasibility Matrix: the score and its reasoning must follow the inputs.
Run: .venv/bin/python -m unittest tests.test_bail_service -v
"""

import itertools
import unittest

from app.services.bail_service import BailFeasibilityService

CATEGORIES = ["Economic Offence", "Murder", "Cheating", "Heinous Offence", "Cyber Crime", "General IPC"]


def assess(category, special=False, custody=45, chargesheet=True, **extra):
    return BailFeasibilityService.assess_bail(
        offense_category=category, is_special_act=special,
        custody_days=custody, chargesheet_filed=chargesheet, **extra)


class TestBailFeasibility(unittest.TestCase):

    def test_bailable_short_custody_is_high(self):
        r = assess("General IPC", special=False, custody=10, chargesheet=True)
        self.assertEqual(r["verdict"], "High Probability")
        self.assertIn("Arnesh Kumar v. State of Bihar (2014) 8 SCC 273", r["relevant_precedents"])

    def test_heinous_special_act_short_custody_is_low(self):
        r = assess("Heinous Offence", special=True, custody=20, chargesheet=False)
        self.assertEqual(r["verdict"], "Low Probability")
        self.assertTrue(any("S.37 NDPS" in f for f in r["risk_factors"]))

    def test_every_input_changes_the_result(self):
        base = assess("Economic Offence", special=False, custody=45, chargesheet=True)
        for changed in (assess("Murder", special=False, custody=45, chargesheet=True),
                        assess("Economic Offence", special=True, custody=45, chargesheet=True),
                        assess("Economic Offence", special=False, custody=400, chargesheet=True),
                        assess("Economic Offence", special=False, custody=45, chargesheet=False)):
            self.assertNotEqual(base["bail_score"], changed["bail_score"])

    def test_offence_severity_orders_the_score(self):
        scores = {c: assess(c)["bail_score"] for c in CATEGORIES}
        self.assertGreater(scores["General IPC"], scores["Economic Offence"])
        self.assertGreater(scores["Economic Offence"], scores["Murder"])
        self.assertGreater(scores["Murder"], scores["Heinous Offence"])

    def test_special_act_lowers_score_and_cites_twin_conditions(self):
        for c in CATEGORIES:
            plain, special = assess(c), assess(c, special=True)
            self.assertLess(special["bail_score"], plain["bail_score"], c)
            self.assertIsNotNone(special["special_act_regime"])
        pmla = assess("Economic Offence", special=True)
        self.assertEqual(pmla["special_act_regime"], "PMLA S.45")
        self.assertIn("Vijay Madanlal Choudhary v. Union of India, 2022 SCC OnLine SC 929", pmla["relevant_precedents"])

    def test_longer_custody_never_lowers_score(self):
        for c, special, chg in itertools.product(CATEGORIES, (False, True), (False, True)):
            scores = [assess(c, special, d, chg)["bail_score"] for d in (0, 30, 90, 180, 365, 730)]
            self.assertEqual(scores, sorted(scores), (c, special, chg))

    def test_long_custody_meaningfully_shifts_score(self):
        short, long_ = assess("Murder", custody=20), assess("Murder", custody=400)
        self.assertGreaterEqual(long_["bail_score"] - short["bail_score"], 20)

    def test_chargesheet_filed_raises_score(self):
        for c in CATEGORIES:
            self.assertGreater(assess(c, chargesheet=True)["bail_score"],
                               assess(c, chargesheet=False)["bail_score"], c)

    def test_default_bail_accrues_without_chargesheet(self):
        r = assess("Economic Offence", custody=95, chargesheet=False)   # 10-year offence → 90 days
        self.assertEqual(r["default_bail_limit_days"], 90)
        self.assertEqual(r["verdict"], "High Probability")
        self.assertIn("Bikramjit Singh v. State of Punjab (2020) 10 SCC 616", r["relevant_precedents"])
        self.assertLess(assess("Economic Offence", custody=85, chargesheet=False)["bail_score"], r["bail_score"])
        # NDPS/UAPA: 180 days, so 120 days is not enough
        self.assertNotIn("Bikramjit Singh v. State of Punjab (2020) 10 SCC 616",
                         assess("Heinous Offence", special=True, custody=120, chargesheet=False)["relevant_precedents"])

    def test_default_bail_not_claimed_once_chargesheet_filed(self):
        r = assess("Murder", custody=200, chargesheet=True)
        self.assertIsNone(r["default_bail_limit_days"])
        self.assertFalse(any("default bail" in f.lower() for f in r["positive_factors"]))

    def test_twin_condition_ceiling_relaxes_with_prolonged_custody(self):
        self.assertLessEqual(assess("General IPC", special=True, custody=60)["bail_score"], 45)
        najeeb = assess("Heinous Offence", special=True, custody=800)
        self.assertIn("Union of India v. K.A. Najeeb (2021) 3 SCC 713", najeeb["relevant_precedents"])

    def test_half_sentence_release_not_for_life_offences(self):
        cyber = assess("Cyber Crime", custody=600)   # half of 3 years = 548 days
        self.assertTrue(any("S.436A" in f for f in cyber["positive_factors"]))
        murder = assess("Murder", custody=6000)
        self.assertFalse(any("S.436A" in f for f in murder["positive_factors"]))

    def test_no_invented_facts_when_optional_inputs_missing(self):
        r = assess("General IPC")
        self.assertFalse(any("antecedents" in f.lower() for f in r["positive_factors"] + r["risk_factors"]))

    def test_precedents_depend_on_inputs(self):
        a = set(assess("General IPC", False, 10, True)["relevant_precedents"])
        b = set(assess("Heinous Offence", True, 800, False)["relevant_precedents"])
        self.assertFalse(a & b)

    def test_distinct_outputs_across_input_grid(self):
        outputs = {assess(c, s, d, ch)["bail_score"]
                   for c, s, d, ch in itertools.product(CATEGORIES, (False, True), (10, 45, 120, 400), (True, False))}
        self.assertGreaterEqual(len(outputs), 25)

    def test_breakdown_sums_to_score_and_response_shape_kept(self):
        for c, s, d, ch in itertools.product(CATEGORIES, (False, True), (0, 100, 1000), (True, False)):
            r = assess(c, s, d, ch)
            self.assertEqual(sum(x["points"] for x in r["score_breakdown"]), r["bail_score"])
            self.assertTrue(5 <= r["bail_score"] <= 95)
            for key in ("bail_score", "verdict", "positive_factors", "risk_factors",
                        "anticipated_objections", "relevant_precedents"):
                self.assertIn(key, r)

    def test_free_text_category_from_cli(self):
        self.assertEqual(assess("Cheating / Financial Crime")["offence_profile"], "Cheating / criminal breach of trust")
        unknown = assess("Something else entirely")
        self.assertTrue(any("not recognised" in f for f in unknown["risk_factors"]))


if __name__ == "__main__":
    unittest.main()
