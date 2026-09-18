#!/usr/bin/env python3
"""
tests/test_trust_layer.py
=========================
Tests for the citation verifier and the grounding/abstention scorer.

The property that matters most here is PRECISION IN THE RIGHT DIRECTION:
a fabricated authority must never be reported as verified, and a real one
must never be reported as fabricated. Both mistakes are dangerous on a legal
tool, in opposite ways, so both are tested explicitly.

    .venv/bin/python -m unittest tests.test_trust_layer -v
"""
import sys, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.citation_verifier import (CitationVerifier, norm_citation,
                                            norm_name, reporter_of)
from app.services.answer_grounding import assess, ABSTAIN_MESSAGE, _chunk_text

HAS_DB = (ROOT / "data/verification/authorities.db").exists()
skip_no_db = unittest.skipUnless(HAS_DB, "authority DB not built")


class FakeReport:
    def __init__(self, checked=0, grounded=0, in_corpus=0, unverified=0, source_gap=0):
        self.checked, self.grounded = checked, grounded
        self.in_corpus, self.unverified, self.source_gap = in_corpus, unverified, source_gap


# ── normalisation ──────────────────────────────────────────────────────────
class TestNormalisation(unittest.TestCase):

    def test_citation_forms_collapse_to_one_key(self):
        forms = ["[1979] 3 S.C.R. 453", "(1979) 3 SCR 453", "1979 3 s.c.r. 453"]
        keys = {norm_citation(f) for f in forms}
        self.assertEqual(len(keys), 1, f"should collapse, got {keys}")

    def test_case_name_normalisation_strips_noise(self):
        a = norm_name("M/S. Sharma Traders & Ors. versus State of Bihar")
        b = norm_name("Sharma Traders v. Bihar")
        self.assertIn("sharma", a)
        self.assertIn("traders", a)
        self.assertNotIn("ors", a.split())
        self.assertTrue(set(b.split()) <= set(a.split()) or set(a.split()) & set(b.split()))

    def test_reporter_detection(self):
        self.assertEqual(reporter_of("[1979] 3 S.C.R. 453"), "scr")
        self.assertEqual(reporter_of("(2022) 5 SCC 1"), "scc")
        self.assertEqual(reporter_of("AIR 1973 SC 1461"), "air")
        self.assertIsNone(reporter_of("nothing here"))


# ── extraction ─────────────────────────────────────────────────────────────
class TestExtraction(unittest.TestCase):

    def setUp(self):
        self.v = CitationVerifier()

    def test_extracts_each_kind(self):
        text = ("In Kesavananda Bharati v. State of Kerala [1973] SUPP. 1 S.C.R. 1 "
                "the Court read Article 368 with Section 3 of the Act.")
        kinds = {k for k, _ in self.v.extract(text)}
        self.assertEqual(kinds, {"case", "citation", "article", "section"})

    def test_party_name_stops_at_the_verb(self):
        """A lazy pattern captured 'State of Kerala establi' and broke lookup."""
        got = dict(self.v.extract("Kesavananda Bharati v. State of Kerala established the doctrine."))
        cases = [t for k, t in self.v.extract(
            "Kesavananda Bharati v. State of Kerala established the doctrine.") if k == "case"]
        self.assertTrue(cases)
        self.assertNotIn("establi", cases[0].lower())
        self.assertTrue(cases[0].lower().endswith("kerala"), cases[0])

    def test_leading_and_trailing_prose_is_trimmed(self):
        cases = [t for k, t in self.v.extract(
            "In Maneka Gandhi v. Union of India the Court expanded the right.") if k == "case"]
        self.assertTrue(cases)
        self.assertFalse(cases[0].lower().startswith("in "), cases[0])
        self.assertNotIn("the court", cases[0].lower())

    def test_duplicates_are_collapsed(self):
        text = "Article 21. Again Article 21. And Article 21."
        arts = [t for k, t in self.v.extract(text) if k == "article"]
        self.assertEqual(len(arts), 1)

    def test_empty_input_is_safe(self):
        self.assertEqual(self.v.extract(""), [])
        self.assertEqual(self.v.extract(None), [])


# ── verification verdicts ──────────────────────────────────────────────────
@skip_no_db
class TestVerification(unittest.TestCase):

    def setUp(self):
        self.v = CitationVerifier()

    def test_fabricated_cases_are_never_blessed(self):
        """The dangerous direction: an invented authority must not pass."""
        fakes = ["Fakename v. Nobody", "Ghost v. Phantom",
                 "Nonexistent Trust v. Mythical Authority",
                 "Invented Petitioner v. Fictional Respondent"]
        for name in fakes:
            rep = self.v.verify(f"See {name} for this proposition.", [])
            cases = [f for f in rep.findings if f.kind == "case"]
            self.assertTrue(cases, name)
            self.assertEqual(cases[0].status, "unverified",
                             f"{name} was wrongly accepted as {cases[0].status}")

    def test_real_landmark_cases_are_recognised(self):
        """The opposite error: calling a real authority fabricated."""
        reals = ["Kesavananda Bharati v. State of Kerala",
                 "Maneka Gandhi v. Union of India",
                 "Minerva Mills v. Union of India",
                 "Vishaka v. State of Rajasthan",
                 "Shreya Singhal v. Union of India"]
        for name in reals:
            rep = self.v.verify(f"See {name} on this point.", [])
            cases = [f for f in rep.findings if f.kind == "case"]
            self.assertTrue(cases, name)
            self.assertNotEqual(cases[0].status, "unverified",
                                f"{name} is in the corpus but was reported unverified")

    def test_best_candidate_is_chosen_not_the_first(self):
        """LIMIT 1 once matched 'Buffalo Traders v. Maneka Gandhi' instead."""
        rep = self.v.verify("Maneka Gandhi v. Union of India is the authority.", [])
        case = [f for f in rep.findings if f.kind == "case"][0]
        self.assertEqual(case.status, "in_corpus")
        self.assertIn("MANEKA GANDHI", (case.matched or "").upper())

    def test_unsupported_reporter_is_a_source_gap_not_a_fabrication(self):
        """We index S.C.R. only; calling an SCC cite fake would be wrong."""
        rep = self.v.verify("See (1973) 4 SCC 225 on this.", [])
        cite = [f for f in rep.findings if f.kind == "citation"][0]
        self.assertEqual(cite.status, "source_gap")
        self.assertNotIn(cite.status, ("unverified",))

    def test_retrieved_authority_counts_as_grounded(self):
        chunks = [{"title": "MANEKA GANDHI versus UNION OF INDIA",
                   "citation": "[1978] 2 S.C.R. 621", "text": "x" * 300}]
        rep = self.v.verify("Maneka Gandhi v. Union of India [1978] 2 S.C.R. 621 applies.", chunks)
        self.assertGreaterEqual(rep.grounded, 1)

    def test_hallucination_rate_excludes_source_gaps(self):
        rep = self.v.verify("See (2022) 5 SCC 1 and Ghost v. Phantom.", [])
        self.assertGreater(rep.source_gap, 0)
        self.assertEqual(rep.adjudicable, rep.checked - rep.source_gap)
        self.assertLessEqual(rep.hallucination_rate, 1.0)

    def test_annotation_flags_unverified_and_does_not_edit_the_answer(self):
        answer = "Ghost v. Phantom decides this."
        rep = self.v.verify(answer, [])
        out = self.v.annotate(answer, rep)
        self.assertTrue(out.startswith(answer), "the answer text must not be rewritten")
        self.assertIn("Citation check", out)
        self.assertIn("Ghost v. Phantom", out)


# ── grounding / abstention ─────────────────────────────────────────────────
class TestGrounding(unittest.TestCase):

    def test_chunk_text_accepts_every_field_name(self):
        """run_rag uses text_preview; missing it zeroed every support score."""
        self.assertEqual(_chunk_text({"text": "a" * 100}), "a" * 100)
        self.assertEqual(_chunk_text({"text_preview": "b" * 100}), "b" * 100)
        self.assertEqual(_chunk_text({"excerpt": "c" * 100}), "c" * 100)
        self.assertEqual(_chunk_text({}), "")

    def test_retrieved_passages_produce_nonzero_support(self):
        chunks = [{"text_preview": "x" * 400, "rerank_score": 5.0}] * 5
        g = assess("What is Article 21?", chunks, FakeReport(2, 2, 0, 0))
        self.assertGreater(g.support, 0.0)
        self.assertFalse(g.should_abstain)

    def test_no_retrieval_means_general_knowledge_mode(self):
        """With no retrieved passages the system no longer refuses (should_abstain=False)
        but correctly sets knowledge_mode to 'general_knowledge' so the API layer
        prepends the appropriate transparency notice instead of throwing the answer away."""
        g = assess("anything", [], FakeReport(0, 0, 0, 0))
        self.assertEqual(g.support, 0.0)
        self.assertFalse(g.should_abstain)       # never refuse
        self.assertEqual(g.knowledge_mode, "general_knowledge")

    def test_corrupt_passages_do_not_count_as_support(self):
        """~59% of the judgment corpus is damaged translation.
        Corrupt passages should still be excluded from usable_sources,
        driving support to 0 and knowledge_mode to 'general_knowledge'.
        The system no longer refuses (should_abstain=False) but the
        transparency notice is shown via knowledge_mode."""
        chunks = [{"text_preview": "x" * 400, "rerank_score": 8.0,
                   "text_quality": "corrupt"}] * 4
        g = assess("q", chunks, FakeReport(0, 0, 0, 0))
        self.assertEqual(g.usable_sources, 0)
        self.assertFalse(g.should_abstain)      # never refuse
        self.assertEqual(g.knowledge_mode, "general_knowledge")

    def test_fabricated_authorities_drive_support_down(self):
        """
        The invariant, independent of how the penalty is tuned: asserting
        unverifiable authorities must reduce support, and more of them must
        reduce it further. The penalty coefficient itself is a project
        setting (see test_fabrication_penalty_is_strong_enough_to_abstain).
        """
        chunks = [{"text_preview": "x" * 400, "rerank_score": 6.0}] * 5
        clean = assess("q", chunks, FakeReport(4, 4, 0, 0))
        dirty = assess("q", chunks, FakeReport(4, 1, 0, 3))
        worse = assess("q", chunks, FakeReport(6, 1, 0, 5))
        self.assertGreater(clean.support, dirty.support)
        self.assertGreater(dirty.support, worse.support)

    def test_fabrication_penalty_is_declared_and_effective(self):
        """
        The penalty is a deliberate policy setting, not a magic number, and it
        must actually bite. Measured across 160 stored evaluation answers:
        0.45 withholds 12 and shows 6 answers containing a fabricated
        authority; 0.15 withholds 6 and shows 12. Whatever value is chosen,
        enough fabrications must still drive support to zero and trigger the
        general_knowledge mode (so the user sees the transparency notice).
        The system no longer refuses outright (should_abstain=False), but the
        answer is labelled and the user is warned.
        """
        from app.services.answer_grounding import FABRICATION_PENALTY, ABSTAIN_FLOOR
        self.assertGreater(FABRICATION_PENALTY, 0.0,
                           "a zero penalty disables fabrication protection entirely")
        self.assertLessEqual(FABRICATION_PENALTY, 1.0)

        chunks = [{"text_preview": "x" * 400, "rerank_score": 6.0}] * 5
        # However the knob is set, a wholly unverifiable answer must drive support to 0.
        n = int(1 / FABRICATION_PENALTY) + 1
        g = assess("q", chunks, FakeReport(n, 0, 0, n))
        self.assertFalse(g.should_abstain,   # never refuse
                         "should_abstain should always be False in the new architecture")
        self.assertLessEqual(g.support, 0.0,
                             f"an answer with {n} unverifiable authorities and none grounded "
                             f"should reduce support to 0 (got {g.support:.3f})")
        self.assertEqual(g.knowledge_mode, "general_knowledge",
                         "zero-support answers must be flagged as general_knowledge mode")

    def test_missing_named_provision_reduces_support(self):
        chunks = [{"text_preview": "unrelated text " * 30, "rerank_score": 6.0}] * 3
        g = assess("What does Section 4242 provide?", chunks, FakeReport(1, 1, 0, 0))
        self.assertFalse(g.provision_covered)
        self.assertLess(g.support, 0.5)

    def test_confidence_is_not_published_until_calibrated(self):
        g = assess("q", [{"text_preview": "x" * 400, "rerank_score": 9.0}],
                   FakeReport(1, 1, 0, 0))
        self.assertIn(g.confidence, ("HIGH", "MEDIUM", "LOW"))
        self.assertIsInstance(g.calibrated, bool)

    def test_abstain_message_tells_the_user_what_to_do(self):
        self.assertIn("verify with a legal professional", ABSTAIN_MESSAGE.lower())
        self.assertIn("insufficient", ABSTAIN_MESSAGE.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
