#!/usr/bin/env python3
"""
tests/test_citation_verification.py
===================================
LegalMind AI — Unit Tests for Citation Verification & Hallucination Defense

Tests:
1. Indian legal citation format regex matching (AIR, SCC, SCR, Article, Section).
2. Presence validation against retrieved evidence chunks.
3. Hallucination scoring and unverified citation detection.
4. Confidence score calibration and downgrading on hallucinated citations.
5. Integration with CitationVerifierService.
"""

import unittest
import sys
from pathlib import Path

# Ensure paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from scripts.citation_verifier import (
    extract_cited_provisions_from_text,
    verify_citations_in_response,
    validate_citation_format,
    format_citation_verification_report,
)
from app.services.citation_verifier import get_citation_verifier_service


class TestCitationVerification(unittest.TestCase):

    def setUp(self):
        self.retrieved_chunks = [
            {
                "citation_id": 1,
                "document_id": "const_art_21",
                "document_type": "constitution",
                "title": "Constitution of India — Article 21",
                "text": "Article 21. Protection of life and personal liberty. No person shall be deprived of his life or personal liberty except according to procedure established by law.",
            },
            {
                "citation_id": 2,
                "document_id": "maneka_gandhi_1978",
                "document_type": "judgment",
                "title": "Maneka Gandhi v. Union of India",
                "text": "In Maneka Gandhi v. Union of India, AIR 1978 SC 597 : (1978) 1 SCC 248, the Supreme Court held that the procedure under Article 21 must be right, just, and fair.",
            },
        ]

    def test_extract_citations_finds_articles_and_reporters(self):
        text = "Under Article 21 and Section 302 of the Indian Penal Code, as held in AIR 1978 SC 597 and (1978) 1 SCC 248."
        extracted = extract_cited_provisions_from_text(text)
        self.assertTrue(any("Article 21" in c for c in extracted))
        self.assertTrue(any("Section 302" in c for c in extracted))
        self.assertTrue(any("AIR 1978 SC 597" in c for c in extracted))
        self.assertTrue(any("SCC" in c for c in extracted))

    def test_validate_citation_format(self):
        is_valid, ctype = validate_citation_format("AIR 1973 SC 1461")
        self.assertTrue(is_valid)
        self.assertEqual(ctype, "judgment")

        is_valid, ctype = validate_citation_format("Article 21")
        self.assertTrue(is_valid)
        self.assertEqual(ctype, "constitution")

        is_valid, ctype = validate_citation_format("Section 438")
        self.assertTrue(is_valid)
        self.assertEqual(ctype, "statute")

        is_valid, ctype = validate_citation_format("Random text 12345")
        self.assertFalse(is_valid)

    def test_fully_grounded_response_has_zero_hallucination(self):
        text = "Article 21 guarantees personal liberty. The Supreme Court in AIR 1978 SC 597 established the fairness standard."
        summary = verify_citations_in_response(text, self.retrieved_chunks)
        self.assertEqual(summary.unverified_count, 0)
        self.assertEqual(summary.hallucination_score, 0.0)
        self.assertEqual(summary.confidence_label, "high")

    def test_fabricated_citation_triggers_hallucination_penalty(self):
        text = (
            "Under Article 21 and Section 9999 of the Imaginary Space Act, 2099, "
            "as decided in Fake Case v. State, AIR 2099 SC 99999."
        )
        summary = verify_citations_in_response(text, self.retrieved_chunks)
        self.assertTrue(summary.unverified_count >= 1)
        self.assertTrue(summary.hallucination_score > 0.0)
        self.assertIn(summary.confidence_label, ("medium", "low"))

    def test_service_wrapper(self):
        service = get_citation_verifier_service()
        text = "Article 21 protection of life. AIR 1978 SC 597."
        summary = service.verify(text, self.retrieved_chunks)
        self.assertIsNotNone(summary)
        report = service.generate_report(summary)
        self.assertIn("Citation Verification", report)


if __name__ == "__main__":
    unittest.main()
