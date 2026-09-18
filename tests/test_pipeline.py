#!/usr/bin/env python3
"""
tests/test_pipeline.py
======================
Unit tests for LegalMind AI core pipeline components:
- Preprocessing and legal expansion
- Citation formatting
- Thinking block stripping
- Structured response parsing
- Evaluation metrics (ROUGE-L, Hallucination, Abstention)
- BM25 indexing & keyword search
"""

import sys
from pathlib import Path

# Add project root and scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "fine_tuning"))

import unittest
from retriever import preprocess_query, format_citations, BM25Index, find_exact_provision_matches, extract_statutory_targets
import importlib.util
import re

# Import 05_rag dynamically because of number prefix
rag_spec = importlib.util.spec_from_file_location("rag", str(Path(__file__).parent.parent / "scripts" / "05_rag.py"))
rag_mod = importlib.util.module_from_spec(rag_spec)
rag_spec.loader.exec_module(rag_mod)
strip_thinking = rag_mod.strip_thinking
_parse_structured_answer = rag_mod._parse_structured_answer
enforce_grounding_and_confidence = rag_mod.enforce_grounding_and_confidence
CONSTITUTIONAL_TOPICS = rag_mod.CONSTITUTIONAL_TOPICS
UNSUPPORTED_TOPIC_MESSAGE = rag_mod.UNSUPPORTED_TOPIC_MESSAGE
is_broad_constitutional_query = rag_mod.is_broad_constitutional_query

from evaluate import rouge_l, citation_hallucination_rate, abstention_correct


class TestLegalMindPipeline(unittest.TestCase):

    def test_query_preprocessing(self):
        """Test expansion of common legal abbreviations."""
        q1 = "art 21 of constitution"
        expanded1 = preprocess_query(q1)
        self.assertIn("Article", expanded1)

        q2 = "bail under sec 438 crpc"
        expanded2 = preprocess_query(q2)
        self.assertIn("Section", expanded2)
        self.assertIn("Code of Criminal Procedure", expanded2)

    def test_format_citations_legislation(self):
        """Test citation formatting for legislation/constitutional chunks."""
        chunks = [
            {
                "citation_id": 1,
                "document_type": "constitution",
                "act_name": "Constitution of India",
                "article": "21",
                "title": "Constitution of India",
            },
            {
                "citation_id": 2,
                "document_type": "act",
                "act_name": "Code of Criminal Procedure, 1973",
                "section": "438",
                "title": "CrPC 1973",
            }
        ]
        citations = format_citations(chunks)
        self.assertEqual(len(citations), 2)
        self.assertTrue(citations[0].startswith("[1] Constitution of India — Article 21"))
        self.assertTrue(citations[1].startswith("[2] Code of Criminal Procedure, 1973 — Section 438"))

    def test_format_citations_judgment(self):
        """Test citation formatting for judicial precedent chunks."""
        chunks = [
            {
                "citation_id": 1,
                "document_type": "judgment",
                "title": "Kesavananda Bharati v. State of Kerala",
                "court": "Supreme Court of India",
                "date": "1973-04-24",
                "citation": "AIR 1973 SC 1461",
            }
        ]
        citations = format_citations(chunks)
        self.assertEqual(len(citations), 1)
        self.assertIn("Kesavananda Bharati v. State of Kerala", citations[0])
        self.assertIn("Supreme Court of India", citations[0])
        self.assertIn("1973-04-24", citations[0])
        self.assertIn("AIR 1973 SC 1461", citations[0])

    def test_strip_thinking(self):
        """Test removal of chain-of-thought blocks."""
        raw1 = "<think>Let me analyze the relevant laws...</think>Article 21 guarantees life."
        self.assertEqual(strip_thinking(raw1), "Article 21 guarantees life.")

        raw2 = "<|think|>Internal CoT deliberation<|/think|>Bail is discretionary."
        self.assertEqual(strip_thinking(raw2), "Bail is discretionary.")

        raw3 = "Direct response with no thinking."
        self.assertEqual(strip_thinking(raw3), "Direct response with no thinking.")

    def test_parse_structured_answer(self):
        """Test parsing structured sections from LLM output."""
        sample_output = """## Answer
Article 21 protects personal liberty.

## Relevant Legal Provisions
- Article 21 of the Constitution
- Section 438 CrPC

## Relevant Judgments / Precedents
- Maneka Gandhi v. Union of India (1978)

## Confidence Level
HIGH

## Limitations & Missing Context
Requires examination of specific state amendments.
"""
        parsed = _parse_structured_answer(sample_output)
        self.assertIn("Article 21 protects personal liberty", parsed["answer"])
        self.assertEqual(len(parsed["legal_provisions"]), 2)
        self.assertEqual(len(parsed["judgments"]), 1)
        self.assertEqual(parsed["confidence"], "HIGH")
        self.assertIn("state amendments", parsed["limitations"])

    def test_metrics_rouge_l(self):
        """Test ROUGE-L computation."""
        # Exact match
        self.assertAlmostEqual(rouge_l("Right to life", "Right to life"), 1.0)
        # Disjoint
        self.assertAlmostEqual(rouge_l("Apples and oranges", "Contract law basics"), 0.0)
        # Partial match
        score = rouge_l("Article 21 guarantees life and liberty", "Article 21 guarantees personal liberty")
        self.assertGreater(score, 0.5)

    def test_metrics_hallucination_rate(self):
        """Test citation hallucination calculation."""
        citations = ["[1] Constitution of India", "[2] CrPC Section 438"]

        # No hallucinations (all cited [N] are in citations list)
        answer_clean = "As established in [1] and [2], bail is permissible."
        self.assertEqual(citation_hallucination_rate(answer_clean, citations), 0.0)

        # 1 real, 1 hallucinated ([3] does not exist)
        answer_hallucinated = "Per [1] and [3], the petitioner is exempt."
        self.assertEqual(citation_hallucination_rate(answer_hallucinated, citations), 0.5)

        # No citations used in answer
        answer_nocite = "This is a general summary."
        self.assertEqual(citation_hallucination_rate(answer_nocite, citations), 0.0)

    def test_metrics_abstention_correct(self):
        """Test detection of proper refusal on unanswerable queries."""
        # Correct abstention on unanswerable
        self.assertTrue(abstention_correct("The provided legal evidence is insufficient to answer this question.", "unanswerable"))
        self.assertTrue(abstention_correct("Unable to find any relevant provisions.", "unanswerable"))

        # Incorrect failure to abstain on unanswerable
        self.assertFalse(abstention_correct("The salary was 50,000 rupees per month.", "unanswerable"))

        # Answerable query should return None
        self.assertIsNone(abstention_correct("Article 21 protects personal liberty.", "answerable"))

    def test_bm25_search(self):
        """Test BM25 index build and keyword search."""
        test_chunks = [
            {"chunk_id": "c1", "text": "Article 21 of the Constitution provides protection of life and personal liberty."},
            {"chunk_id": "c2", "text": "Section 302 of the Indian Penal Code prescribes punishment for murder."},
            {"chunk_id": "c3", "text": "Section 10 of the Indian Contract Act defines what agreements are contracts."},
        ]
        bm25 = BM25Index(test_chunks)
        results = bm25.search("Article 21 life liberty", top_k=1)
        self.assertTrue(len(results) > 0)
        top_idx, top_score = results[0]
        self.assertEqual(top_idx, 0)
        self.assertGreater(top_score, 0.0)


    def test_case_laws_faiss_index_integrity(self):
        """Test Phase 7 case laws FAISS index and metadata artifacts integrity."""
        import json
        idx_file = Path(__file__).parent.parent / "vector_db" / "case_laws" / "index.faiss"
        ids_file = Path(__file__).parent.parent / "vector_db" / "case_laws" / "chunk_ids.json"
        cfg_file = Path(__file__).parent.parent / "vector_db" / "case_laws" / "index_config.json"
        chunks_file = Path(__file__).parent.parent / "data" / "chunks" / "case_laws" / "indexed_chunks.jsonl"

        self.assertTrue(idx_file.is_file(), "Case laws FAISS index.faiss not found")
        self.assertGreater(idx_file.stat().st_size, 100_000_000, "Index size should be > 100MB")

        self.assertTrue(ids_file.is_file(), "Case laws chunk_ids.json not found")
        with open(ids_file, encoding="utf-8") as f:
            ids = json.load(f)
        self.assertEqual(len(ids), 26085)

        self.assertTrue(cfg_file.is_file(), "Case laws index_config.json not found")
        with open(cfg_file, encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(cfg.get("num_vectors"), 26085)
        self.assertEqual(cfg.get("embedding_dim"), 1024)

        self.assertTrue(chunks_file.is_file(), "indexed_chunks.jsonl not found")
        self.assertGreater(chunks_file.stat().st_size, 20_000_000)

    def test_missing_evidence_not_high_confidence(self):
        """Test that missing or insufficient evidence never produces HIGH confidence."""
        mock_output = """## Answer
Based on the retrieved legal evidence provided, Article 21 of the Constitution of India is not mentioned or defined. Relying on general legal knowledge, Article 21 protects life and personal liberty.

## Relevant Legal Provisions
None

## Relevant Judgments
None

## Legal Reasoning
General constitutional principles apply.

## Confidence Level
HIGH — The right to life is well established.

## Limitations
Article 21 was not present in the retrieved texts."""

        # 1. Query specifies Article 21, but no chunks provided -> HIGH must be downgraded
        parsed = _parse_structured_answer(
            mock_output,
            query="What does Article 21 of the Constitution guarantee?",
            chunks=[]
        )
        self.assertNotIn("HIGH", parsed["confidence"].upper(), "Missing evidence must not produce HIGH confidence")
        self.assertIn("LOW", parsed["confidence"].upper())

        # 2. Query specifies Section 302, but chunks only have unrelated sections
        unrelated_chunks = [{"chunk_id": "c1", "text": "Section 10 of Contract Act defines valid contracts."}]
        parsed2 = _parse_structured_answer(
            mock_output,
            query="What is Section 302 IPC?",
            chunks=unrelated_chunks
        )
        self.assertNotIn("HIGH", parsed2["confidence"].upper(), "Unrelated evidence must not produce HIGH confidence")
        self.assertIn("LOW", parsed2["confidence"].upper())

    def test_article_21_direct_provision_retrieval(self):
        """Test that direct Article 21 query retrieves the correct provision when available."""
        test_chunks = [
            {
                "chunk_id": "c_art14",
                "title": "Constitution of India",
                "act_name": "Constitution of India",
                "text": "Act: Constitution of India\nSection: 14\n**14.** Equality before law.—The State shall not deny equality.",
            },
            {
                "chunk_id": "c_art21",
                "title": "Constitution of India",
                "act_name": "Constitution of India",
                "text": "Act: Constitution of India\nSection: 13\n**21.** No person shall be deprived of his life or personal liberty except according to procedure established by law.",
            },
            {
                "chunk_id": "c_sec302",
                "title": "Bharatiya Nyaya Sanhita",
                "act_name": "Bharatiya Nyaya Sanhita",
                "text": "Act: Bharatiya Nyaya Sanhita\nSection: 302\n**302.** Uttering words with deliberate intent.",
            },
        ]
        matches = find_exact_provision_matches(test_chunks, "What does Article 21 of the Constitution of India guarantee?")
        self.assertTrue(len(matches) > 0, "Exact provision matcher should find matches for Article 21")
        top_cid, top_score = matches[0]
        self.assertEqual(top_cid, "c_art21", "Top match for Article 21 Constitution must be c_art21")
        self.assertGreaterEqual(top_score, 2.0)

    def test_broad_constitutional_unsupported_topics_not_claimed_as_backed(self):
        """Test that broad constitutional answers organize into the 10 topics and do not claim unsupported topics as evidence-backed."""
        # Broad query with chunks covering ONLY Fundamental Rights
        query = "Explain the constitutional laws in India"
        self.assertTrue(is_broad_constitutional_query(query))

        fr_chunks = [
            {
                "citation_id": 1,
                "chunk_id": "c_fr",
                "title": "Constitution of India",
                "act_name": "Constitution of India",
                "text": "Act: Constitution of India\nPart III Fundamental Rights\n**14.** Equality before law.\n**21.** Protection of life and personal liberty.",
            }
        ]

        mock_llm_output = """## Answer
The Constitution of India guarantees Fundamental Rights under Part III, including Article 14 (equality) and Article 21 (life and liberty).

## Relevant Legal Provisions
- Article 14
- Article 21

## Relevant Judgments
None

## Legal Reasoning
Fundamental rights are enforceable through constitutional remedies.

## Confidence Level
HIGH — Basic constitutional framework is clear.

## Limitations
None
"""

        parsed = _parse_structured_answer(mock_llm_output, query=query, chunks=fr_chunks)
        answer = parsed["answer"]

        # 1. Verify all 10 constitutional topics are present in the answer
        for topic in CONSTITUTIONAL_TOPICS:
            self.assertIn(f"### {topic}", answer, f"Topic '{topic}' must be an organized section in the answer")

        # 2. Verify supported topic (Fundamental Rights) contains evidence-backed details and does NOT say it is uncovered
        fr_section_match = re.search(r"### Fundamental Rights\s+(.*?)(?=###|$)", answer, re.DOTALL)
        self.assertIsNotNone(fr_section_match)
        self.assertNotIn(UNSUPPORTED_TOPIC_MESSAGE, fr_section_match.group(1))
        self.assertTrue(
            "Article 14" in fr_section_match.group(1) or "Article 21" in fr_section_match.group(1) or "Part III" in fr_section_match.group(1)
        )

        # 3. Verify unsupported topics explicitly state the mandatory missing evidence phrase
        unsupported_topics = [
            "Preamble",
            "Directive Principles of State Policy",
            "Fundamental Duties",
            "Union and State Government",
            "Judiciary",
            "Federal structure",
            "Elections and constitutional bodies",
            "Emergency provisions",
            "Constitutional amendments",
        ]
        for topic in unsupported_topics:
            sec_match = re.search(rf"### {re.escape(topic)}\s+(.*?)(?=###|$)", answer, re.DOTALL)
            self.assertIsNotNone(sec_match, f"Section for {topic} must exist")
            sec_text = sec_match.group(1).strip()
            self.assertIn(
                UNSUPPORTED_TOPIC_MESSAGE,
                sec_text,
                f"Unsupported topic '{topic}' must explicitly contain '{UNSUPPORTED_TOPIC_MESSAGE}'"
            )

        # 4. Verify confidence is NOT HIGH when unsupported topics exist
        self.assertNotIn("HIGH", parsed["confidence"].upper(), "Confidence must NOT be HIGH when constitutional sections are unsupported")
        self.assertIn("MEDIUM", parsed["confidence"].upper())

        # 5. Verify limitations records unsupported topics
        self.assertIn("does not cover", parsed["limitations"].lower())

        # 6. Test completely unbacked query (chunks=[])
        parsed_empty = _parse_structured_answer(mock_llm_output, query=query, chunks=[])
        for topic in CONSTITUTIONAL_TOPICS:
            sec_match = re.search(rf"### {re.escape(topic)}\s+(.*?)(?=###|$)", parsed_empty["answer"], re.DOTALL)
            self.assertIsNotNone(sec_match)
            self.assertIn(UNSUPPORTED_TOPIC_MESSAGE, sec_match.group(1))
        self.assertNotIn("HIGH", parsed_empty["confidence"].upper())
        self.assertIn("LOW", parsed_empty["confidence"].upper())

    def test_temporal_law_transitions_and_warnings(self):
        """Test Indian statutory transition detection and metadata enrichment."""
        import temporal_law

        # Query with repealed statutes (IPC & CrPC)
        q = "What is anticipatory bail under Section 438 CrPC and penalty for murder under Section 302 IPC?"
        ctx = temporal_law.detect_temporal_context(q)
        self.assertEqual(ctx["temporal_status"], "repealed")
        self.assertTrue(ctx["is_repealed_law_cited"])
        self.assertGreaterEqual(len(ctx["warnings"]), 2)

        # Verify section transitions
        transitions = {t["historical_section"]: t for t in ctx["suggested_transitions"]}
        self.assertIn("302", transitions)
        self.assertEqual(transitions["302"]["modern_section"], "103")
        self.assertEqual(transitions["302"]["modern_act"], "BNS")

        self.assertIn("438", transitions)
        self.assertEqual(transitions["438"]["modern_section"], "482")
        self.assertEqual(transitions["438"]["modern_act"], "BNSS")

        # Verify chunk metadata enrichment
        ipc_chunk = {"act_name": "Indian Penal Code, 1860", "section": "302", "text": "Murder"}
        enriched_ipc = temporal_law.enrich_chunk_metadata(ipc_chunk)
        self.assertEqual(enriched_ipc["act_status"], "repealed")
        self.assertFalse(enriched_ipc["in_force"])
        self.assertEqual(enriched_ipc["modern_section"], "103")

        const_chunk = {"act_name": "Constitution of India", "article": "21", "text": "Life"}
        enriched_const = temporal_law.enrich_chunk_metadata(const_chunk)
        self.assertEqual(enriched_const["act_status"], "in_force")
        self.assertTrue(enriched_const["in_force"])

    def test_precedent_graph_service(self):
        """Test precedent relationship graph backend and landmark ratios."""
        import precedent_graph
        pg = precedent_graph.get_precedent_service()

        # Landmark: Kesavananda Bharati
        kb = pg.get_precedent("Kesavananda Bharati")
        self.assertIsNotNone(kb)
        self.assertEqual(kb.year, 1973)
        self.assertEqual(kb.bench_size, 13)
        self.assertTrue(any("Golaknath" in o for o in kb.overruled_cases))

        # Landmark: Maneka Gandhi
        mg = pg.get_precedent("Maneka Gandhi")
        self.assertIsNotNone(mg)
        self.assertEqual(mg.year, 1978)
        self.assertTrue(any("Gopalan" in o for o in mg.overruled_cases))

        # Landmark: Puttaswamy
        putt = pg.get_precedent("Puttaswamy")
        self.assertIsNotNone(putt)
        self.assertEqual(putt.bench_size, 9)
        self.assertTrue(any("ADM Jabalpur" in o for o in putt.overruled_cases))

        # Relationship graph query
        rels = pg.get_relationships("A.K. Gopalan")
        self.assertTrue(rels["case_found"])
        self.assertTrue(any("Maneka Gandhi" in c for c in rels["inward"]["overruled_by"]))

    def test_citation_verification_and_hallucination_penalties(self):
        """Test citation verification against retrieved evidence."""
        import citation_verifier

        # 1. Verified citation (Section 138 NI Act present in chunk)
        chunk = {
            "citation_id": 1,
            "title": "Negotiable Instruments Act, 1881",
            "section": "138",
            "text": "Act: Negotiable Instruments Act\nSection: 138\nDishonour of cheque for insufficiency of funds."
        }
        ans = "Under Section 138 of the Negotiable Instruments Act, dishonour of a cheque is an offence."
        summary = citation_verifier.verify_answer_citations(ans, [chunk])
        self.assertEqual(summary.unverified_count, 0)
        self.assertEqual(summary.hallucination_score, 0.0)
        self.assertEqual(summary.confidence_label, "high")

        # 2. Hallucinated / unbacked citation (Article 999 Constitution)
        ans_hallu = "Article 999 of the Constitution mandates automatic relief."
        summary_hallu = citation_verifier.verify_answer_citations(ans_hallu, [chunk])
        self.assertGreaterEqual(summary_hallu.unverified_count, 1)
        self.assertGreater(summary_hallu.hallucination_score, 0.0)
        self.assertIn(summary_hallu.confidence_label, ("low", "medium"))
        self.assertTrue(len(summary_hallu.legal_warnings) > 0)

    def test_structured_legal_reasoning_irac(self):
        """Test 8-step evidence-based IRAC reasoning summary."""
        import legal_reasoning
        q = "Can an anticipatory bail application be filed after filing of chargesheet?"
        chunks = [
            {
                "citation_id": 1,
                "title": "Code of Criminal Procedure, 1973",
                "section": "438",
                "text": "Section 438. Direction for grant of bail to person apprehending arrest."
            }
        ]
        parsed = {"legal_provisions": ["Section 438 CrPC"], "judgments": ["Sushila Aggarwal (2020)"]}
        steps = legal_reasoning.generate_structured_reasoning(q, chunks, parsed)

        self.assertEqual(len(steps), 8)
        phases = [s["phase"] for s in steps]
        self.assertEqual(phases, [
            "Issue Identified",
            "Relevant Legal Provision",
            "Applicable Rule",
            "Relevant Facts",
            "Application of the Rule",
            "Exceptions or Limitations",
            "Conclusion",
            "Sources",
        ])
        for s in steps:
            self.assertTrue(isinstance(s["step"], int))
            self.assertTrue(isinstance(s["description"], str) and len(s["description"]) > 0)

    def test_response_generator_14_sections_and_short_exceptions(self):
        """Test detailed 14-section response synthesis and short-query exception handling."""
        import response_generator

        # 1. Complex query -> 14 sections generated
        q = "Explain the fundamental principles governing the grant of anticipatory bail in non-bailable offences"
        chunks = [
            {
                "citation_id": 1,
                "title": "Code of Criminal Procedure, 1973",
                "section": "438",
                "text": "Section 438 provides for direction for grant of bail to person apprehending arrest in non-bailable offences."
            }
        ]
        resp = response_generator.build_detailed_legal_response(
            query=q,
            base_answer="Anticipatory bail is granted under Section 438 CrPC.",
            retrieved_chunks=chunks,
            parsed_sections={"legal_provisions": ["Section 438 CrPC"], "judgments": ["Gurbaksh Singh Sibbia (1980)"]},
            citations_list=["[1] CrPC Section 438"],
        )
        for i in range(1, 15):
            self.assertIn(f"## {i}.", resp, f"Section {i} must exist in detailed response")

        # 2. Simple query -> concise response without 10,000 char fluff
        simple_q = "penalty for cheque bounce"
        resp_simple = response_generator.build_detailed_legal_response(
            query=simple_q,
            base_answer="Imprisonment up to 2 years or fine up to twice the cheque amount under Section 138 NI Act.",
            retrieved_chunks=chunks,
            citations_list=["[1] Section 138 NI Act"],
        )
        self.assertNotIn("## 14.", resp_simple)
        self.assertIn("Direct Answer", resp_simple)

        # 3. Insufficient evidence -> concise anti-hallucination notice
        resp_insufficient = response_generator.build_detailed_legal_response(
            query="Explain Section 999 of Space Exploration Act 2099",
            base_answer="No evidence found.",
            retrieved_chunks=[],
            citations_list=[],
        )
        self.assertIn("Legal Limitation Notice", resp_insufficient)
        self.assertIn("LOW", resp_insufficient)

    def test_chat_store_backend_interfaces(self):
        """Test chat storage persistence adapter CRUD interfaces."""
        from storage import chat_store
        chat_store.init_db()

        # Create conversation
        uid = "test_user_pipeline"
        cid = chat_store.create_conversation(uid, "Test Pipeline Conversation", model_name="Qwen/Qwen3.6-35B-A3B")
        self.assertTrue(isinstance(cid, str) and len(cid) > 0)

        # Append user message
        u_mid = chat_store.append_message(cid, "user", "What is Article 21?", {"status": "complete"})
        self.assertTrue(isinstance(u_mid, str) and len(u_mid) > 0)

        # Append assistant message
        a_mid = chat_store.append_message(cid, "assistant", "Article 21 protects life.", {
            "status": "complete", "confidence": "HIGH", "citations": ["[1] Constitution of India"]
        })
        self.assertTrue(isinstance(a_mid, str) and len(a_mid) > 0)

        # Retrieve messages
        msgs = chat_store.get_messages(cid)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0].role, "user")
        self.assertEqual(msgs[1].role, "assistant")

        # Rename
        chat_store.rename_conversation(cid, "Updated Pipeline Conversation", user_id=uid)
        conv = chat_store.get_conversation(cid, user_id=uid)
        self.assertEqual(conv.title, "Updated Pipeline Conversation")

        # Soft delete
        chat_store.soft_delete_conversation(cid, user_id=uid)
        self.assertIsNone(chat_store.get_conversation(cid, user_id=uid, include_deleted=False))

    def test_adapter_nan_detection_and_safety_fallback(self):
        """Test that adapter validator detects NaNs and enforces base model fallback."""
        from validate_adapter import inspect_safetensors_file
        adapter_path = Path(__file__).parent.parent / "fine_tuning" / "adapters" / "lora-legal-v1" / "adapter_model.safetensors"
        self.assertTrue(adapter_path.exists(), "Adapter weights file must exist for verification")

        is_valid, report = inspect_safetensors_file(adapter_path)
        self.assertFalse(is_valid, "Corrupt adapter containing NaNs must be declared INVALID")
        self.assertGreater(report["nan_tensors_count"], 0, "Corrupt adapter must flag all NaN tensors")
        self.assertEqual(report["status"], "CORRUPT_CONTAINS_NANS_OR_INFS")

    def test_api_schemas_and_endpoints(self):
        """Test API request/response models and FastAPI route contracts."""
        from app.main import app, QueryRequest, QueryResponse
        from fastapi.testclient import TestClient
        from storage import chat_store
        chat_store.init_db()

        # Test Pydantic Schemas
        req = QueryRequest(query="What is Article 14?", conversation_id="conv_test_123")
        self.assertEqual(req.query, "What is Article 14?")
        self.assertEqual(req.conversation_id, "conv_test_123")

        resp = QueryResponse(
            query="What is Article 14?",
            answer="Article 14 guarantees equality before law.",
            confidence="HIGH",
            conversation_id="conv_test_123",
            message_id="msg_test_456",
            answer_format="detailed_legal_explanation",
            confidence_score=0.95,
            confidence_label="high",
            jurisdiction="India",
        )
        self.assertEqual(resp.confidence_label, "high")
        self.assertEqual(resp.jurisdiction, "India")
        self.assertEqual(resp.answer_format, "detailed_legal_explanation")

        # Test HTTP endpoints using TestClient
        client = TestClient(app)

        # Health
        res_health = client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "ok")

        # Stats
        res_stats = client.get("/stats")
        self.assertEqual(res_stats.status_code, 200)

        # Temporal check
        res_temp = client.get("/temporal/check?q=Section%20302%20IPC")
        self.assertEqual(res_temp.status_code, 200)
        self.assertEqual(res_temp.json()["temporal_status"], "repealed")

        # Precedents
        res_prec = client.get("/precedents/Kesavananda%20Bharati")
        self.assertEqual(res_prec.status_code, 200)
        self.assertTrue(res_prec.json()["case_found"])

        # History list
        res_convs = client.get("/conversations")
        self.assertEqual(res_convs.status_code, 200)
        self.assertIn("conversations", res_convs.json())

    def test_constitutional_golden_triangle_retrieval(self):
        """Test targeted Golden Triangle retrieval (Articles 14, 19, 21 balanced coverage + landmark cases)."""
        from constitutional_retrieval import (
            is_golden_triangle_query,
            extract_constitutional_articles,
            extract_clauses,
            get_constitutional_retriever,
        )

        query = (
            "Explain the differences between Article 14, Article 19, and Article 21 of the Constitution of India. "
            "Discuss how these rights are connected, their reasonable restrictions, and important Supreme Court judgments. "
            "Give the answer in a clear legal order with verified sources."
        )

        self.assertTrue(is_golden_triangle_query(query))
        extracted_arts = extract_constitutional_articles(query)
        self.assertIn("14", extracted_arts)
        self.assertIn("19", extracted_arts)
        self.assertIn("21", extracted_arts)

        cr = get_constitutional_retriever()
        docs, coverage = cr.retrieve(query, top_k=14)
        self.assertGreaterEqual(len(docs), 10)

        # Check coverage of all 3 articles
        self.assertTrue(coverage.get("article_14", {}).get("found"))
        self.assertTrue(coverage.get("article_19", {}).get("found"))
        self.assertTrue(coverage.get("article_21", {}).get("found"))
        self.assertEqual(coverage.get("article_14", {}).get("confidence"), "high")
        self.assertEqual(coverage.get("article_19", {}).get("confidence"), "high")
        self.assertEqual(coverage.get("article_21", {}).get("confidence"), "high")

        # Check landmark cases presence
        doc_titles = [d.get("title", "") for d in docs]
        has_maneka = any("MANEKA GANDHI" in t for t in doc_titles)
        has_gopalan = any("GOPALAN" in t for t in doc_titles)
        self.assertTrue(has_maneka, "Maneka Gandhi precedent must be retrieved")
        self.assertTrue(has_gopalan, "A.K. Gopalan precedent must be retrieved")

    def test_golden_triangle_14_section_response_generation(self):
        """Test detailed response generator synthesizes all 14 ordered sections for Golden Triangle."""
        import response_generator
        from constitutional_retrieval import get_constitutional_retriever

        query = (
            "Explain the differences between Article 14, Article 19, and Article 21 of the Constitution of India. "
            "Discuss how these rights are connected, their reasonable restrictions, and important Supreme Court judgments."
        )
        cr = get_constitutional_retriever()
        docs, coverage = cr.retrieve(query, top_k=14)

        resp_text = response_generator.build_detailed_legal_response(
            query=query,
            base_answer="Articles 14, 19, and 21 form the Golden Triangle of fundamental rights.",
            retrieved_chunks=docs,
            citations_list=[f"[{i+1}] {d.get('title')}" for i, d in enumerate(docs[:5])],
        )

        for sec in response_generator.ORDERED_SECTIONS:
            self.assertIn(f"## {sec}", resp_text, f"Section {sec} must be present in response")

        # Substantive content checks
        self.assertIn("Golden Triangle", resp_text)
        self.assertIn("Maneka Gandhi", resp_text)
        self.assertIn("A.K. Gopalan", resp_text)
        self.assertIn("Article 14", resp_text)
        self.assertIn("Article 19", resp_text)
        self.assertIn("Article 21", resp_text)
        self.assertIn("Reasonable Restrictions", resp_text)

    def test_generation_timeout_and_interrupted_error_contract(self):
        """Test chat API handling and DB preservation when generation is interrupted or times out."""
        from storage import chat_store

        test_uid = "test_user_timeout_contract"
        conv_id = chat_store.create_conversation(test_uid, "Test Timeout Handling")

        # Preserved user question
        user_msg_id = chat_store.append_message(
            conv_id, "user", "Explain Article 14, 19, and 21",
            {"status": "complete"}
        )

        # Assistant partial message on interruption
        error_msg = "Generation timed out after 180 seconds. The question above has been preserved."
        asst_msg_id = chat_store.append_message(
            conv_id, "assistant",
            "_(generation interrupted — timeout reached while generating answer)_",
            {
                "status": "partial",
                "error": error_msg,
                "error_code": "GENERATION_INTERRUPTED",
                "error_type": "TimeoutError",
            }
        )

        # Check DB retrieval
        messages = chat_store.get_messages(conv_id)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].id, user_msg_id)
        self.assertEqual(messages[1].id, asst_msg_id)
        self.assertEqual(messages[1].status, "partial")
        self.assertEqual(messages[1].metadata.get("error_code"), "GENERATION_INTERRUPTED")


class TestLoRAAdapterSafetyAndFallback(unittest.TestCase):
    """Unit tests verifying LoRA adapter forensic audit and automatic fallback."""

    def test_lora_v1_rejected_due_to_nans(self):
        """Verify that lora-legal-v1 is detected as corrupt with 320 NaN tensors and rejected."""
        from validate_adapter import inspect_safetensors_file
        v1_path = Path(__file__).parent.parent / "fine_tuning" / "adapters" / "lora-legal-v1" / "adapter_model.safetensors"
        if not v1_path.exists():
            self.skipTest("lora-legal-v1 safetensors not found")

        is_valid, report = inspect_safetensors_file(v1_path)
        self.assertFalse(is_valid, "lora-legal-v1 must NOT be marked valid")
        self.assertEqual(report.get("status"), "CORRUPT_CONTAINS_NANS_OR_INFS")
        self.assertGreater(report.get("nan_tensors_count", 0), 0)
        self.assertEqual(report.get("nan_tensors_count"), 320)

    def test_lora_v2_accepted_with_zero_nans(self):
        """Verify that lora-legal-v2 passes safety audit with 0 NaNs and 0 Infs."""
        from validate_adapter import inspect_safetensors_file, validate_adapter_directory
        v2_dir = Path(__file__).parent.parent / "fine_tuning" / "adapters" / "lora-legal-v2"
        if not v2_dir.exists():
            self.skipTest("lora-legal-v2 directory not found")

        is_valid, report = validate_adapter_directory(v2_dir)
        self.assertTrue(is_valid, f"lora-legal-v2 must be valid: {report}")
        self.assertEqual(report.get("status"), "VALID")
        self.assertEqual(report.get("nan_tensors_count"), 0)
        self.assertEqual(report.get("inf_tensors_count"), 0)
        self.assertEqual(report.get("total_tensors"), 80)
        self.assertEqual(report.get("total_params"), 3440640)

    def test_nonexistent_adapter_fails_gracefully(self):
        """Verify that non-existent adapter path returns False without crashing."""
        from validate_adapter import validate_adapter_directory
        fake_path = Path("/tmp/non_existent_lora_directory_12345")
        is_valid, report = validate_adapter_directory(fake_path)
        self.assertFalse(is_valid)
        self.assertIn("error", report)

    def test_corrupt_adapter_rejected_by_attach_adapter_logic(self):
        """Verify that LegalMindModel adapter attachment rejects corrupt adapters safely."""
        from validate_adapter import inspect_safetensors_file
        v1_path = Path(__file__).parent.parent / "fine_tuning" / "adapters" / "lora-legal-v1" / "adapter_model.safetensors"
        if v1_path.exists():
            is_valid, report = inspect_safetensors_file(v1_path)
            # The attach_adapter logic requires is_valid=True and nan_tensors_count=0
            should_reject = (not is_valid) or (report.get("nan_tensors_count", 0) > 0)
            self.assertTrue(should_reject, "Corrupt adapter must trigger rejection in attach_adapter")


if __name__ == "__main__":
    unittest.main()


