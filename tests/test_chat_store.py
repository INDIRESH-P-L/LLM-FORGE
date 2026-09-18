#!/usr/bin/env python3
"""
tests/test_chat_store.py
========================
Unit tests for storage/chat_store.py and storage/exporters.py.

Runs against a real temp database file (not :memory:) — WAL mode, the
busy-timeout behaviour and cross-process locking are precisely what needs
testing, and none of them exist for an in-memory database.

    python -m unittest tests.test_chat_store -v     # from the project root
    python tests/test_chat_store.py                 # equivalent

stdlib unittest, matching tests/test_pipeline.py; pytest is not a dependency
of this project.
"""

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from storage import chat_store as store
from storage import exporters

SAMPLE_CITATIONS = [
    {
        "case_name": "Vijay Madanlal Choudhary v. Union of India",
        "citation": "(2022) SCC OnLine SC 929",
        "court": "Supreme Court of India",
        "date": "2022-07-27",
        "judges": None,
        "paragraph_no": None,
        "source_id": "sc_2022_pmla_0001",
        "excerpt": "The twin conditions under Section 45 of the PMLA are mandatory…",
    },
    {
        "case_name": "Nikesh Tarachand Shah v. Union of India",
        "citation": "(2018) 11 SCC 1",
        "court": "Supreme Court of India",
        "date": "2017-11-23",
        "judges": None,
        "paragraph_no": None,
        "source_id": "sc_2017_pmla_0002",
        "excerpt": "Section 45 as it then stood was held to be unconstitutional…",
    },
]


class ChatStoreTestCase(unittest.TestCase):
    """Base: a throwaway database per test, via LEGALMIND_DB_PATH."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="legalmind-test-")
        self.db_file = Path(self.tmpdir) / "nested" / "legalmind-test.db"
        self._prev_env = os.environ.get("LEGALMIND_DB_PATH")
        os.environ["LEGALMIND_DB_PATH"] = str(self.db_file)
        store.close_connection()
        store._initialised.discard(str(self.db_file))
        store.init_db()

    def tearDown(self):
        store.close_connection()
        store._initialised.discard(str(self.db_file))
        if self._prev_env is None:
            os.environ.pop("LEGALMIND_DB_PATH", None)
        else:
            os.environ["LEGALMIND_DB_PATH"] = self._prev_env
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # helpers ---------------------------------------------------------------

    def _exchange(self, user_id="u1", question="What is anticipatory bail?",
                  answer="Anticipatory bail is governed by Section 438 CrPC.",
                  citations=None):
        cid = store.create_conversation(user_id, question, model_name="Qwen3.6-35B-A3B")
        store.append_message(cid, "user", question)
        store.append_message(cid, "assistant", answer, {
            "citations": citations if citations is not None else SAMPLE_CITATIONS,
            "latency_ms": 4210,
            "status": "complete",
        })
        return cid


# ---------------------------------------------------------------------------
# Schema and migrations
# ---------------------------------------------------------------------------

class TestSchema(ChatStoreTestCase):

    def test_db_file_and_parent_directory_are_created(self):
        self.assertTrue(self.db_file.exists(), "init_db must create the database file")
        self.assertTrue(self.db_file.parent.is_dir(), "init_db must create missing parent dirs")

    def test_init_db_is_idempotent(self):
        first = store.init_db()
        second = store.init_db(force=True)
        third = store.init_db(force=True)
        self.assertEqual(first, second)
        self.assertEqual(second, third)
        self.assertEqual(first, max(v for v, _ in store.MIGRATIONS))

    def test_schema_version_recorded_once_per_migration(self):
        store.init_db(force=True)
        rows = store.get_connection().execute(
            "SELECT version FROM schema_version ORDER BY version").fetchall()
        self.assertEqual([r["version"] for r in rows], [v for v, _ in store.MIGRATIONS])

    def test_required_pragmas_are_set_on_every_connection(self):
        conn = store.get_connection()
        self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
        self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        self.assertEqual(conn.execute("PRAGMA synchronous").fetchone()[0], 1)  # NORMAL
        self.assertGreaterEqual(conn.execute("PRAGMA busy_timeout").fetchone()[0], 5000)

    def test_expected_indexes_exist(self):
        names = {r["name"] for r in store.get_connection().execute(
            "SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
        self.assertIn("idx_messages_conv_created", names)
        self.assertIn("idx_conversations_user_updated", names)
        self.assertIn("idx_messages_client_token", names)

    def test_fts_table_exists(self):
        row = store.get_connection().execute(
            "SELECT name FROM sqlite_master WHERE name='messages_fts'").fetchone()
        self.assertIsNotNone(row, "FTS5 virtual table must be created")

    def test_role_and_status_constraints_are_enforced(self):
        cid = store.create_conversation("u1", "t")
        with self.assertRaises(ValueError):
            store.append_message(cid, "moderator", "nope")
        with self.assertRaises(ValueError):
            store.append_message(cid, "user", "hi", {"status": "halfway"})

    def test_foreign_key_cascade_removes_messages(self):
        cid = self._exchange()
        self.assertEqual(len(store.get_messages(cid)), 2)
        store.purge_conversation(cid)
        self.assertEqual(store.get_messages(cid), [])
        self.assertEqual(store.count_messages(), 0)


# ---------------------------------------------------------------------------
# Conversations and messages
# ---------------------------------------------------------------------------

class TestConversations(ChatStoreTestCase):

    def test_conversation_id_is_a_uuid4_not_an_integer(self):
        cid = store.create_conversation("u1", "Bail under PMLA")
        self.assertRegex(cid, r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-")

    def test_create_requires_user_id(self):
        with self.assertRaises(ValueError):
            store.create_conversation("", "title")

    def test_title_is_cleaned_and_truncated_to_60_chars(self):
        long_q = ("  Whether   the twin conditions under Section 45 of the "
                  "Prevention of Money Laundering Act 2002 survive Nikesh Shah?  ")
        cid = store.create_conversation("u1", long_q)
        title = store.get_conversation(cid).title
        self.assertLessEqual(len(title), store.TITLE_MAX_CHARS + 1)  # +1 for the ellipsis
        self.assertTrue(title.startswith("Whether the twin conditions"))
        self.assertNotIn("  ", title)

    def test_make_title_never_returns_empty(self):
        self.assertEqual(store.make_title("   "), "Untitled conversation")
        self.assertEqual(store.make_title("### "), "Untitled conversation")

    def test_messages_come_back_in_order(self):
        cid = store.create_conversation("u1", "t")
        expected = []
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            text = f"message {i}"
            store.append_message(cid, role, text)
            expected.append((role, text))
        got = [(m.role, m.content) for m in store.get_messages(cid)]
        self.assertEqual(got, expected)

    def test_append_bumps_updated_at_in_the_same_write(self):
        cid = store.create_conversation("u1", "t")
        before = store.get_conversation(cid)
        store.append_message(cid, "user", "does updated_at move?")
        after = store.get_conversation(cid)
        self.assertGreater(after.updated_at, before.updated_at)
        # The bump must be atomic with the insert: the newest message's
        # timestamp is exactly the conversation's updated_at.
        newest = store.get_messages(cid)[-1]
        self.assertEqual(after.updated_at, newest.created_at)

    def test_citations_round_trip_as_structured_json(self):
        cid = self._exchange()
        answer = store.get_messages(cid)[-1]
        self.assertEqual(len(answer.citations), 2)
        self.assertEqual(answer.citations[0]["case_name"],
                         "Vijay Madanlal Choudhary v. Union of India")
        self.assertEqual(answer.citations[0]["court"], "Supreme Court of India")
        # Fields the corpus does not carry are preserved as null, not dropped.
        self.assertIn("judges", answer.citations[0])
        self.assertIsNone(answer.citations[0]["judges"])
        self.assertIsNone(answer.citations[0]["paragraph_no"])

    def test_citations_column_accepts_null_when_pipeline_gives_none(self):
        cid = store.create_conversation("u1", "t")
        mid = store.append_message(cid, "assistant", "answer with no retrieval")
        raw = store.get_connection().execute(
            "SELECT citations FROM messages WHERE id = ?", (mid,)).fetchone()
        self.assertIsNone(raw["citations"])
        self.assertEqual(store.get_message(mid).citations, [])

    def test_metadata_is_split_between_typed_columns_and_json(self):
        cid = store.create_conversation("u1", "t")
        mid = store.append_message(cid, "assistant", "text", {
            "token_count": 812,
            "latency_ms": 5300,
            "status": "partial",
            "feedback": 1,
            "confidence": "HIGH",
            "error": "socket closed",
        })
        msg = store.get_message(mid)
        self.assertEqual(msg.token_count, 812)
        self.assertEqual(msg.latency_ms, 5300)
        self.assertEqual(msg.status, "partial")
        self.assertEqual(msg.feedback, 1)
        # Un-promoted keys land in the JSON metadata blob.
        self.assertEqual(msg.metadata["confidence"], "HIGH")
        self.assertEqual(msg.metadata["error"], "socket closed")
        self.assertNotIn("token_count", msg.metadata)

    def test_status_defaults_to_complete(self):
        cid = store.create_conversation("u1", "t")
        mid = store.append_message(cid, "user", "hello")
        self.assertEqual(store.get_message(mid).status, "complete")

    def test_error_and_partial_rows_are_storable_and_readable(self):
        cid = store.create_conversation("u1", "t")
        store.append_message(cid, "user", "long question")
        store.append_message(cid, "assistant", "half an ans",
                             {"status": "partial", "error": "client disconnected"})
        store.append_message(cid, "user", "another")
        store.append_message(cid, "assistant", "", {"status": "error", "error": "CUDA OOM"})
        statuses = [m.status for m in store.get_messages(cid)]
        self.assertEqual(statuses, ["complete", "partial", "complete", "error"])

    def test_update_message_finalises_a_row(self):
        cid = store.create_conversation("u1", "t")
        mid = store.append_message(cid, "assistant", "partial text", {"status": "partial"})
        self.assertTrue(store.update_message(mid, content="full text", status="complete",
                                             citations=SAMPLE_CITATIONS, latency_ms=900))
        msg = store.get_message(mid)
        self.assertEqual(msg.content, "full text")
        self.assertEqual(msg.status, "complete")
        self.assertEqual(len(msg.citations), 2)
        self.assertEqual(msg.latency_ms, 900)

    def test_feedback_can_be_recorded_and_is_owner_scoped(self):
        cid = self._exchange(user_id="owner")
        mid = store.get_messages(cid)[-1].id
        self.assertTrue(store.set_feedback(mid, -1, user_id="owner"))
        self.assertEqual(store.get_message(mid).feedback, -1)
        self.assertFalse(store.set_feedback(mid, 1, user_id="someone-else"))
        self.assertEqual(store.get_message(mid).feedback, -1)
        with self.assertRaises(ValueError):
            store.set_feedback(mid, 5)

    def test_count_messages_matches_raw_sql(self):
        self._exchange(user_id="a")
        self._exchange(user_id="b")
        raw = store.get_connection().execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
        self.assertEqual(store.count_messages(), raw)
        self.assertEqual(store.count_messages("a"), 2)
        self.assertEqual(store.count_messages("b"), 2)


# ---------------------------------------------------------------------------
# Idempotency — the duplicate-row guard
# ---------------------------------------------------------------------------

class TestIdempotency(ChatStoreTestCase):

    def test_same_client_token_does_not_create_a_second_row(self):
        cid = store.create_conversation("u1", "t")
        first = store.append_message(cid, "user", "Is bail available?",
                                     {"client_token": "submit-abc"})
        second = store.append_message(cid, "user", "Is bail available?",
                                      {"client_token": "submit-abc"})
        self.assertEqual(first, second)
        self.assertEqual(len(store.get_messages(cid)), 1)

    def test_distinct_tokens_create_distinct_rows(self):
        cid = store.create_conversation("u1", "t")
        store.append_message(cid, "user", "q1", {"client_token": "t1"})
        store.append_message(cid, "user", "q1", {"client_token": "t2"})
        self.assertEqual(len(store.get_messages(cid)), 2)

    def test_messages_without_a_token_are_never_deduplicated(self):
        cid = store.create_conversation("u1", "t")
        store.append_message(cid, "user", "same text")
        store.append_message(cid, "user", "same text")
        self.assertEqual(len(store.get_messages(cid)), 2)

    def test_unique_index_blocks_a_duplicate_token_at_the_sql_level(self):
        cid = store.create_conversation("u1", "t")
        store.append_message(cid, "user", "q", {"client_token": "dup"})
        with self.assertRaises(sqlite3.IntegrityError):
            with store.transaction() as conn:
                conn.execute(
                    "INSERT INTO messages (id, conversation_id, role, content,"
                    " created_at, status, client_token) VALUES (?,?,?,?,?,?,?)",
                    (store.new_id(), cid, "user", "q", store.utcnow(), "complete", "dup"),
                )

    def test_token_lookup_returns_the_stored_message(self):
        cid = store.create_conversation("u1", "t")
        mid = store.append_message(cid, "user", "q", {"client_token": "look-me-up"})
        found = store.get_message_by_client_token("look-me-up")
        self.assertIsNotNone(found)
        self.assertEqual(found.id, mid)
        self.assertIsNone(store.get_message_by_client_token("never-used"))

    def test_get_next_message_finds_the_answer_to_a_question(self):
        cid = store.create_conversation("u1", "t")
        store.append_message(cid, "user", "q", {"client_token": "tok"})
        store.append_message(cid, "assistant", "a")
        question = store.get_message_by_client_token("tok")
        answer = store.get_next_message(cid, question)
        self.assertIsNotNone(answer)
        self.assertEqual(answer.content, "a")

    def test_get_next_message_returns_none_for_an_unanswered_question(self):
        cid = store.create_conversation("u1", "t")
        store.append_message(cid, "user", "q", {"client_token": "tok"})
        question = store.get_message_by_client_token("tok")
        self.assertIsNone(store.get_next_message(cid, question))


# ---------------------------------------------------------------------------
# Sidebar: listing, rename, pin, soft delete, undo
# ---------------------------------------------------------------------------

class TestSidebarOperations(ChatStoreTestCase):

    def test_list_is_newest_updated_first(self):
        a = store.create_conversation("u1", "first")
        b = store.create_conversation("u1", "second")
        c = store.create_conversation("u1", "third")
        store.append_message(a, "user", "touch a")   # a becomes most recent
        order = [x.id for x in store.list_conversations("u1")]
        self.assertEqual(order[0], a)
        self.assertEqual(set(order), {a, b, c})

    def test_pinned_conversations_sort_above_everything(self):
        old = store.create_conversation("u1", "old but pinned")
        new = store.create_conversation("u1", "new and unpinned")
        store.append_message(new, "user", "touch")
        store.set_pinned(old, True)
        listed = store.list_conversations("u1")
        self.assertEqual(listed[0].id, old)
        self.assertTrue(listed[0].pinned)
        self.assertFalse(listed[1].pinned)

    def test_unpinning_restores_recency_order(self):
        old = store.create_conversation("u1", "old")
        new = store.create_conversation("u1", "new")
        store.append_message(new, "user", "touch")
        store.set_pinned(old, True)
        store.set_pinned(old, False)
        self.assertEqual(store.list_conversations("u1")[0].id, new)

    def test_rename_persists_and_rejects_blank(self):
        cid = store.create_conversation("u1", "original")
        self.assertTrue(store.rename_conversation(cid, "  Section 138 NI Act notice  "))
        self.assertEqual(store.get_conversation(cid).title, "Section 138 NI Act notice")
        with self.assertRaises(ValueError):
            store.rename_conversation(cid, "   ")

    def test_soft_delete_hides_but_keeps_rows(self):
        cid = self._exchange()
        store.soft_delete_conversation(cid)
        self.assertEqual(store.list_conversations("u1"), [])
        self.assertEqual(len(store.list_conversations("u1", include_deleted=True)), 1)
        # The messages are untouched — that is what makes undo possible.
        self.assertEqual(len(store.get_messages(cid)), 2)
        self.assertIsNotNone(store.get_conversation(cid).deleted_at)

    def test_restore_is_the_undo(self):
        cid = self._exchange()
        store.soft_delete_conversation(cid)
        self.assertTrue(store.restore_conversation(cid))
        listed = store.list_conversations("u1")
        self.assertEqual(len(listed), 1)
        self.assertIsNone(listed[0].deleted_at)
        self.assertEqual(len(store.get_messages(cid)), 2)

    def test_summary_carries_count_and_preview(self):
        cid = self._exchange()
        summary = store.list_conversations("u1")[0]
        self.assertEqual(summary.message_count, 2)
        self.assertIn("Section 438", summary.preview)
        self.assertEqual(summary.model_name, "Qwen3.6-35B-A3B")

    def test_limit_and_offset_page_through(self):
        ids = []
        for i in range(5):
            cid = store.create_conversation("u1", f"conv {i}")
            store.append_message(cid, "user", f"touch {i}")
            ids.append(cid)
        page1 = store.list_conversations("u1", limit=2, offset=0)
        page2 = store.list_conversations("u1", limit=2, offset=2)
        page3 = store.list_conversations("u1", limit=2, offset=4)
        self.assertEqual([len(page1), len(page2), len(page3)], [2, 2, 1])
        seen = [c.id for c in page1 + page2 + page3]
        self.assertEqual(len(set(seen)), 5)


# ---------------------------------------------------------------------------
# Per-user isolation — no accounts yet, but no leaks either
# ---------------------------------------------------------------------------

class TestUserIsolation(ChatStoreTestCase):

    def test_users_only_see_their_own_conversations(self):
        mine = self._exchange(user_id="browser-a", question="My PMLA question")
        theirs = self._exchange(user_id="browser-b", question="Their tenancy question")
        a_list = store.list_conversations("browser-a")
        b_list = store.list_conversations("browser-b")
        self.assertEqual([c.id for c in a_list], [mine])
        self.assertEqual([c.id for c in b_list], [theirs])

    def test_a_fresh_user_sees_an_empty_history(self):
        self._exchange(user_id="browser-a")
        self.assertEqual(store.list_conversations("browser-brand-new"), [])

    def test_owner_scoped_lookup_hides_another_users_conversation(self):
        cid = self._exchange(user_id="browser-a")
        self.assertIsNotNone(store.get_conversation(cid, user_id="browser-a"))
        self.assertIsNone(store.get_conversation(cid, user_id="browser-b"))

    def test_one_user_cannot_mutate_anothers_conversation(self):
        cid = self._exchange(user_id="browser-a")
        self.assertFalse(store.rename_conversation(cid, "hijacked", user_id="browser-b"))
        self.assertFalse(store.soft_delete_conversation(cid, user_id="browser-b"))
        self.assertFalse(store.set_pinned(cid, True, user_id="browser-b"))
        conv = store.get_conversation(cid)
        self.assertNotEqual(conv.title, "hijacked")
        self.assertIsNone(conv.deleted_at)
        self.assertFalse(conv.pinned)

    def test_search_never_crosses_users(self):
        self._exchange(user_id="browser-a", question="anticipatory bail in Rajasthan")
        self._exchange(user_id="browser-b", question="anticipatory bail in Kerala")
        hits = store.search_messages("browser-a", "anticipatory bail")
        self.assertTrue(hits)
        for hit in hits:
            conv = store.get_conversation(hit.conversation_id)
            self.assertEqual(conv.user_id, "browser-a")


# ---------------------------------------------------------------------------
# History search (FTS5)
# ---------------------------------------------------------------------------

class TestSearch(ChatStoreTestCase):

    def setUp(self):
        super().setUp()
        self.c1 = self._exchange(
            user_id="u1",
            question="Can bail be denied under PMLA Section 45?",
            answer="The twin conditions in Section 45 PMLA are mandatory after Vijay Madanlal.")
        self.c2 = self._exchange(
            user_id="u1",
            question="What is the limitation period for a Section 138 NI Act complaint?",
            answer="One month from the date the cause of action arises.")

    def test_finds_terms_in_questions_and_answers(self):
        self.assertTrue(any(h.conversation_id == self.c1
                            for h in store.search_messages("u1", "PMLA")))
        self.assertTrue(any(h.conversation_id == self.c1
                            for h in store.search_messages("u1", "twin conditions")))
        self.assertTrue(any(h.conversation_id == self.c2
                            for h in store.search_messages("u1", "limitation")))

    def test_prefix_matching_works_for_partial_typing(self):
        self.assertTrue(store.search_messages("u1", "limita"))

    def test_snippet_marks_the_match(self):
        hits = store.search_messages("u1", "PMLA")
        self.assertTrue(hits)
        self.assertTrue(any("<mark>" in h.snippet for h in hits))

    def test_no_match_returns_empty(self):
        self.assertEqual(store.search_messages("u1", "zzzznonexistentterm"), [])

    def test_empty_or_punctuation_only_query_returns_empty(self):
        for q in ("", "   ", "***", "-", '"'):
            self.assertEqual(store.search_messages("u1", q), [], f"query {q!r}")

    def test_fts_operator_characters_do_not_raise(self):
        # Raw input reaching MATCH unescaped would be a syntax error, not a miss.
        for q in ('bail AND OR NOT', 'Section 45 * "', 'bail -PMLA', 'NEAR(a b)', "it's"):
            try:
                store.search_messages("u1", q)
            except Exception as e:
                self.fail(f"search_messages raised on {q!r}: {e}")

    def test_index_follows_edits_and_deletes(self):
        cid = store.create_conversation("u1", "t")
        mid = store.append_message(cid, "user", "mandamus against a tribunal")
        self.assertTrue(store.search_messages("u1", "mandamus"))
        store.update_message(mid, content="certiorari against a tribunal")
        self.assertEqual(store.search_messages("u1", "mandamus"), [])
        self.assertTrue(store.search_messages("u1", "certiorari"))
        store.purge_conversation(cid)
        self.assertEqual(store.search_messages("u1", "certiorari"), [])

    def test_soft_deleted_conversations_drop_out_of_search(self):
        self.assertTrue(store.search_messages("u1", "PMLA"))
        store.soft_delete_conversation(self.c1)
        self.assertEqual(store.search_messages("u1", "PMLA"), [])
        store.restore_conversation(self.c1)
        self.assertTrue(store.search_messages("u1", "PMLA"))

    def test_rebuild_index_recovers_search(self):
        store.get_connection().execute("INSERT INTO messages_fts(messages_fts) VALUES ('delete-all')")
        store.rebuild_search_index()
        self.assertTrue(store.search_messages("u1", "PMLA"))


# ---------------------------------------------------------------------------
# Interrupted exchanges
# ---------------------------------------------------------------------------

class TestReconciliation(ChatStoreTestCase):

    def test_unanswered_question_becomes_a_partial_row(self):
        cid = store.create_conversation("u1", "killed mid-answer")
        store.append_message(cid, "user", "Explain the doctrine of basic structure.")
        self.assertEqual(store.reconcile_interrupted(), 1)
        messages = store.get_messages(cid)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1].role, "assistant")
        self.assertEqual(messages[1].status, "partial")
        self.assertEqual(messages[1].metadata["interrupted_user_message_id"], messages[0].id)
        self.assertIn("error", messages[1].metadata)

    def test_completed_exchanges_are_left_alone(self):
        cid = self._exchange()
        self.assertEqual(store.reconcile_interrupted(), 0)
        self.assertEqual(len(store.get_messages(cid)), 2)

    def test_reconcile_is_idempotent_across_restarts(self):
        cid = store.create_conversation("u1", "t")
        store.append_message(cid, "user", "q")
        self.assertEqual(store.reconcile_interrupted(), 1)
        self.assertEqual(store.reconcile_interrupted(), 0)   # second startup
        self.assertEqual(store.reconcile_interrupted(), 0)   # third
        self.assertEqual(len(store.get_messages(cid)), 2)

    def test_only_the_trailing_question_is_reconciled(self):
        cid = store.create_conversation("u1", "t")
        store.append_message(cid, "user", "q1")
        store.append_message(cid, "assistant", "a1")
        store.append_message(cid, "user", "q2")          # died here
        self.assertEqual(store.reconcile_interrupted(), 1)
        statuses = [(m.role, m.status) for m in store.get_messages(cid)]
        self.assertEqual(statuses, [("user", "complete"), ("assistant", "complete"),
                                    ("user", "complete"), ("assistant", "partial")])

    def test_reconciles_across_several_conversations(self):
        for i in range(3):
            cid = store.create_conversation("u1", f"conv {i}")
            store.append_message(cid, "user", f"q{i}")
        self.assertEqual(store.reconcile_interrupted(), 3)


# ---------------------------------------------------------------------------
# Concurrency — "database is locked" must never surface
# ---------------------------------------------------------------------------

class TestConcurrency(ChatStoreTestCase):

    def test_concurrent_writers_in_many_threads(self):
        """Simulates several browsers submitting at the same moment."""
        n_threads, per_thread = 8, 25
        cids = [store.create_conversation(f"user-{i}", f"conv {i}") for i in range(n_threads)]
        errors: list[BaseException] = []
        start = threading.Barrier(n_threads)

        def worker(idx: int):
            try:
                start.wait(timeout=10)
                for j in range(per_thread):
                    store.append_message(cids[idx], "user", f"thread {idx} message {j}",
                                         {"client_token": f"t-{idx}-{j}"})
            except BaseException as e:      # noqa: BLE001 — the test is the assertion
                errors.append(e)
            finally:
                store.close_connection()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        self.assertEqual(errors, [], f"concurrent writes raised: {errors[:3]}")
        self.assertEqual(store.count_messages(), n_threads * per_thread)
        for i, cid in enumerate(cids):
            self.assertEqual(len(store.get_messages(cid)), per_thread, f"conversation {i}")

    def test_concurrent_readers_while_writing(self):
        """WAL means a long read must not block the writer, or vice versa."""
        cid = store.create_conversation("u1", "reader/writer")
        errors: list[BaseException] = []
        stop = threading.Event()

        def reader():
            try:
                while not stop.is_set():
                    store.get_messages(cid)
                    store.list_conversations("u1")
                    store.search_messages("u1", "concurrent")
            except BaseException as e:      # noqa: BLE001
                errors.append(e)
            finally:
                store.close_connection()

        readers = [threading.Thread(target=reader) for _ in range(4)]
        for t in readers:
            t.start()
        try:
            for i in range(100):
                store.append_message(cid, "user", f"concurrent write {i}")
        finally:
            stop.set()
            for t in readers:
                t.join(timeout=30)

        self.assertEqual(errors, [], f"concurrent reads raised: {errors[:3]}")
        self.assertEqual(len(store.get_messages(cid)), 100)

    def test_concurrent_writers_in_separate_processes(self):
        """
        Two OS processes on the same file — the real cross-process lock test,
        and the case WAL + busy_timeout exists for. This is also what happens
        when the sqlite3 CLI is used against a live server's database.
        """
        n_procs, per_proc = 4, 20
        cids = [store.create_conversation(f"proc-user-{i}", f"proc conv {i}")
                for i in range(n_procs)]

        script = (
            "import os, sys\n"
            f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
            "from storage import chat_store as store\n"
            "cid, n, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]\n"
            "store.init_db()\n"
            "for j in range(n):\n"
            "    store.append_message(cid, 'user', f'{tag} message {j}',\n"
            "                         {'client_token': f'{tag}-{j}'})\n"
            "print('ok')\n"
        )
        env = dict(os.environ, LEGALMIND_DB_PATH=str(self.db_file))
        procs = [
            subprocess.Popen([sys.executable, "-c", script, cids[i], str(per_proc), f"p{i}"],
                             env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for i in range(n_procs)
        ]
        failures = []
        for i, proc in enumerate(procs):
            out, err = proc.communicate(timeout=120)
            if proc.returncode != 0:
                failures.append(f"process {i} exited {proc.returncode}: {err.strip()[-400:]}")
            self.assertNotIn("database is locked", (err or "").lower(),
                             f"process {i} hit a lock error:\n{err}")
        self.assertEqual(failures, [], "\n".join(failures))

        for i, cid in enumerate(cids):
            self.assertEqual(len(store.get_messages(cid)), per_proc, f"process {i}'s rows")
        self.assertEqual(store.count_messages(), n_procs * per_proc)

    def test_transaction_rolls_back_completely_on_failure(self):
        cid = store.create_conversation("u1", "t")
        before = store.count_messages()
        with self.assertRaises(RuntimeError):
            with store.transaction() as conn:
                conn.execute(
                    "INSERT INTO messages (id, conversation_id, role, content, created_at,"
                    " status) VALUES (?,?,?,?,?,?)",
                    (store.new_id(), cid, "user", "should vanish", store.utcnow(), "complete"))
                raise RuntimeError("boom")
        self.assertEqual(store.count_messages(), before)
        self.assertEqual(store.search_messages("u1", "vanish"), [])

    def test_nested_transactions_commit_once_all_or_nothing(self):
        cid = store.create_conversation("u1", "t")
        before = store.count_messages()
        with self.assertRaises(RuntimeError):
            with store.transaction():
                store.append_message(cid, "user", "outer")      # nested transaction
                store.append_message(cid, "assistant", "inner")
                raise RuntimeError("boom")
        self.assertEqual(store.count_messages(), before,
                         "an aborted exchange must leave no partial rows")


# ---------------------------------------------------------------------------
# Durability across process restarts
# ---------------------------------------------------------------------------

class TestDurability(ChatStoreTestCase):

    def test_rows_survive_closing_and_reopening_the_database(self):
        cid = self._exchange()
        for i in range(3):
            store.append_message(cid, "user", f"question {i}")
            store.append_message(cid, "assistant", f"answer {i}")
        expected = [(m.role, m.content) for m in store.get_messages(cid)]

        store.close_connection()                 # stand-in for Ctrl+C
        store._initialised.discard(str(self.db_file))
        store.init_db()                          # stand-in for restart

        self.assertEqual([(m.role, m.content) for m in store.get_messages(cid)], expected)
        self.assertEqual(len(store.list_conversations("u1")), 1)

    def test_a_separate_process_sees_committed_rows(self):
        """This is acceptance criterion 8: the CLI count matches the store."""
        cid = self._exchange()
        store.append_message(cid, "user", "one more")
        expected = store.count_messages()

        script = (
            "import sqlite3, sys\n"
            "conn = sqlite3.connect(sys.argv[1])\n"
            "print(conn.execute('select count(*) from messages').fetchone()[0])\n"
        )
        out = subprocess.run([sys.executable, "-c", script, str(self.db_file)],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(int(out.stdout.strip()), expected)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class TestExport(ChatStoreTestCase):

    def setUp(self):
        super().setUp()
        self.cid = self._exchange(
            user_id="u1",
            question="Can bail be denied under PMLA if Section 45 twin conditions are unmet?",
            answer="No. **Section 45** twin conditions are mandatory [1], per Nikesh Shah [2].")
        self.conv = store.get_conversation(self.cid)
        self.messages = store.get_messages(self.cid)

    def test_filename_is_date_then_slug(self):
        name = exporters.export_filename(self.conv, "md")
        self.assertRegex(name, r"^\d{4}-\d{2}-\d{2}-[a-z0-9-]+\.md$")
        self.assertTrue(name.endswith(".md"))
        self.assertIn("bail", name)
        self.assertTrue(exporters.export_filename(self.conv, "txt").endswith(".txt"))

    def test_slugify_strips_punctuation_and_case(self):
        self.assertEqual(store.slugify("Section 45, PMLA — Twin Conditions?"),
                         "section-45-pmla-twin-conditions")
        self.assertEqual(store.slugify("!!!"), "conversation")

    def test_markdown_has_title_both_turns_and_numbered_authorities(self):
        md = exporters.to_markdown(self.conv, self.messages)
        self.assertIn(f"# {self.conv.title}", md)
        self.assertIn("## Question", md)
        self.assertIn("## LegalMind AI", md)
        self.assertIn("Section 45", md)
        self.assertIn("## Authorities", md)
        self.assertIn("1. **Vijay Madanlal Choudhary v. Union of India", md)
        self.assertIn("2. **Nikesh Tarachand Shah v. Union of India", md)
        self.assertIn("Court: Supreme Court of India", md)
        self.assertIn("Source: sc_2022_pmla_0001", md)

    def test_authorities_are_deduplicated_in_first_cited_order(self):
        store.append_message(self.cid, "user", "and again?")
        store.append_message(self.cid, "assistant", "Same authorities.",
                             {"citations": list(reversed(SAMPLE_CITATIONS))})
        authorities = exporters.collect_authorities(store.get_messages(self.cid))
        self.assertEqual(len(authorities), 2)
        self.assertEqual(authorities[0]["case_name"],
                         "Vijay Madanlal Choudhary v. Union of India")

    def test_absent_fields_are_omitted_not_printed_as_none(self):
        md = exporters.to_markdown(self.conv, self.messages)
        self.assertNotIn("Judges: None", md)
        self.assertNotIn("Paragraph: None", md)
        self.assertNotIn("None", md.split("## Authorities")[1])

    def test_plain_text_export_has_an_authorities_block(self):
        txt = exporters.to_text(self.conv, self.messages)
        self.assertIn("AUTHORITIES", txt)
        self.assertIn("[1] Vijay Madanlal Choudhary v. Union of India", txt)
        self.assertIn("[2] Nikesh Tarachand Shah v. Union of India", txt)
        self.assertNotIn("<mark>", txt)

    def test_export_without_citations_says_so_rather_than_breaking(self):
        cid = store.create_conversation("u1", "no authorities here")
        store.append_message(cid, "user", "a general question")
        store.append_message(cid, "assistant", "a general answer")
        conv = store.get_conversation(cid)
        md = exporters.to_markdown(conv, store.get_messages(cid))
        self.assertIn("## Authorities", md)
        self.assertIn("No authorities were recorded", md)

    def test_partial_and_error_rows_are_flagged_in_the_export(self):
        cid = store.create_conversation("u1", "interrupted")
        store.append_message(cid, "user", "long question")
        store.append_message(cid, "assistant", "half", {"status": "partial"})
        conv = store.get_conversation(cid)
        md = exporters.to_markdown(conv, store.get_messages(cid))
        self.assertIn("answer incomplete", md)

    def test_string_citations_do_not_break_the_export(self):
        """
        `run_rag()["citations"]` is a list of pre-formatted strings, not the
        structured authorities /api/chat writes. Any writer may store either,
        so rendering must cope with both instead of raising mid-export.
        """
        cid = store.create_conversation("u1", "string citations")
        store.append_message(cid, "user", "Define bail.")
        store.append_message(cid, "assistant", "Bail is...", {"citations": [
            "[1] Bharatiya Nagarik Suraksha Sanhita",
            "[2] Juvenile Justice",
            "[3] Bharatiya Nagarik Suraksha Sanhita",   # duplicate
        ]})
        conv = store.get_conversation(cid)
        messages = store.get_messages(cid)

        authorities = exporters.collect_authorities(messages)
        self.assertEqual(len(authorities), 2, "duplicates must still collapse")
        self.assertEqual(authorities[0]["case_name"], "Bharatiya Nagarik Suraksha Sanhita",
                         "the [n] marker must be stripped")

        md = exporters.to_markdown(conv, messages)
        self.assertIn("1. **Bharatiya Nagarik Suraksha Sanhita**", md)
        self.assertIn("2. **Juvenile Justice**", md)
        self.assertIn("*Relied on:*", md)
        txt = exporters.to_text(conv, messages)
        self.assertIn("[1] Bharatiya Nagarik Suraksha Sanhita", txt)

    def test_mixed_and_junk_citation_entries_are_survivable(self):
        cid = store.create_conversation("u1", "mixed")
        store.append_message(cid, "user", "q")
        store.append_message(cid, "assistant", "a", {"citations": [
            SAMPLE_CITATIONS[0], "[2] Some Act", "", None, 42, {"citation": "(2020) 1 SCC 1"},
        ]})
        conv = store.get_conversation(cid)
        messages = store.get_messages(cid)
        names = [c.get("case_name") or c.get("citation")
                 for c in exporters.collect_authorities(messages)]
        self.assertIn("Vijay Madanlal Choudhary v. Union of India", names)
        self.assertIn("Some Act", names)
        self.assertIn("(2020) 1 SCC 1", names)
        self.assertNotIn(None, names)
        for fmt in ("md", "txt"):
            _, _, body = exporters.render(conv, messages, fmt)
            self.assertIn("1.", body) if fmt == "md" else self.assertIn("[1]", body)

    def test_render_returns_filename_media_type_and_body(self):
        for fmt, expected in (("md", "text/markdown"), ("txt", "text/plain")):
            name, media_type, body = exporters.render(self.conv, self.messages, fmt)
            self.assertTrue(name.endswith(f".{fmt}"))
            self.assertIn(expected, media_type)
            self.assertTrue(body.strip())


# ---------------------------------------------------------------------------
# Shared activity feed — the global, cross-user "recent searches" list
# ---------------------------------------------------------------------------

class TestSharedActivity(ChatStoreTestCase):

    def test_feed_combines_every_users_searches(self):
        """The whole point: one laptop's search shows up for everyone."""
        self._exchange(user_id="laptop-a", question="What is quo warranto?")
        self._exchange(user_id="laptop-b", question="What is a writ of mandamus?")
        self._exchange(user_id="laptop-c", question="Explain anticipatory bail.")
        queries = [e.query for e in store.list_activity()]
        self.assertEqual(len(queries), 3, queries)
        self.assertIn("What is quo warranto?", queries)
        self.assertIn("What is a writ of mandamus?", queries)
        self.assertIn("Explain anticipatory bail.", queries)

    def test_feed_is_newest_first(self):
        for i in range(4):
            self._exchange(user_id=f"u{i}", question=f"question {i}")
        stamps = [e.created_at for e in store.list_activity()]
        self.assertEqual(stamps, sorted(stamps, reverse=True))

    def test_feed_contains_only_questions_not_answers(self):
        self._exchange(user_id="u1", question="Q?", answer="A very long answer.")
        entries = store.list_activity()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].query, "Q?")

    def test_feed_pagination_does_not_repeat_or_skip(self):
        for i in range(12):
            self._exchange(user_id=f"u{i % 3}", question=f"paged question {i}")
        self.assertEqual(store.count_activity(), 12)
        page1 = store.list_activity(limit=5, offset=0)
        page2 = store.list_activity(limit=5, offset=5)
        page3 = store.list_activity(limit=5, offset=10)
        self.assertEqual([len(page1), len(page2), len(page3)], [5, 5, 2])
        ids = [e.message_id for e in page1 + page2 + page3]
        self.assertEqual(len(set(ids)), 12, "pages must not overlap")

    def test_feed_search_spans_users(self):
        self._exchange(user_id="a", question="Bail under PMLA section 45")
        self._exchange(user_id="b", question="Tenancy dispute in Kerala")
        hits = store.list_activity(query="PMLA")
        self.assertEqual(len(hits), 1)
        self.assertIn("PMLA", hits[0].query)
        self.assertEqual(store.count_activity(query="PMLA"), 1)
        self.assertEqual(store.list_activity(query="zzzznothing"), [])

    def test_feed_search_survives_fts_operator_characters(self):
        self._exchange(user_id="a", question="Bail under PMLA")
        for q in ('bail AND OR NOT', 'PMLA -* "', 'NEAR(a b)', "it's", "***"):
            try:
                store.list_activity(query=q)
                store.count_activity(query=q)
            except Exception as e:
                self.fail(f"activity search raised on {q!r}: {e}")

    def test_soft_deleted_conversations_drop_out_of_the_feed(self):
        cid = self._exchange(user_id="a", question="Please forget this one")
        self._exchange(user_id="b", question="Keep this one")
        self.assertEqual(store.count_activity(), 2)
        store.soft_delete_conversation(cid)
        queries = [e.query for e in store.list_activity()]
        self.assertEqual(queries, ["Keep this one"])
        self.assertEqual(store.count_activity(), 1)
        store.restore_conversation(cid)
        self.assertEqual(store.count_activity(), 2)

    def test_who_is_stable_and_does_not_leak_the_raw_id(self):
        uid = "3f50a1b2-0000-4000-8000-e4beed780d78"
        self._exchange(user_id=uid, question="who am i?")
        entry = store.list_activity()[0]
        self.assertEqual(entry.who, store._who(uid))
        self.assertNotIn(uid, entry.who)
        self.assertTrue(entry.who.startswith("user-"))
        self.assertEqual(store._who(None), "anon")

    def test_unanswered_question_shows_as_pending(self):
        cid = store.create_conversation("u1", "in flight")
        store.append_message(cid, "user", "still generating…")
        entry = store.list_activity()[0]
        self.assertFalse(entry.answered)
        self.assertIsNone(entry.answer_status)

    def test_failed_answer_status_is_visible_in_the_feed(self):
        cid = store.create_conversation("u1", "boom")
        store.append_message(cid, "user", "this one failed")
        store.append_message(cid, "assistant", "", {"status": "error", "error": "OOM"})
        entry = store.list_activity()[0]
        self.assertTrue(entry.answered)
        self.assertEqual(entry.answer_status, "error")

    def test_public_read_ignores_ownership_but_hides_deleted(self):
        cid = self._exchange(user_id="someone-else", question="public please")
        conv = store.get_public_conversation(cid)
        self.assertIsNotNone(conv, "the feed must be readable by anyone")
        self.assertEqual(conv.user_id, "someone-else")
        store.soft_delete_conversation(cid)
        self.assertIsNone(store.get_public_conversation(cid),
                          "a deleted conversation must not stay publicly readable")

    def test_feed_reflects_concurrent_writers_from_several_threads(self):
        """Several people searching at once all land in the one feed."""
        n_threads, per_thread = 6, 10
        cids = [store.create_conversation(f"device-{i}", f"c{i}") for i in range(n_threads)]
        errors = []
        start = threading.Barrier(n_threads)

        def worker(i):
            try:
                start.wait(timeout=10)
                for j in range(per_thread):
                    store.append_message(cids[i], "user", f"device {i} search {j}",
                                         {"client_token": f"feed-{i}-{j}"})
            except BaseException as e:      # noqa: BLE001
                errors.append(e)
            finally:
                store.close_connection()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=60)

        self.assertEqual(errors, [], f"concurrent searches raised: {errors[:3]}")
        self.assertEqual(store.count_activity(), n_threads * per_thread)
        seen = {e.message_id for e in store.list_activity(limit=1000)}
        self.assertEqual(len(seen), n_threads * per_thread,
                         "every concurrent search must appear exactly once")

    def test_feed_is_readable_while_writes_are_happening(self):
        cid = store.create_conversation("writer", "live")
        errors, stop = [], threading.Event()

        def reader():
            try:
                while not stop.is_set():
                    store.list_activity(limit=20)
                    store.count_activity()
            except BaseException as e:      # noqa: BLE001
                errors.append(e)
            finally:
                store.close_connection()

        readers = [threading.Thread(target=reader) for _ in range(3)]
        for t in readers: t.start()
        try:
            for i in range(60):
                store.append_message(cid, "user", f"live search {i}")
        finally:
            stop.set()
            for t in readers: t.join(timeout=30)

        self.assertEqual(errors, [], f"feed reads raised during writes: {errors[:3]}")
        self.assertEqual(store.count_activity(), 60)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats(ChatStoreTestCase):

    def test_stats_counts_live_deleted_and_incomplete(self):
        a = self._exchange(user_id="u1")
        b = self._exchange(user_id="u2")
        store.append_message(b, "assistant", "oops", {"status": "error", "error": "OOM"})
        store.soft_delete_conversation(a)
        s = store.stats()
        self.assertEqual(s["conversations"], 1)
        self.assertEqual(s["deleted_conversations"], 1)
        self.assertEqual(s["messages"], 5)
        self.assertEqual(s["incomplete_messages"], 1)
        self.assertEqual(s["users"], 2)
        self.assertEqual(s["schema_version"], max(v for v, _ in store.MIGRATIONS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
