"""
tests/test_retriever_rebind.py
==============================
Regression tests for:
    AttributeError: 'MultiRetriever' object has no attribute '_retrieval_cache_lock'

Cause: after scripts/05_rag.py changed, app/main.py re-bound HybridRetriever's
methods onto the live retriever. With LEGALMIND_RETRIEVAL_SETS set, that live
object is a MultiRetriever wrapping HybridRetrievers, so it received a
retrieve() that expects HybridRetriever internals it does not have.

Run: .venv/bin/python -m unittest tests.test_retriever_rebind -v
"""

import pickle
import threading
import unittest

from app.main import _rebind_retriever
from app.services.multi_retriever import MultiRetriever


class FakeHybridRetriever:
    """Stands in for a HybridRetriever instance created before the reload."""

    def __init__(self, name="idx", explode_on_index=False):
        self.name = name
        self.chunks = [{"chunk_id": f"{name}-1", "article_number": "57", "text": "Article 57 …"}]
        self.bm25 = object()
        self.explode_on_index = explode_on_index
        self._retrieval_cache = {}
        self._retrieval_cache_lock = threading.Lock()

    def retrieve(self, query, top_k=5):
        return [{"chunk_id": f"{self.name}-1", "text": "old method"}]


class ReloadedHybridRetriever:
    """Stands in for the freshly reloaded HybridRetriever class."""

    def retrieve(self, query, top_k=5):
        with self._retrieval_cache_lock:          # what used to blow up
            return [{"chunk_id": f"{self.name}-1", "text": "new method", "chunks_seen": len(self.chunks)}]

    def _find_exact_provision_matches(self, query):
        return []

    def _build_provision_index(self):
        if getattr(self, "explode_on_index", False):
            raise RuntimeError("index build failed")
        self._provision_index = {("article", "57"): [f"{self.name}-1"]}


class TestRebindOntoMultiRetriever(unittest.TestCase):

    def _multi(self, **kw):
        return MultiRetriever([("legislation", FakeHybridRetriever("legislation", **kw)),
                               ("case_laws", FakeHybridRetriever("case_laws", **kw))])

    def test_wrapper_keeps_its_own_retrieve(self):
        multi = self._multi()
        _rebind_retriever(multi, ReloadedHybridRetriever)
        # The bug: the wrapper's retrieve was replaced by HybridRetriever's.
        self.assertEqual(multi.retrieve.__func__, MultiRetriever.retrieve)
        self.assertFalse(hasattr(multi, "_retrieval_cache_lock"),
                         "HybridRetriever internals must not be grafted onto the wrapper")

    def test_inner_retrievers_are_rebound(self):
        multi = self._multi()
        self.assertEqual(_rebind_retriever(multi, ReloadedHybridRetriever), 2)
        for _, inner in multi.retrievers:
            self.assertEqual(inner.retrieve.__func__, ReloadedHybridRetriever.retrieve)
            self.assertIsNotNone(getattr(inner, "_retrieval_cache_lock", None))
            self.assertEqual(inner._provision_index, {("article", "57"): [f"{inner.name}-1"]})

    def test_query_after_rebind_does_not_raise(self):
        """The end-to-end symptom: the next query used to raise AttributeError."""
        multi = self._multi()
        _rebind_retriever(multi, ReloadedHybridRetriever)
        out = multi.retrieve("what is the artucle 57", top_k=3)
        self.assertTrue(out and out[0]["text"] == "new method")

    def test_failed_index_build_still_leaves_a_lock(self):
        multi = self._multi(explode_on_index=True)
        for _, inner in multi.retrievers:        # force the guarded path
            del inner._retrieval_cache_lock
            del inner._retrieval_cache
        _rebind_retriever(multi, ReloadedHybridRetriever)
        for _, inner in multi.retrievers:
            self.assertIsNotNone(getattr(inner, "_retrieval_cache_lock", None))
            self.assertEqual(inner._retrieval_cache, {})
        self.assertTrue(multi.retrieve("article 57", top_k=2))

    def test_plain_hybrid_retriever_still_rebinds(self):
        single = FakeHybridRetriever("only")
        self.assertEqual(_rebind_retriever(single, ReloadedHybridRetriever), 1)
        self.assertEqual(single.retrieve.__func__, ReloadedHybridRetriever.retrieve)

    def test_repeated_reloads_are_stable(self):
        multi = self._multi()
        for _ in range(5):
            _rebind_retriever(multi, ReloadedHybridRetriever)
            self.assertTrue(multi.retrieve("article 57", top_k=2))


class TestHybridRetrieverCacheGuards(unittest.TestCase):
    """The defensive half: an instance whose __init__ never ran must still work."""

    @classmethod
    def setUpClass(cls):
        try:
            from retriever import HybridRetriever
        except Exception as e:      # heavy optional deps
            raise unittest.SkipTest(f"retriever module unavailable: {e}")
        cls.HybridRetriever = HybridRetriever

    def test_cache_lock_created_lazily(self):
        obj = self.HybridRetriever.__new__(self.HybridRetriever)   # no __init__
        self.assertFalse(hasattr(obj, "_retrieval_cache_lock"))
        with obj._cache_lock():
            obj._retrieval_cache["k"] = ["v"]
        self.assertIsNotNone(obj._retrieval_cache_lock)
        with obj._cache_lock():                                     # reuses the same lock
            self.assertEqual(obj._retrieval_cache["k"], ["v"])

    def test_pickle_round_trip_restores_lock(self):
        obj = self.HybridRetriever.__new__(self.HybridRetriever)
        obj._retrieval_cache = {("q", 5): [{"chunk_id": "c1"}]}
        obj._retrieval_cache_lock = threading.Lock()
        restored = pickle.loads(pickle.dumps(obj))                  # lock is unpicklable
        self.assertIsNotNone(getattr(restored, "_retrieval_cache_lock", None))
        with restored._cache_lock():
            self.assertIn(("q", 5), restored._retrieval_cache)


if __name__ == "__main__":
    unittest.main()
