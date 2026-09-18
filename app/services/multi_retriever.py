#!/usr/bin/env python3
"""
app/services/multi_retriever.py
===============================
Query several HybridRetriever indexes and merge the results.

The app could previously load exactly one vector store, and it was pointed at
`vector_db/legislation/`. That is why a question about Kesavananda Bharati
returned five chunks of the Constitution and no judgment at all: the case-law
index was never consulted. A legal answer usually needs both — the statutory
provision AND the judgment interpreting it — so retrieval has to span them.

Merging is by reranker score where available, falling back to reciprocal rank
fusion, which is what the underlying HybridRetriever already uses to combine
BM25 with dense search. Each source keeps its own metadata, so citation
verification can still tell a judgment from a statute.

This wraps the existing retriever rather than modifying it; scripts/retriever.py
is untouched.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("legalmind.retrieval")


class MultiRetriever:
    """Drop-in stand-in for HybridRetriever with a .retrieve(query, top_k)."""

    def __init__(self, retrievers: list[tuple[str, Any]]):
        # [(name, HybridRetriever)] — name is carried onto each chunk so the
        # answer (and the verifier) can tell which corpus a passage came from.
        self.retrievers = [(n, r) for n, r in retrievers if r is not None]
        log.info(f"MultiRetriever over {len(self.retrievers)} index(es): "
                 f"{[n for n, _ in self.retrievers]}")

    def __bool__(self):
        return bool(self.retrievers)

    def retrieve(self, query: str, top_k: int = 7) -> list[dict]:
        if not self.retrievers:
            return []
        if len(self.retrievers) == 1:
            return self._tag(self.retrievers[0][0], self.retrievers[0][1].retrieve(query, top_k))

        pooled: list[dict] = []
        for name, r in self.retrievers:
            try:
                # Over-fetch from each so the merge has something to choose
                # between rather than rubber-stamping each index's top-k.
                got = r.retrieve(query, top_k=max(top_k, 10))
            except Exception as e:
                log.warning(f"retriever '{name}' failed: {e}")
                continue
            pooled.extend(self._tag(name, got))

        def key(c: dict) -> float:
            if c.get("rerank_score") is not None:
                return float(c["rerank_score"])
            # RRF scores are ~0.01-0.05; shift below any real rerank logit so
            # reranked hits always sort above un-reranked ones.
            return -100.0 + float(c.get("rrf_score") or 0.0)

        pooled.sort(key=key, reverse=True)

        # Deduplicate on chunk id, then on the leading text, since the same
        # judgment can appear in more than one index.
        out, seen_id, seen_txt = [], set(), set()
        for c in pooled:
            cid = c.get("chunk_id") or c.get("document_id")
            txt = (c.get("text") or "")[:160].strip().lower()
            if cid and cid in seen_id:
                continue
            if txt and txt in seen_txt:
                continue
            if cid:
                seen_id.add(cid)
            if txt:
                seen_txt.add(txt)
            out.append(c)
            if len(out) >= top_k:
                break

        # Renumber so [Source N] in the prompt matches the list we return.
        for i, c in enumerate(out, start=1):
            c["citation_id"] = i
        return out

    @staticmethod
    def _tag(name: str, chunks: list[dict]) -> list[dict]:
        for c in chunks or []:
            c.setdefault("corpus", name)
        return chunks or []
