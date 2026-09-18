#!/usr/bin/env python3
"""
scripts/retriever.py
====================
LegalMind AI — Hybrid Retrieval Engine

Architecture:
    User Query
        │
        ▼
    Query preprocessing
        │
    ┌───┴───┐
    │       │
  FAISS   BM25
  Semantic Keyword
    │       │
    └───┬───┘
        ▼
    Score Fusion (Reciprocal Rank Fusion)
        │
        ▼
    Cross-encoder Reranker
        │
        ▼
    Top-K chunks with citations

This module is importable and is used by:
    - scripts/05_rag.py  (CLI testing)
    - app/main.py        (FastAPI server)

Configuration constants at top of file; also overridable at runtime.
"""

import copy
import json
import logging
import math
import os
import re
import difflib
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

# Force offline mode to prevent HuggingFace timeout delays
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

log = logging.getLogger("retriever")

# ---------------------------------------------------------------------------
# Default configuration (overridable via HybridRetriever constructor)
# ---------------------------------------------------------------------------
DEFAULT_VECTOR_DB_DIR  = "vector_db/legislation/"
DEFAULT_CHUNKS_DIR     = "data/chunks/legislation/"
DEFAULT_EMBEDDINGS_DIR = "embeddings/"
DEFAULT_EMBED_MODEL    = "BAAI/bge-m3"
DEFAULT_RERANK_MODEL   = "BAAI/bge-reranker-v2-m3"
DEFAULT_EMBED_DEVICE   = "cuda:4"      # spare GPU for embedding
DEFAULT_RERANK_DEVICE  = "cuda:4"

TOP_K_SEMANTIC = 20    # candidates from FAISS
TOP_K_BM25     = 20    # candidates from BM25
TOP_K_FINAL    = 7     # after reranking

# Reciprocal Rank Fusion constant
RRF_K = 60

# Legal query preprocessing patterns
_LEGAL_ABBR = {
    r"\bsc\b": "Supreme Court",
    r"\bhc\b": "High Court",
    r"\bipc\b": "Indian Penal Code",
    r"\bcrpc\b": "Code of Criminal Procedure",
    r"\bcpc\b": "Code of Civil Procedure",
    r"\bart\b": "Article",
    r"\bsec\b": "Section",
    r"\bsc/st\b": "Scheduled Castes and Scheduled Tribes",
    r"\bndps\b": "Narcotic Drugs and Psychotropic Substances",
    r"\bpoca\b": "Prevention of Corruption Act",
    r"\buapa\b": "Unlawful Activities Prevention Act",
}


# ---------------------------------------------------------------------------
# Utility: legal query preprocessing
# ---------------------------------------------------------------------------

def extract_statutory_targets(query: str) -> list[dict]:
    """
    Extract statutory and constitutional target provisions from query.
    e.g., 'Article 21', 'Section 302', 'Section 45 PMLA', 'Article 14'
    """
    targets = []
    # Articles
    for m in re.finditer(r"\b(?:article|art\.?)\s*(\d+[a-z]?)\b", query, re.IGNORECASE):
        targets.append({"type": "article", "num": m.group(1).lower(), "raw": m.group(0)})
    # Sections
    for m in re.finditer(r"\b(?:section|sec\.?)\s*(\d+[a-z]?)\b", query, re.IGNORECASE):
        targets.append({"type": "section", "num": m.group(1).lower(), "raw": m.group(0)})
    return targets


# ---------------------------------------------------------------------------
# Typo tolerance
# ---------------------------------------------------------------------------
# A misspelled legal term ("artucle 57") is invisible to the keyword half of
# retrieval: BM25 matches tokens literally, and the statutory matcher looks for
# the words "article"/"section" before a number, so the provision index is never
# consulted. Dense/embedding search degrades more gracefully but cannot recover
# an exact provision. Correcting the query up front fixes both.
#
# Deliberately conservative: only tokens of >= 5 letters, only when a legal term
# is a close match (difflib ratio) AND within 2 edits, and never a token that is
# already known. Words in _LEGAL_VOCAB are left untouched, so listing proper
# nouns (case names) there protects them from being "corrected".

_LEGAL_VOCAB: set[str] = {
    # structure of the law
    "article", "articles", "section", "sections", "clause", "clauses", "sub-section",
    "subsection", "schedule", "schedules", "chapter", "proviso", "provision", "provisions",
    "act", "acts", "code", "codes", "statute", "statutes", "statutory", "ordinance",
    "amendment", "amendments", "rule", "rules", "regulation", "regulations", "notification",
    "constitution", "constitutional", "preamble", "fundamental", "directive", "principles",
    # courts and people
    "court", "courts", "supreme", "high", "district", "sessions", "magistrate", "tribunal",
    "bench", "coram", "judge", "judges", "justice", "judiciary", "judgment", "judgement",
    "judgments", "judgements", "order", "orders", "decree", "ruling", "verdict",
    "petitioner", "petitioners", "respondent", "respondents", "appellant", "appellants",
    "accused", "complainant", "prosecution", "prosecutor", "defence", "counsel", "advocate",
    "plaintiff", "defendant", "witness", "witnesses", "affidavit", "pleading", "pleadings",
    # process
    "petition", "petitions", "appeal", "appeals", "revision", "review", "writ", "writs",
    "habeas", "corpus", "mandamus", "certiorari", "prohibition", "quo", "warranto",
    "quashing", "quash", "injunction", "interim", "stay", "summons", "warrant", "notice",
    "jurisdiction", "maintainability", "limitation", "precedent", "precedents", "ratio",
    "obiter", "dicta", "binding", "overruled", "distinguished", "remand", "remanded",
    # criminal
    "criminal", "penal", "offence", "offences", "offense", "cognizable", "bailable",
    "bail", "anticipatory", "custody", "arrest", "detention", "chargesheet", "charge",
    "charges", "investigation", "acquittal", "acquitted", "conviction", "convicted",
    "sentence", "sentencing", "punishment", "imprisonment", "murder", "homicide",
    "culpable", "dacoity", "robbery", "theft", "cheating", "forgery", "criminal",
    "conspiracy", "abetment", "kidnapping", "extortion", "trespass", "defamation",
    # civil / commercial
    "civil", "contract", "contracts", "agreement", "arbitration", "arbitral", "damages",
    "specific", "performance", "negotiable", "instruments", "cheque", "dishonour",
    "dishonor", "insolvency", "bankruptcy", "partnership", "company", "companies",
    "shareholder", "director", "liability", "indemnity", "tort", "negligence",
    "property", "easement", "tenancy", "eviction", "partition", "succession", "will",
    "testament", "mortgage", "lease", "licence", "license", "trademark", "copyright",
    "patent", "infringement",
    # family and personal
    "marriage", "divorce", "maintenance", "custody", "guardianship", "adoption", "dowry",
    "cruelty", "harassment", "domestic", "violence",
    # evidence and procedure
    "evidence", "procedure", "presumption", "burden", "proof", "admissible", "confession",
    "examination", "cross-examination", "testimony", "prima", "facie",
    # statutes and abbreviations
    "ipc", "crpc", "cpc", "bns", "bnss", "bsa", "pmla", "ndps", "uapa", "pocso", "sarfaesi",
    "gst", "rti", "nia", "cbi", "ed", "fir", "fera", "fema", "mcoca", "tada", "posh",
    # rights vocabulary
    "rights", "right", "liberty", "equality", "privacy", "dignity", "arbitrary",
    "arbitrariness", "proportionality", "reasonable", "restrictions", "discrimination",
    "secularism", "federalism", "separation", "powers", "basic", "structure", "doctrine",
    # frequently named parties / cases — listed so they are never "corrected"
    "kesavananda", "bharati", "puttaswamy", "maneka", "gandhi", "shayara", "bano",
    "vishaka", "nikesh", "tarachand", "shah", "vijay", "madanlal", "choudhary",
    "arnesh", "kumar", "satender", "antil", "sanjay", "chandra", "navtej", "johar",
    "indira", "sawhney", "minerva", "mills", "golaknath", "union", "india", "state",
    "kerala", "maharashtra", "delhi", "bihar", "punjab", "bengal",
}
_VOCAB_LIST: list[str] = sorted(_LEGAL_VOCAB)

_TYPO_MIN_LEN = 5      # leave short words ("act", "writ", "bail") alone
_TYPO_CUTOFF = 0.84    # difflib similarity ratio
_TYPO_MAX_EDITS = 2    # per the feature request: edit distance 1-2


def _edit_distance(a: str, b: str, cap: int = 3) -> int:
    """Levenshtein distance, short-circuited once it exceeds `cap`."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


def correct_legal_typos(text: str) -> str:
    """
    Repair misspelled legal terms ("artucle" -> "article") before retrieval.
    Unknown words that are not close to any legal term are left untouched.
    """
    if not text:
        return text
    corrections: list[tuple[str, str]] = []

    def fix(match: "re.Match") -> str:
        token = match.group(0)
        low = token.lower()
        if len(low) < _TYPO_MIN_LEN or low in _LEGAL_VOCAB:
            return token
        candidates = difflib.get_close_matches(low, _VOCAB_LIST, n=1, cutoff=_TYPO_CUTOFF)
        if not candidates:
            return token
        best = candidates[0]
        if _edit_distance(low, best, _TYPO_MAX_EDITS) > _TYPO_MAX_EDITS:
            return token
        corrections.append((token, best))
        if token.isupper():
            return best.upper()
        if token[0].isupper():
            return best.capitalize()
        return best

    # letters only: never touch numbers or mixed tokens like "138A"
    out = re.sub(r"[A-Za-z]+", fix, text)
    if corrections:
        log.info("Query typo correction: " + ", ".join(f"{a}->{b}" for a, b in corrections))
    return out


def preprocess_query(query: str) -> str:
    """
    Expand legal abbreviations and normalise query text.
    Keeps original query intact but appends expanded terms.
    """
    query = correct_legal_typos(query.strip())
    expanded = query
    for pattern, expansion in _LEGAL_ABBR.items():
        expanded = re.sub(pattern, expansion, expanded, flags=re.IGNORECASE)
    return expanded


# ---------------------------------------------------------------------------
# BM25 index wrapper
# ---------------------------------------------------------------------------

class _SimpleBM25Okapi:
    """Pure-Python dependency-free BM25Okapi fallback when rank_bm25 is unavailable."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_len) / self.corpus_size) if self.corpus_size > 0 else 1.0

        self.doc_freqs = []
        nd = {}
        for doc in corpus:
            freqs = {}
            for term in doc:
                freqs[term] = freqs.get(term, 0) + 1
            self.doc_freqs.append(freqs)
            for term in freqs:
                nd[term] = nd.get(term, 0) + 1

        self.idf = {}
        idf_sum = 0.0
        negative_idfs = []
        for term, freq in nd.items():
            val = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[term] = val
            idf_sum += val
            if val < 0:
                negative_idfs.append(term)
        avg_idf = (idf_sum / len(self.idf)) if self.idf else 0.0
        eps = self.epsilon * avg_idf
        for term in negative_idfs:
            self.idf[term] = eps

    def get_scores(self, query: list[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        doc_len = np.array(self.doc_len, dtype=np.float32)
        denom_base = self.k1 * (1.0 - self.b + self.b * doc_len / self.avgdl)
        for q in query:
            if q not in self.idf:
                continue
            q_idf = self.idf[q]
            q_freq = np.array([(doc.get(q, 0)) for doc in self.doc_freqs], dtype=np.float32)
            denom = q_freq + denom_base
            mask = denom > 0
            scores[mask] += q_idf * (q_freq[mask] * (self.k1 + 1.0) / denom[mask])
        return scores


class BM25Index:
    """Thin wrapper around rank_bm25.BM25Okapi with legal tokenisation and fallback."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.chunk_ids = [c.get("chunk_id", f"chunk_{i}") for i, c in enumerate(chunks)]
        self._build(chunks)

    def _tokenise(self, text: str) -> list[str]:
        # Tokenize words, digits, and legal composite terms
        text = text.lower()
        words = re.findall(r"\b[a-z0-9]+\b", text)
        tokens = list(words)
        phrases = re.findall(r"\b(?:article|art|section|sec)\s+\d+[a-z]?\b", text)
        for p in phrases:
            tokens.append(re.sub(r"\s+", " ", p))
        return tokens

    def _build(self, chunks: list[dict]):
        corpus = [self._tokenise(c.get("text", "")) for c in chunks]
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(corpus)
            self._available = True
            log.info(f"BM25 index built on {len(chunks):,} chunks")
        except ImportError:
            log.info("rank_bm25 not installed — using built-in BM25Okapi fallback")
            self.bm25 = _SimpleBM25Okapi(corpus)
            self._available = True

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Returns list of (chunk_index, score) sorted by descending score."""
        if not self._available or self.bm25 is None:
            return []
        tokens = self._tokenise(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        if len(scores) == 0:
            return []
        # Get top-k indices using argpartition for O(N) performance
        if len(scores) > top_k:
            top_partition = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_partition[np.argsort(scores[top_partition])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


# ---------------------------------------------------------------------------
# FAISS semantic search
# ---------------------------------------------------------------------------

class FAISSIndex:
    """FAISS index wrapper for dense vector search."""

    def __init__(self, vector_db_dir: str, embed_model: str, device: str):
        self.db_dir = Path(vector_db_dir)
        self.embed_model_name = embed_model
        self.device = device
        self._loaded = False
        self._query_embed_cache: dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._load()

    def _load(self):
        index_file  = self.db_dir / "index.faiss"
        ids_file    = self.db_dir / "chunk_ids.json"

        if not index_file.exists():
            # Auto-detect subdirectory (e.g. vector_db/legislation)
            candidates = list(self.db_dir.glob("*/index.faiss"))
            if candidates:
                index_file = candidates[0]
                ids_file = index_file.parent / "chunk_ids.json"
                self.db_dir = index_file.parent
                log.info(f"Using FAISS index in subdirectory: {self.db_dir}")
            else:
                log.warning(f"FAISS index not found at {index_file}. Run 04_build_faiss.py first.")
                return

        try:
            import faiss
            self.index = faiss.read_index(str(index_file))
            log.info(f"FAISS index loaded: {self.index.ntotal:,} vectors")
        except ImportError:
            log.warning("FAISS not installed. Semantic search disabled.")
            return
        except Exception as e:
            log.error(f"Failed to load FAISS index: {e}")
            return

        with open(ids_file, encoding="utf-8") as f:
            self.chunk_ids = json.load(f)

        # Load embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer(self.embed_model_name, device=self.device)
            log.info(f"Embedding model loaded for retrieval: {self.embed_model_name}")
            self._loaded = True
        except Exception as e:
            log.error(f"Failed to load embedding model: {e}")

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Returns list of (chunk_id, score) sorted by descending score."""
        if not self._loaded:
            return []

        norm_q = query.strip().lower()
        with self._cache_lock:
            cached_vec = self._query_embed_cache.get(norm_q)

        if cached_vec is not None:
            query_vec = cached_vec
        else:
            query_vec = self.embed_model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
            with self._cache_lock:
                if len(self._query_embed_cache) >= 1024:
                    self._query_embed_cache.pop(next(iter(self._query_embed_cache)))
                self._query_embed_cache[norm_q] = query_vec

        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            results.append((self.chunk_ids[idx], float(score)))
        return results


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

class Reranker:
    """Cross-encoder reranker for candidate re-scoring."""

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self._loaded = False
        self._load()

    def _load(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device)
            if self.device.startswith("cuda") and hasattr(self.model, "model"):
                try:
                    self.model.model.eval()
                    if hasattr(self.model.model, "half"):
                        self.model.model.half()
                        log.info(f"Reranker converted to FP16 on {self.device}")
                except Exception as e:
                    log.debug(f"Could not convert reranker to half precision: {e}")
            self._loaded = True
            log.info(f"Reranker loaded: {self.model_name}")
        except ImportError:
            log.warning("sentence-transformers not installed — reranker disabled.")
        except Exception as e:
            log.warning(f"Could not load reranker {self.model_name}: {e}. Reranker disabled.")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
    ) -> list[dict]:
        """
        Re-score and return top_k candidates.
        Each candidate dict must have a 'text' field.
        """
        if not self._loaded or not candidates:
            return candidates[:top_k]

        pairs = [(query, c.get("text", "")[:512]) for c in candidates]
        try:
            import torch
            with torch.inference_mode():
                scores = self.model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
            for i, c in enumerate(candidates):
                c["rerank_score"] = float(scores[i])
            ranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
            return ranked[:top_k]
        except Exception as e:
            log.warning(f"Reranker prediction failed: {e}. Using pre-rerank order.")
            return candidates[:top_k]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    results_lists: list[list[tuple[str, float]]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.
    Each input list is [(chunk_id, score), ...] in descending score order.
    Returns merged [(chunk_id, rrf_score), ...] in descending order.
    """
    scores: dict[str, float] = {}
    for result_list in results_lists:
        for rank, (chunk_id, _) in enumerate(result_list):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def find_exact_provision_matches(chunks: list[dict], query: str, top_n: int = 10) -> list[tuple[str, float]]:
    """
    Scan chunks for exact section/article matches (e.g. Article 21, Article 14, Section 302, Section 45 PMLA).
    Prioritizes:
      - Exact article/section number definition (e.g. '**21.**', 'Section: 21', 'Article 21')
      - Matching Act title / document_id (e.g. 'Constitution' when query asks for Constitution)
    Returns list of (chunk_id, score).
    """
    targets = extract_statutory_targets(query)
    if not targets:
        return []

    q_lower = query.lower()
    act_hints = []
    if "constitution" in q_lower:
        act_hints.append("constitution")
    if "pmla" in q_lower or "money laundering" in q_lower:
        act_hints.extend(["money laundering", "pmla"])
    if "ipc" in q_lower or "penal code" in q_lower:
        act_hints.extend(["penal", "nyaya"])
    if "crpc" in q_lower or "criminal procedure" in q_lower:
        act_hints.extend(["criminal procedure", "nagarik suraksha"])
    if "cpc" in q_lower or "civil procedure" in q_lower:
        act_hints.append("civil procedure")
    if "contract" in q_lower:
        act_hints.append("contract")
    if "companies" in q_lower:
        act_hints.append("companies")

    # Precompile target regexes for performance
    compiled_targets = []
    for t in targets:
        num = re.escape(t["num"])
        exact_rx = re.compile(rf"(?:\*\*{num}\.\*\*|Section:\s*{num}\b|\barticle\s+{num}\b|\bsection\s+{num}\b)", re.IGNORECASE)
        num_rx   = re.compile(rf"\b{num}\b", re.IGNORECASE)
        compiled_targets.append((exact_rx, num_rx))

    scored_matches = []
    for idx, chunk in enumerate(chunks):
        cid = chunk.get("chunk_id", f"chunk_{idx}")
        text = chunk.get("text", "")
        title = (chunk.get("title") or "") + " " + (chunk.get("act_name") or "")
        title_lower = title.lower()
        chunk_art = str(chunk.get("article_number") or chunk.get("article") or "").lower().strip()
        chunk_sec = str(chunk.get("section_number") or chunk.get("section") or "").lower().strip()

        chunk_score = 0.0

        # Direct metadata match on target number
        for t in targets:
            tnum = t["num"].lower()
            if t["type"] == "article" and chunk_art == tnum:
                score = 3.5 if "constitution" in title_lower else 2.5
                if score > chunk_score:
                    chunk_score = score
            elif t["type"] == "section" and chunk_sec == tnum:
                score = 3.5 if (act_hints and any(h in title_lower for h in act_hints)) else 2.5
                if score > chunk_score:
                    chunk_score = score

        # Text regex matching
        for exact_rx, num_rx in compiled_targets:
            exact_def = bool(exact_rx.search(text))
            if not exact_def and not num_rx.search(text):
                continue

            base_s = 2.0 if exact_def else 0.5
            if act_hints:
                if any(hint in title_lower for hint in act_hints):
                    base_s += 2.0
                else:
                    base_s *= 0.3
            else:
                base_s += 1.0

            if base_s > chunk_score:
                chunk_score = base_s

        if chunk_score >= 1.5:
            scored_matches.append((cid, chunk_score))

    scored_matches.sort(key=lambda x: x[1], reverse=True)
    return scored_matches[:top_n]


class HybridRetriever:
    """
    Main retrieval class combining FAISS semantic search, BM25 keyword
    search, RRF score fusion, and cross-encoder reranking.
    """

    def __init__(
        self,
        chunks: list[dict] | None = None,
        chunks_dir: str = DEFAULT_CHUNKS_DIR,
        vector_db_dir: str = DEFAULT_VECTOR_DB_DIR,
        embeddings_dir: str = DEFAULT_EMBEDDINGS_DIR,
        embed_model: str = DEFAULT_EMBED_MODEL,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        embed_device: str = DEFAULT_EMBED_DEVICE,
        rerank_device: str = DEFAULT_RERANK_DEVICE,
        top_k_semantic: int = TOP_K_SEMANTIC,
        top_k_bm25: int = TOP_K_BM25,
        top_k_final: int = TOP_K_FINAL,
    ):
        self.top_k_semantic = top_k_semantic
        self.top_k_bm25 = top_k_bm25
        self.top_k_final = top_k_final

        # Load chunks (for BM25 and metadata lookup)
        if chunks is not None:
            self.chunks = chunks
        else:
            self.chunks = self._load_chunks(chunks_dir, vector_db_dir)

        self._chunk_by_id: dict[str, dict] = {
            c.get("chunk_id", f"c{i}"): c
            for i, c in enumerate(self.chunks)
        }

        # Build BM25 index
        log.info("Building BM25 index…")
        self.bm25 = BM25Index(self.chunks)

        # Load FAISS
        log.info("Loading FAISS index…")
        self.faiss = FAISSIndex(vector_db_dir, embed_model, embed_device)

        # Load reranker
        log.info("Loading reranker…")
        self.reranker = Reranker(rerank_model, rerank_device)

        # Retrieval result cache & inverted statutory index for ultra-fast lookup
        self._retrieval_cache: dict[tuple[str, int], list[dict]] = {}
        self._retrieval_cache_lock = threading.Lock()
        self._build_provision_index()

        log.info("HybridRetriever ready.")

    # ── Cache plumbing ─────────────────────────────────────────────────────
    # Normally created in __init__. These guards keep retrieval working when an
    # instance predates the attribute (hot-reload rebinding, an unpickled or
    # copied instance whose __init__ never ran) instead of raising
    # AttributeError: ... object has no attribute '_retrieval_cache_lock'.
    def _cache_lock(self):
        lock = getattr(self, "_retrieval_cache_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._retrieval_cache_lock = lock
        if getattr(self, "_retrieval_cache", None) is None:
            self._retrieval_cache = {}
        return lock

    def __getstate__(self):
        # threading.Lock is not picklable; drop it and rebuild on restore.
        state = self.__dict__.copy()
        state.pop("_retrieval_cache_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._retrieval_cache = state.get("_retrieval_cache") or {}
        self._retrieval_cache_lock = threading.Lock()

    def _build_provision_index(self):
        """
        Build an in-memory inverted provision lookup index.
        Maps (type, num) -> list of chunk_ids for instant O(1) statutory matching.
        """
        self._provision_index: dict[tuple[str, str], list[str]] = {}
        for chunk in self.chunks:
            cid = chunk.get("chunk_id")
            if not cid:
                continue
            art = str(chunk.get("article_number") or chunk.get("article") or "").lower().strip()
            if art:
                clean_art = re.sub(r"[^\w]", "", art)
                self._provision_index.setdefault(("article", clean_art), []).append(cid)
                num_only = re.match(r"^\d+", clean_art)
                if num_only and num_only.group(0) != clean_art:
                    self._provision_index.setdefault(("article", num_only.group(0)), []).append(cid)

            sec = str(chunk.get("section_number") or chunk.get("section") or "").lower().strip()
            if sec:
                clean_sec = re.sub(r"[^\w]", "", sec)
                self._provision_index.setdefault(("section", clean_sec), []).append(cid)
                num_only = re.match(r"^\d+", clean_sec)
                if num_only and num_only.group(0) != clean_sec:
                    self._provision_index.setdefault(("section", num_only.group(0)), []).append(cid)

    @staticmethod
    def _load_chunks(chunks_dir: str, vector_db_dir: str = "") -> list[dict]:
        chunks = []
        chunks_path = Path(chunks_dir)

        # Detect target chunk IDs if vector_db_dir contains chunk_ids.json
        target_ids = None
        if vector_db_dir:
            ids_file = Path(vector_db_dir) / "chunk_ids.json"
            if ids_file.is_file():
                try:
                    with open(ids_file, encoding="utf-8") as f:
                        target_ids = set(json.load(f))
                    log.info(f"Targeting {len(target_ids):,} indexed chunks from {ids_file}")
                except Exception as e:
                    log.warning(f"Could not read chunk_ids from {ids_file}: {e}")
                    target_ids = None

        if (chunks_path / "indexed_chunks.jsonl").is_file():
            chunk_files = [chunks_path / "indexed_chunks.jsonl"]
        elif chunks_path.is_file():
            chunk_files = [chunks_path]
        elif (chunks_path / "chunks.jsonl").is_file():
            chunk_files = [chunks_path / "chunks.jsonl"]
        elif vector_db_dir and (chunks_path / Path(vector_db_dir).name / "indexed_chunks.jsonl").is_file():
            chunk_files = [chunks_path / Path(vector_db_dir).name / "indexed_chunks.jsonl"]
        elif vector_db_dir and (chunks_path / Path(vector_db_dir).name / "chunks.jsonl").is_file():
            chunk_files = [chunks_path / Path(vector_db_dir).name / "chunks.jsonl"]
        elif (chunks_path / "legislation" / "chunks.jsonl").is_file() and not (Path(vector_db_dir) / "case_laws" / "index.faiss").exists():
            chunk_files = [chunks_path / "legislation" / "chunks.jsonl"]
        else:
            chunk_files = list(chunks_path.glob("**/chunks.jsonl"))
            if len(chunk_files) > 1 and any("legislation" in str(p) for p in chunk_files):
                chunk_files = [p for p in chunk_files if "legislation" in str(p)]

        if not chunk_files:
            log.error(f"chunks.jsonl not found in {chunks_dir} or its subdirectories. Run 02_chunk.py first.")
            return []

        for chunks_file in chunk_files:
            log.info(f"Loading chunks from {chunks_file}…")
            with open(chunks_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            c = json.loads(line)
                            if target_ids is not None:
                                cid = c.get("chunk_id")
                                if cid in target_ids:
                                    chunks.append(c)
                                    if len(chunks) >= len(target_ids):
                                        break
                            else:
                                chunks.append(c)
                        except json.JSONDecodeError:
                            pass
        log.info(f"Loaded {len(chunks):,} chunks for BM25")
        return chunks

    def _find_exact_provision_matches(self, query: str, top_n: int = 10) -> list[tuple[str, float]]:
        targets = extract_statutory_targets(query)
        if not targets:
            return []

        # Fast lookup via inverted provision index
        candidate_cids = set()
        for t in targets:
            clean_num = re.sub(r"[^\w]", "", t["num"].lower())
            for cid in self._provision_index.get((t["type"], clean_num), []):
                candidate_cids.add(cid)
            num_only = re.match(r"^\d+", clean_num)
            if num_only:
                for cid in self._provision_index.get((t["type"], num_only.group(0)), []):
                    candidate_cids.add(cid)

        if candidate_cids:
            subset_chunks = [self._chunk_by_id[cid] for cid in candidate_cids if cid in self._chunk_by_id]
            return find_exact_provision_matches(subset_chunks, query, top_n)
        return find_exact_provision_matches(self.chunks, query, top_n)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        return_scores: bool = True,
    ) -> list[dict]:
        """
        Full retrieval pipeline:
        1. Query preprocessing & normalization
        2. LRU retrieval cache lookup (sub-millisecond instant return)
        3. Exact provision matching (via inverted index)
        4. Semantic search (FAISS dense vectors with query embedding cache)
        5. Keyword search (BM25 with argpartition)
        6. RRF score fusion
        7. Candidate pruning & cross-encoder reranking
        8. Return top_k chunks with citation metadata
        """
        t0 = time.time()
        top_k = top_k or self.top_k_final
        processed_query = preprocess_query(query)
        cache_key = (processed_query.strip().lower(), top_k)

        with self._cache_lock():
            cached = self._retrieval_cache.get(cache_key)
        if cached is not None:
            copied = copy.deepcopy(cached)
            for c in copied:
                c["retrieval_latency_s"] = 0.001
            log.debug(f"Retrieval cache hit for '{processed_query[:40]}' (0.001s)")
            return copied

        # ── Exact provision search ─────────────────────────────────────────
        exact_matches = self._find_exact_provision_matches(processed_query)

        # ── Semantic search ────────────────────────────────────────────────
        semantic_results = self.faiss.search(processed_query, self.top_k_semantic)
        log.debug(f"FAISS returned {len(semantic_results)} results")

        # ── BM25 search ────────────────────────────────────────────────────
        bm25_results_raw = self.bm25.search(processed_query, self.top_k_bm25)
        bm25_results: list[tuple[str, float]] = [
            (self.chunks[idx].get("chunk_id", f"chunk_{idx}"), score)
            for idx, score in bm25_results_raw
        ]
        log.debug(f"BM25 returned {len(bm25_results)} results")

        # ── RRF fusion (incorporating exact matches) ───────────────────────
        fusion_inputs = [semantic_results, bm25_results]
        if exact_matches:
            fusion_inputs.insert(0, exact_matches)
        fused = reciprocal_rank_fusion(fusion_inputs)

        # Prune candidate pool for cross-encoder reranking (speeds up reranker by 3x-4x)
        max_candidates = min(len(fused), max(10, top_k * 2 + 2))
        candidate_ids = [cid for cid, _ in fused[:max_candidates]]
        for cid, _ in exact_matches:
            if cid in self._chunk_by_id and cid not in candidate_ids:
                candidate_ids.insert(0, cid)

        # ── Lookup chunk metadata ──────────────────────────────────────────
        candidates: list[dict] = []
        for i, cid in enumerate(candidate_ids):
            chunk = self._chunk_by_id.get(cid)
            if chunk is None:
                continue
            c = dict(chunk)   # shallow copy to avoid mutating the source
            c["rrf_score"] = float(fused[i][1]) if i < len(fused) else 0.0
            candidates.append(c)

        if not candidates:
            log.warning("No candidates found. Check that the index is built and chunks are loaded.")
            return []

        # ── Rerank ────────────────────────────────────────────────────────
        reranked = self.reranker.rerank(query, candidates, top_k)

        # ── Add citation metadata ──────────────────────────────────────────
        for i, chunk in enumerate(reranked):
            chunk["citation_id"] = i + 1
            chunk["retrieval_latency_s"] = round(time.time() - t0, 3)

        # Cache result
        with self._cache_lock():
            if len(self._retrieval_cache) >= 512:
                self._retrieval_cache.pop(next(iter(self._retrieval_cache)))
            self._retrieval_cache[cache_key] = copy.deepcopy(reranked)

        log.debug(f"Retrieval complete in {time.time()-t0:.3f}s, returning {len(reranked)} chunks")
        return reranked


# ---------------------------------------------------------------------------
# Citation formatter
# ---------------------------------------------------------------------------

def format_citations(chunks: list[dict]) -> list[str]:
    """
    Format retrieved chunks into numbered citation strings.

    Returns list of citation strings like:
        [1] Constitution of India — Article 21
        [2] Maneka Gandhi v. Union of India — Supreme Court — 1978 — (1978) 1 SCC 248
    """
    citations = []
    for chunk in chunks:
        cid = chunk.get("citation_id", "?")
        doc_type = chunk.get("document_type", "unknown")
        title = chunk.get("title", "Unknown Document")

        if doc_type in ("act", "constitution"):
            act = chunk.get("act_name") or title
            art = chunk.get("article", "")
            sec = chunk.get("section", "")
            part = f"Article {art}" if art else (f"Section {sec}" if sec else "")
            citation = f"[{cid}] {act}" + (f" — {part}" if part else "")

        elif doc_type == "judgment":
            court = chunk.get("court", "")
            date  = chunk.get("date", "")
            ref   = chunk.get("citation", "")
            parts = [title]
            if court: parts.append(court)
            if date:  parts.append(date)
            if ref:   parts.append(ref)
            citation = f"[{cid}] " + " — ".join(parts)

        else:
            src = chunk.get("source", "Unknown Source")
            citation = f"[{cid}] {title} ({src})"

        citations.append(citation)

    return citations


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")

    parser = argparse.ArgumentParser(description="LegalMind AI — Retriever test")
    parser.add_argument("--query", "-q", required=True, help="Legal query to retrieve")
    parser.add_argument("--top-k", type=int, default=TOP_K_FINAL)
    parser.add_argument("--chunks-dir",     default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--vector-db-dir",  default=DEFAULT_VECTOR_DB_DIR)
    parser.add_argument("--embed-model",    default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--rerank-model",   default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--embed-device",   default=DEFAULT_EMBED_DEVICE)
    parser.add_argument("--no-rerank",      action="store_true")
    args = parser.parse_args()

    reranker = DEFAULT_RERANK_MODEL if not args.no_rerank else None

    retriever = HybridRetriever(
        chunks_dir    = args.chunks_dir,
        vector_db_dir = args.vector_db_dir,
        embed_model   = args.embed_model,
        rerank_model  = reranker or "",
        embed_device  = args.embed_device,
        top_k_final   = args.top_k,
    )

    print(f"\n{'='*70}")
    print(f"QUERY: {args.query}")
    print(f"{'='*70}\n")

    results = retriever.retrieve(args.query, top_k=args.top_k)

    if not results:
        print("No results found.")
        sys.exit(0)

    citations = format_citations(results)
    for i, (chunk, citation) in enumerate(zip(results, citations)):
        print(f"\n{citation}")
        print(f"  Score (RRF): {chunk.get('rrf_score', 0):.4f}  "
              f"Rerank: {chunk.get('rerank_score', 'N/A')}")
        print(f"  Type: {chunk.get('document_type','?')}  "
              f"Source: {chunk.get('source','?')}")
        print(f"  Text (preview): {chunk.get('text','')[:300]}…")
        print()
