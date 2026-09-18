#!/usr/bin/env python3
"""
scripts/constitutional_retrieval.py
====================================
LegalMind AI — Constitutional & Golden Triangle Retrieval Engine

Specialized retrieval orchestrator for Indian Constitutional inquiries:
- Multi-article extraction and balanced retrieval (Articles 14, 19, 21)
- Golden Triangle query routing (Articles 14, 19, 21 + Maneka Gandhi + Reasonable Restrictions)
- Clause-level extraction (e.g., Article 19(1)(a)–(g), 19(2)–(6))
- Cross-corpus case law retrieval linking landmark Supreme Court precedents:
    * Maneka Gandhi v. Union of India (1978)
    * A.K. Gopalan v. State of Madras (1950)
    * E.P. Royappa v. State of Tamil Nadu (1974)
    * State of West Bengal v. Anwar Ali Sarkar (1952)
    * Romesh Thappar v. State of Madras (1950)
    * Bennett Coleman v. Union of India (1972)
    * Olga Tellis v. Bombay Municipal Corporation (1985)
    * K.S. Puttaswamy v. Union of India (2017)
    * Shreya Singhal v. Union of India (2015)
- Prevents single-article domination: guarantees evidence representation for each queried article
- Computes per-article evidence coverage:
    { "article_14": {...}, "article_19": {...}, "article_21": {...} }
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger("constitutional_retrieval")

# ---------------------------------------------------------------------------
# Query Intent & Entity Extraction
# ---------------------------------------------------------------------------

_GOLDEN_TRIANGLE_KEYWORDS = [
    "golden triangle",
    "articles 14, 19 and 21",
    "articles 14, 19, and 21",
    "article 14, 19, 21",
    "article 14, article 19, and article 21",
    "article 14, article 19, article 21",
    "article 14 and article 21",
    "article 19 and article 21",
    "equality, freedom and personal liberty",
    "equality, freedom, and personal liberty",
    "maneka gandhi doctrine",
    "trinity of fundamental rights",
]

_LANDMARK_CONSTITUTIONAL_CASES = {
    "maneka gandhi": [
        "MANEKA GANDHI versus UNION OF INDIA",
        "Maneka Gandhi v. Union of India",
    ],
    "gopalan": [
        "A.K. GOPALAN versus THE STATE OF MADRAS",
        "A.K. Gopalan v. State of Madras",
    ],
    "royappa": [
        "E. P. ROYAPPA versus STATE OF TAMIL NADU & ANR",
        "E.P. Royappa v. State of Tamil Nadu",
    ],
    "anwar ali": [
        "THE STATE OF WEST BENGAL versus ANWAR ALI SARKAR",
        "State of West Bengal v. Anwar Ali Sarkar",
    ],
    "romesh thappar": [
        "ROMESH THAPPAR versus THE STATE OF MADRAS",
        "Romesh Thappar v. State of Madras",
    ],
    "bennett coleman": [
        "BENNETT COLEMAN & CO. (P) LTD versus PUNYA PRIYA DAS GUPTA",
        "Bennett Coleman & Co. v. Union of India",
    ],
    "olga tellis": [
        "OLGA TELLIS & ORS versus BOMBAY MUNICIPAL CORPORATION & ORS. ETC",
        "Olga Tellis v. Bombay Municipal Corporation",
    ],
}


def is_constitutional_query(query: str) -> bool:
    """Check if query is asking about the Constitution of India or its articles."""
    q_lower = query.lower()
    if any(kw in q_lower for kw in _GOLDEN_TRIANGLE_KEYWORDS):
        return True
    if "constitution" in q_lower:
        return True
    if re.search(r"\b(?:article|art\.?)\s*\d+", q_lower):
        return True
    if any(k in q_lower for k in ["fundamental right", "fundamental duty", "directive principles", "writ of", "habeas corpus", "mandamus", "certiorari", "quo warranto", "basic structure"]):
        return True
    return False


def is_golden_triangle_query(query: str) -> bool:
    """Check if query specifically touches the Golden Triangle (Arts 14, 19, 21)."""
    q_lower = query.lower()
    if any(kw in q_lower for kw in _GOLDEN_TRIANGLE_KEYWORDS):
        return True
    articles = extract_constitutional_articles(query)
    art_set = set(articles)
    if {"14", "19", "21"}.issubset(art_set):
        return True
    if len(art_set.intersection({"14", "19", "21"})) >= 2 and ("connect" in q_lower or "differ" in q_lower or "restrict" in q_lower or "triangle" in q_lower):
        return True
    return False


def extract_constitutional_articles(query: str) -> list[str]:
    """Extract distinct article numbers present in query, including plural sequences."""
    seen = set()
    res = []

    def _add(val: str):
        val = val.strip().upper()
        if val and val not in seen:
            seen.add(val)
            res.append(val)

    # 1. Direct "Article N" or "Art. N" matches
    for m in re.finditer(r"\b(?:articles?|art\.?s?)\s*(\d+[a-z]?)\b", query, re.IGNORECASE):
        _add(m.group(1))

    # 2. Plural sequences: "Articles 14, 19, and 21" or "Articles 14, 19 & 21" or "Articles 14 and 21"
    for m in re.finditer(r"\b(?:articles?|art\.?s?)\s+([0-9a-z\s,and&]+)", query, re.IGNORECASE):
        sub_nums = re.findall(r"\b(\d+[a-z]?)\b", m.group(1), re.IGNORECASE)
        for num in sub_nums:
            _add(num)

    # 3. Range mentions: "Articles 14 to 18"
    range_match = re.search(r"\b(?:articles?|art\.?s?)\s+(\d+)\s+(?:to|through|-)\s+(\d+)\b", query, re.IGNORECASE)
    if range_match:
        try:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if 1 <= start <= end <= 395 and end - start <= 10:
                for i in range(start, end + 1):
                    _add(str(i))
        except Exception:
            pass

    return res


def extract_clauses(query: str) -> list[str]:
    """Extract clause references (e.g. 19(1)(a), clause (2), etc.)."""
    clauses = []
    for m in re.finditer(r"\b(?:clause|sub-clause|cl\.?)\s*\(?([0-9a-z]+)\)?", query, re.IGNORECASE):
        clauses.append(m.group(1).lower())
    for m in re.finditer(r"\b19\s*\(\s*([1-6])\s*\)(?:\s*\(\s*([a-g])\s*\))?", query, re.IGNORECASE):
        cl = m.group(1)
        sub = m.group(2)
        if sub:
            clauses.append(f"19({cl})({sub.lower()})")
        else:
            clauses.append(f"19({cl})")
    return list(set(clauses))


# ---------------------------------------------------------------------------
# Constitutional Corpus Cache
# ---------------------------------------------------------------------------

_CONSTITUTION_CHUNKS_CACHE: list[dict] | None = None
_CASE_LAWS_CHUNKS_CACHE: list[dict] | None = None


def get_constitutional_chunks() -> list[dict]:
    """Load and cache constitutional chunks from data/processed/constitution_articles.jsonl."""
    global _CONSTITUTION_CHUNKS_CACHE
    if _CONSTITUTION_CHUNKS_CACHE is not None:
        return _CONSTITUTION_CHUNKS_CACHE

    project_root = Path(__file__).resolve().parent.parent
    const_file = project_root / "data" / "processed" / "constitution_articles.jsonl"
    chunks = []
    if const_file.exists():
        with open(const_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        chunks.append(json.loads(line))
                    except Exception:
                        pass

    # Also search legislation chunks.jsonl for any additional constitutional chunks
    leg_file = project_root / "data" / "chunks" / "legislation" / "chunks.jsonl"
    if leg_file.exists():
        with open(leg_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        c = json.loads(line)
                        if c.get("document_type") == "constitution" or "constitution" in (c.get("act_name") or "").lower():
                            # Don't add duplicate chunk_ids
                            if not any(x.get("chunk_id") == c.get("chunk_id") for x in chunks):
                                chunks.append(c)
                    except Exception:
                        pass

    _CONSTITUTION_CHUNKS_CACHE = chunks
    log.info(f"Cached {len(chunks)} constitutional chunks.")
    return _CONSTITUTION_CHUNKS_CACHE


def get_landmark_case_chunks() -> list[dict]:
    """Load and cache landmark constitutional cases from data/chunks/case_laws/indexed_chunks.jsonl."""
    global _CASE_LAWS_CHUNKS_CACHE
    if _CASE_LAWS_CHUNKS_CACHE is not None:
        return _CASE_LAWS_CHUNKS_CACHE

    project_root = Path(__file__).resolve().parent.parent
    case_file = project_root / "data" / "chunks" / "case_laws" / "indexed_chunks.jsonl"
    landmark_chunks = []

    target_titles = [
        "MANEKA GANDHI versus UNION OF INDIA",
        "A.K. GOPALAN versus THE STATE OF MADRAS",
        "E. P. ROYAPPA versus STATE OF TAMIL NADU",
        "THE STATE OF WEST BENGAL versus ANWAR ALI SARKAR",
        "ROMESH THAPPAR versus THE STATE OF MADRAS",
        "BENNETT COLEMAN & CO. (P) LTD versus PUNYA PRIYA DAS GUPTA",
        "OLGA TELLIS & ORS versus BOMBAY MUNICIPAL CORPORATION",
    ]

    if case_file.exists():
        with open(case_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                for target in target_titles:
                    if target in line_str:
                        try:
                            obj = json.loads(line_str)
                            landmark_chunks.append(obj)
                            break
                        except Exception:
                            pass

    _CASE_LAWS_CHUNKS_CACHE = landmark_chunks
    log.info(f"Cached {len(landmark_chunks)} landmark constitutional case chunks.")
    return _CASE_LAWS_CHUNKS_CACHE


# ---------------------------------------------------------------------------
# Multi-Article Constitutional Retrieval Router
# ---------------------------------------------------------------------------

class ConstitutionalRetriever:
    """
    High-precision retrieval router ensuring balanced, evidence-grounded
    constitutional responses.
    """

    def __init__(self, base_retriever: Any = None):
        self.base_retriever = base_retriever
        self.const_chunks = get_constitutional_chunks()
        self.case_chunks = get_landmark_case_chunks()
        self._cache: dict[tuple[str, int], tuple[list[dict], dict[str, Any]]] = {}

    def retrieve(
        self,
        query: str,
        top_k: int = 12,
    ) -> tuple[list[dict], dict[str, Any]]:
        """
        Execute balanced constitutional retrieval.
        Returns:
          - merged_documents: list of retrieved documents with citations
          - article_coverage: dictionary mapping articles to coverage details
        """
        cache_key = (query.strip().lower(), top_k)
        if cache_key in self._cache:
            import copy
            cached_docs, cached_cov = self._cache[cache_key]
            return copy.deepcopy(cached_docs), copy.deepcopy(cached_cov)

        is_gt = is_golden_triangle_query(query)
        articles = extract_constitutional_articles(query)
        clauses = extract_clauses(query)

        # Fallback to standard retriever if not a constitutional query
        if not is_constitutional_query(query) and not is_gt and not articles:
            if self.base_retriever is not None:
                docs = self.base_retriever.retrieve(query, top_k=top_k)
                return docs, {}
            return [], {}

        log.info(f"Executing Constitutional Retrieval (Golden Triangle={is_gt}, Articles={articles}, Clauses={clauses})")

        # 1. Exact article retrieval
        article_buckets: dict[str, list[dict]] = {art: [] for art in articles}
        if is_gt:
            for art in ["14", "19", "21"]:
                if art not in article_buckets:
                    article_buckets[art] = []

        all_const = get_constitutional_chunks()

        # Route matching chunks into buckets
        for c in all_const:
            c_art = str(c.get("article_number") or c.get("article") or "").upper()
            c_id = c.get("chunk_id", "")
            c_text = c.get("text", "")

            for target_art in article_buckets.keys():
                if target_art == c_art or f"_{target_art}" in c_id.lower() or f"Article {target_art}." in c_text:
                    if c not in article_buckets[target_art]:
                        article_buckets[target_art].append(c)

        # Golden triangle doctrine chunk
        gt_chunks = [c for c in all_const if "golden_triangle" in c.get("chunk_id", "")]

        # 2. Case laws matching
        # 2. Case laws matching with constitutional priority
        cases_pool = get_landmark_case_chunks()
        q_lower = query.lower()
        
        # Priority map for Golden Triangle / Constitutional cases
        case_priority_order = [
            "maneka gandhi",
            "a.k. gopalan",
            "royappa",
            "anwar ali sarkar",
            "romesh thappar",
            "olga tellis",
            "bennett coleman",
        ]
        
        retrieved_cases: list[dict] = []
        seen_case_titles = set()
        
        if is_gt:
            # Add cases by strict constitutional relevance priority
            for pkey in case_priority_order:
                for c in cases_pool:
                    title_lower = (c.get("title") or "").lower()
                    tkey = title_lower[:30]
                    if pkey in title_lower and tkey not in seen_case_titles:
                        retrieved_cases.append(c)
                        seen_case_titles.add(tkey)
                        break
        else:
            # Check specific query mentions first
            for alias, names in _LANDMARK_CONSTITUTIONAL_CASES.items():
                if alias in q_lower:
                    for c in cases_pool:
                        title_lower = (c.get("title") or "").lower()
                        tkey = title_lower[:30]
                        if any(n.lower() in title_lower for n in names) and tkey not in seen_case_titles:
                            retrieved_cases.append(c)
                            seen_case_titles.add(tkey)
                            break
            # If still room, add top constitutional precedents
            for pkey in case_priority_order:
                for c in cases_pool:
                    title_lower = (c.get("title") or "").lower()
                    tkey = title_lower[:30]
                    if pkey in title_lower and tkey not in seen_case_titles:
                        retrieved_cases.append(c)
                        seen_case_titles.add(tkey)
                        break

        # 3. Dense & BM25 retrieval from base retriever for surrounding context
        supplementary_docs: list[dict] = []
        if self.base_retriever is not None:
            try:
                raw_docs = self.base_retriever.retrieve(query, top_k=top_k)
                for d in raw_docs:
                    # Avoid duplicates
                    if not any(d.get("chunk_id") == x.get("chunk_id") for x in supplementary_docs):
                        supplementary_docs.append(d)
            except Exception as e:
                log.warning(f"Base retriever supplementary retrieval failed: {e}")

        # 4. Balanced Merging Strategy
        # To avoid one article dominating, guarantee equal slots for each target article
        merged_docs: list[dict] = []
        seen_cids = set()

        def add_chunk(chunk: dict, category: str = ""):
            cid = chunk.get("chunk_id") or chunk.get("document_id") or f"doc_{len(merged_docs)}"
            if cid in seen_cids:
                return
            seen_cids.add(cid)
            item = dict(chunk)
            item["citation_id"] = len(merged_docs) + 1
            if category:
                item["constitutional_category"] = category
            merged_docs.append(item)

        # A. Add Golden Triangle synthesis chunk first if GT query
        if is_gt:
            for c in gt_chunks:
                add_chunk(c, category="Golden Triangle Doctrine")

        # B. Add 2 chunks per target article (balanced representation)
        max_per_art = 2 if len(article_buckets) >= 3 else 3
        for art, chunks in article_buckets.items():
            for c in chunks[:max_per_art]:
                add_chunk(c, category=f"Article {art} Primary Text")

        # C. Add landmark cases (up to 5)
        for c in retrieved_cases[:5]:
            add_chunk(c, category="Landmark Constitutional Precedent")

        # D. Add connected articles (Article 13, 32, 226) if GT query
        if is_gt:
            connected = [c for c in all_const if c.get("article_number") in ["13", "32", "226"]]
            for c in connected[:3]:
                add_chunk(c, category="Connected Constitutional Provision")

        # E. Fill remaining slots from supplementary search (up to top_k)
        for d in supplementary_docs:
            if len(merged_docs) >= top_k:
                break
            add_chunk(d, category="Supplementary Legal Evidence")

        # 5. Compute Per-Article Coverage
        article_coverage = {}
        for art in ["14", "19", "21"]:
            chunks_found = [d for d in merged_docs if art == str(d.get("article_number") or d.get("article") or "").upper() or f"_{art}" in str(d.get("chunk_id", "")).lower() or f"Article {art}" in d.get("text", "")]
            cnt = len(chunks_found)
            found = cnt > 0
            conf = "high" if cnt >= 2 else ("medium" if cnt == 1 else "low")
            article_coverage[f"article_{art}"] = {
                "found": found,
                "evidence_count": cnt,
                "confidence": conf,
            }

        log.info(f"Constitutional Retrieval yielded {len(merged_docs)} balanced chunks. Coverage: {article_coverage}")
        if len(self._cache) >= 256:
            self._cache.pop(next(iter(self._cache)))
        import copy
        self._cache[cache_key] = (copy.deepcopy(merged_docs), copy.deepcopy(article_coverage))
        return merged_docs, article_coverage


# Global instance
_constitutional_retriever: ConstitutionalRetriever | None = None


def get_constitutional_retriever(base_retriever: Any = None) -> ConstitutionalRetriever:
    global _constitutional_retriever
    if _constitutional_retriever is None or base_retriever is not None:
        _constitutional_retriever = ConstitutionalRetriever(base_retriever=base_retriever)
    return _constitutional_retriever
