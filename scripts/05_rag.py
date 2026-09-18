#!/usr/bin/env python3
"""
scripts/05_rag.py
=================
LegalMind AI — Complete RAG Generation Pipeline

Loads the local Qwen3.6-35B-A3B model ONCE, then answers legal queries
by:
  1. Retrieving top-K relevant chunks via HybridRetriever
  2. Formatting a structured evidence-grounded prompt
  3. Generating a structured legal answer with Qwen3.6
  4. Extracting the final answer (suppressing internal <think> chains)
  5. Displaying answer with numbered citations

This script keeps the model loaded for interactive use. For server
deployment use app/main.py (FastAPI).

Usage:
    python scripts/05_rag.py
    python scripts/05_rag.py --query "Explain Article 21" --no-interactive
    python scripts/05_rag.py --gpu 1 --top-k 7
"""

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Add parent dir to sys.path so scripts/ imports work when called directly
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rag")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH      = "/home/sece2026-student12/LegalMindAI/models/Qwen3.6-35B-A3B"
DEFAULT_GPU     = 0           # GPU 0 (maps to physical GPU 1 via CUDA_VISIBLE_DEVICES)
DEFAULT_TOP_K   = 3
MAX_NEW_TOKENS  = 2048
TEMPERATURE     = 0.2         # Lower = more deterministic for legal Q&A
TOP_P           = 0.9

# ---------------------------------------------------------------------------
# System / RAG Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are LegalMind AI, an expert AI legal assistant specialising in Indian law.

YOU MUST ALWAYS PRODUCE A THOROUGH, DETAILED, AND FULLY COMPLETED LEGAL ANSWER. Never truncate or stop an explanation midway. If retrieved evidence is unavailable, answer comprehensively from your Indian legal knowledge and say so explicitly.

── EVIDENCE DISCLOSURE PROTOCOL ─────────────────────────────────────────
For EVERY factual or legal proposition you make, mark it with ONE of:
  [VERIFIED SOURCE]          — directly supported by a retrieved passage (cite [N])
  [GENERAL LEGAL KNOWLEDGE]  — from your training knowledge, not retrieved
  [MODEL INFERENCE]          — your reasoned interpretation from the law
  [USER PROVIDED]            — a fact stated by the user in the question

── STRICT RULES ─────────────────────────────────────────────────────────────
DO NOT:
  ✗ Stop or truncate an explanation midway — ensure every legal point, statutory test, and judgment is fully explained to its logical conclusion.
  ✗ Repeat the user's question in the answer
  ✗ Fabricate case names, citation numbers, section numbers, paragraph numbers, dates
  ✗ Mark unverified citations as verified
  ✗ Include retrieved chunks that are clearly unrelated to the question
  ✗ Produce empty sections — skip a section heading entirely if you have nothing to say
  ✗ Repeat the same disclaimer more than once
  ✗ Claim a legal provision exists unless you retrieved it or you are certain from training
  ✗ Present general knowledge as if it came from a retrieved document

DO:
  ✓ Provide thorough, detailed, and comprehensive legal analysis for all substantive legal inquiries.
  ✓ Read the Evidence Coverage line in the prompt to understand how much retrieved material is available
  ✓ Use [Source N] notation for every retrieved passage you draw on
  ✓ Distinguish clearly between verified and unverified information
  ✓ Answer from general legal knowledge when retrieval is insufficient, and say so
  ✓ Correct spelling mistakes and informal language silently — answer the intended question
  ✓ Ensure every sentence and legal section is complete, structured, and fully articulated.

── CONFIDENCE LABELS ──────────────────────────────────────────────────────────
Assign ONE confidence level in the ## Confidence Level section:
  HIGH confidence   — directly supported by verified retrieved sources
  MEDIUM confidence — partially supported; some parts rely on general legal knowledge
  LOW confidence    — source verification unavailable; answer from general knowledge only
  NEVER assign HIGH confidence when no relevant evidence was retrieved.

── REQUIRED OUTPUT FORMAT ──────────────────────────────────────────────────────
Use EXACTLY these Markdown headers. Omit a section only if it is genuinely empty:

## Answer
[Direct, comprehensive answer in clear legal language, thoroughly explaining the core principles.]

## Legal Position
[Relevant legal principles, statutes, constitutional provisions, or procedures. Mark each with evidence tags.]

## Relevant Authorities
[Case law, statutory provisions, constitutional articles. ONLY include authorities supported by retrieved evidence OR reliably known from legal training. Mark each [VERIFIED SOURCE] or [GENERAL LEGAL KNOWLEDGE]. If a citation cannot be verified, write: (Unverified — not found in retrieved corpus).]

## Application to the Question
[How the legal principles apply to the user’s specific facts. State clearly when facts are incomplete or disputed.]

## Practical Next Steps
[Procedural steps, documents to collect, questions to ask a lawyer, or issues to verify. Omit if not applicable.]

## Confidence Level
[HIGH / MEDIUM / LOW — with a one-sentence explanation.]

## Limitations
[State whether based on retrieved evidence, general knowledge, or both. ONE disclaimer only. Recommend professional advice for actual proceedings.]"""


# ---------------------------------------------------------------------------
# Broad Constitutional Query Management
# ---------------------------------------------------------------------------

CONSTITUTIONAL_TOPICS = [
    "Preamble",
    "Fundamental Rights",
    "Directive Principles of State Policy",
    "Fundamental Duties",
    "Union and State Government",
    "Judiciary",
    "Federal structure",
    "Elections and constitutional bodies",
    "Emergency provisions",
    "Constitutional amendments",
]

UNSUPPORTED_TOPIC_MESSAGE = "The retrieved evidence does not cover this topic."

CONSTITUTIONAL_TOPIC_PATTERNS = {
    "Preamble": [
        r"\bpreamble\b",
        r"\bwe,\s*the\s*people\s*of\s*india\b",
        r"\bsovereign\s+socialist\s+secular\b",
    ],
    "Fundamental Rights": [
        r"\bfundamental\s+rights?\b",
        r"\bpart\s+iii\b",
        r"\bpart\s+3\b",
        r"\barticle\s+(?:1[2-9]|2[0-9]|3[0-5])\b",
        r"\bright\s+to\s+(?:equality|freedom|life|education)\b",
        r"\barticles?\s+(?:14|19|21|21a|22|25|32)\b",
    ],
    "Directive Principles of State Policy": [
        r"\bdirective\s+principles?\b",
        r"\bdpsp\b",
        r"\bpart\s+iv\b",
        r"\bpart\s+4\b",
        r"\barticle\s+(?:3[6-9]|4[0-9]|5[0-1])\b",
        r"\buniform\s+civil\s+code\b",
    ],
    "Fundamental Duties": [
        r"\bfundamental\s+duties\b",
        r"\bpart\s+iv-?a\b",
        r"\barticle\s+51-?a\b",
    ],
    "Union and State Government": [
        r"\bunion\s+and\s+state\s+government\b",
        r"\bunion\s+executive\b",
        r"\bpresident\s+of\s+india\b",
        r"\bprime\s+minister\b",
        r"\bcouncil\s+of\s+ministers\b",
        r"\bgovernor\b",
        r"\bchief\s+minister\b",
        r"\bstate\s+legislature\b",
        r"\bparliament\s+of\s+india\b",
        r"\blok\s+sabha\b",
        r"\brajya\s+sabha\b",
        r"\bpart\s+v\b",
        r"\bpart\s+vi\b",
        r"\barticle\s+(?:5[2-9]|[6-9][0-9]|1[0-1][0-9]|15[2-9]|1[6-9][0-9]|2[0-1][0-3])\b",
    ],
    "Judiciary": [
        r"\bjudiciary\b",
        r"\bsupreme\s+court\b",
        r"\bhigh\s+courts?\b",
        r"\bsubordinate\s+courts?\b",
        r"\bjudicial\s+review\b",
        r"\bchief\s+justice\b",
        r"\barticle\s+(?:12[4-9]|13[0-9]|14[0-7]|21[4-9]|22[0-7])\b",
        r"\bwrit\s+jurisdiction\b",
    ],
    "Federal structure": [
        r"\bfederal\s+structure\b",
        r"\bfederalism\b",
        r"\bunion\s+of\s+states\b",
        r"\bcentre-state\b",
        r"\bcenter-state\b",
        r"\bseventh\s+schedule\b",
        r"\bunion\s+list\b",
        r"\bstate\s+list\b",
        r"\bconcurrent\s+list\b",
        r"\bdistribution\s+of\s+powers\b",
        r"\barticle\s+24[5-9]\b",
        r"\barticle\s+25[0-5]\b",
    ],
    "Elections and constitutional bodies": [
        r"\belections?\s+and\s+constitutional\s+bodies\b",
        r"\belection\s+commission\b",
        r"\bcomptroller\s+and\s+auditor\s+general\b",
        r"\bcag\b",
        r"\bunion\s+public\s+service\s+commission\b",
        r"\bupsc\b",
        r"\bfinance\s+commission\b",
        r"\battorney\s+general\b",
        r"\badvocate\s+general\b",
        r"\barticle\s+324\b",
        r"\barticle\s+148\b",
        r"\barticle\s+280\b",
        r"\barticle\s+315\b",
    ],
    "Emergency provisions": [
        r"\bemergency\s+provisions?\b",
        r"\bnational\s+emergency\b",
        r"\bpresident'?s\s+rule\b",
        r"\bstate\s+emergency\b",
        r"\bfinancial\s+emergency\b",
        r"\bproclamation\s+of\s+emergency\b",
        r"\bpart\s+xviii\b",
        r"\barticle\s+35[2-9]\b",
        r"\barticle\s+360\b",
    ],
    "Constitutional amendments": [
        r"\bconstitutional\s+amendments?\b",
        r"\bamendment\s+of\s+the\s+constitution\b",
        r"\bprocedure\s+for\s+amendment\b",
        r"\bpart\s+xx\b",
        r"\barticle\s+368\b",
        r"\bbasic\s+structure\b",
    ],
}


def is_broad_constitutional_query(query: str) -> bool:
    """
    Check if the query is a broad inquiry about the Indian Constitution.
    Specific article/section queries (e.g. Article 21, Section 302) return False.
    """
    if not query:
        return False
    q = query.lower().strip()

    # Direct provision queries with specific article or section numbers are NOT broad queries
    if re.search(r"\b(?:article|art\.?|section|sec\.?)\s*\d+[a-z]?\b", q):
        return False

    broad_indicators = [
        r"\bconstitutional\s+laws?\b",
        r"\bconstitution\s+of\s+india\b",
        r"\bindian\s+constitution\b",
        r"\bconstitution\b.*\b(?:india|indian)\b",
        r"\b(?:india|indian)\b.*\bconstitution\b",
        r"\bconstitutional\s+(?:framework|structure|scheme|system|provisions|principles|features)\b",
        r"\bexplain\s+the\s+constitution\b",
        r"\boverview\s+of\s+(?:the\s+)?(?:indian\s+)?constitution\b",
    ]
    return any(re.search(pat, q) for pat in broad_indicators)


def check_topic_support(topic: str, chunks: list[dict] | None) -> bool:
    """Check if a specific constitutional topic has supporting retrieved evidence."""
    if not chunks:
        return False
    patterns = CONSTITUTIONAL_TOPIC_PATTERNS.get(topic, [])
    for chunk in chunks:
        text = chunk.get("text", "")
        title = chunk.get("title", "")
        act_name = chunk.get("act_name", "")
        content = f"{title} {act_name} {text}"
        if any(re.search(p, content, re.IGNORECASE) for p in patterns):
            return True
    return False


# ---------------------------------------------------------------------------
# Multi-turn Conversation & Continuation Helpers
# ---------------------------------------------------------------------------

def is_continuation_query(query: str) -> bool:
    """
    Returns True if the query is an explicit or implicit request to continue,
    proceed, elaborate, or go further into the existing conversation.
    """
    if not query:
        return False
    q = query.strip().lower()
    q = re.sub(r"[.!?,:;\"']+$", "", q).strip()

    exact_matches = {
        "continue", "continue please", "please continue", "continue...",
        "continue this", "continue that", "continue the conversation",
        "continue discussion", "continue answer", "continue response",
        "go on", "go on please", "please go on",
        "keep going", "keep going please",
        "proceed", "proceed please", "please proceed",
        "carry on", "carry on please",
        "tell me more", "tell me more details",
        "elaborate", "elaborate please", "please elaborate",
        "elaborate further", "expand", "expand further", "expand on this",
        "more details", "what else", "what next", "next", "more",
        "can you continue", "could you continue", "continue further",
        "keep explaining", "continue explaining",
    }
    if q in exact_matches:
        return True

    patterns = [
        r"^continue\b",
        r"\bcontinue\b.*(?:conversation|discussion|explanation|points|analysis|above|this)",
        r"^please continue\b",
        r"^go on\b",
        r"^keep going\b",
        r"^proceed\b",
        r"^carry on\b",
        r"^(?:can|could) you (?:please )?(?:continue|elaborate|expand|go on)\b",
        r"^(?:please )?(?:elaborate|expand)(?: further| on this)?\b",
        r"^tell me more\b",
    ]
    for pat in patterns:
        if re.search(pat, q):
            return True
    return False


def is_short_followup_query(query: str) -> bool:
    """
    Returns True if the query appears to be a context-dependent short follow-up
    e.g. 'what are the exceptions?', 'why?', 'what about section 45(1)?', 'cite more cases'.
    """
    if not query:
        return False
    q = query.strip().lower()
    words = q.split()
    if len(words) <= 5:
        followup_cues = [
            r"\b(?:why|how|what about|and|then|exceptions?|cases?|judgments?|examples?|provisos?)\b",
            r"\b(?:it|this|that|these|those|them)\b",
        ]
        if any(re.search(cue, q) for cue in followup_cues):
            return True
    return False


def _fetch_recent_history_from_db(query: str) -> list[dict]:
    """
    Fallback context recovery: if conversation_history was not passed directly,
    look up the most recent conversation in the chat store where the current query
    was committed as the latest user message.
    """
    try:
        from storage import chat_store
        db_p = chat_store.db_path()
        if not db_p.exists():
            return []

        import sqlite3
        con = sqlite3.connect(f"file:{db_p}?mode=ro", uri=True)
        cur = con.cursor()
        # Find the conversation whose most recent user message matches `query`
        cur.execute(
            "SELECT conversation_id, id FROM messages WHERE role = 'user' AND content = ? ORDER BY created_at DESC LIMIT 1",
            (query,)
        )
        row = cur.fetchone()
        if not row:
            cur.execute(
                "SELECT conversation_id, id FROM messages WHERE role = 'user' ORDER BY created_at DESC LIMIT 1"
            )
            row = cur.fetchone()

        if not row:
            con.close()
            return []

        conv_id, current_msg_id = row
        cur.execute(
            "SELECT role, content, status FROM messages WHERE conversation_id = ? AND id != ? ORDER BY created_at ASC",
            (conv_id, current_msg_id)
        )
        rows = cur.fetchall()
        con.close()

        history = []
        for r_role, r_content, r_status in rows:
            if r_content and r_status != "error":
                history.append({"role": r_role, "content": r_content})
        return history
    except Exception as e:
        log.warning(f"Could not recover conversation history from DB: {e}")
        return []


def extract_conversation_context(query: str, history: list[dict] | None) -> tuple[str, str, str]:
    """
    Extracts:
      (substantive_topic_query, previous_assistant_text, rewritten_retrieval_query)
    from conversation history if query is a continuation or short follow-up.
    """
    if not history:
        return query, "", query

    turns = [m for m in history if isinstance(m, dict) and m.get("content")]
    if not turns:
        return query, "", query

    # Find the most recent substantive user query (not a continuation itself)
    last_user_query = ""
    for m in reversed(turns):
        if m.get("role") == "user":
            c = m.get("content", "").strip()
            if c and not is_continuation_query(c) and len(c.split()) >= 3:
                last_user_query = c
                break

    if not last_user_query:
        for m in turns:
            if m.get("role") == "user" and m.get("content", "").strip():
                last_user_query = m["content"].strip()
                break

    last_assistant_response = ""
    for m in reversed(turns):
        if m.get("role") == "assistant":
            c = m.get("content", "").strip()
            if c and "interrupted" not in c and "could not be generated" not in c:
                last_assistant_response = c
                break

    is_cont = is_continuation_query(query)
    is_flw = is_short_followup_query(query)

    if (is_cont or is_flw) and last_user_query:
        if is_cont:
            cleaned_q = re.sub(r"\b(continue|please|go on|elaborate|further)\b", "", query, flags=re.I).strip()
            if cleaned_q:
                retrieval_query = f"{last_user_query} {cleaned_q}"
            else:
                retrieval_query = f"{last_user_query} detailed statutory analysis exceptions landmark judgments"
        else:
            retrieval_query = f"{last_user_query} {query}"

        return last_user_query, last_assistant_response, retrieval_query

    return query, last_assistant_response, query


# ---------------------------------------------------------------------------
# Chunk Relevance Filtering
# ---------------------------------------------------------------------------

def _filter_relevant_chunks(
    query: str,
    chunks: list[dict],
    min_overlap: int = 1,
    max_chunks: int = 6,
) -> list[dict]:
    """
    Remove chunks that are clearly unrelated to the query.

    Scoring:
      - Keyword overlap between query tokens and chunk text.
      - Chunks with zero overlap are excluded unless there are no others.
      - Returns up to `max_chunks` chunks ordered by descending relevance.

    This prevents random statutory sections (e.g., BNSS attachment clauses)
    from polluting the context when the user asks about bail or property law.
    """
    if not chunks:
        return []

    # If chunks come from ConstitutionalRetriever or are dedicated constitutional chunks, preserve them
    has_const = any(
        c.get("constitutional_category") or c.get("document_type") == "constitution" or "const_" in str(c.get("chunk_id", ""))
        for c in chunks
    )
    if has_const:
        return chunks[:max_chunks]

    # Tokenise query into meaningful keywords (strip stop words, preserve numbers like 14, 19, 21, 45, 302)
    _STOP = {
        "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at",
        "to", "for", "with", "by", "from", "and", "or", "not", "be", "it",
        "what", "which", "how", "why", "when", "who", "does", "do", "can",
        "please", "tell", "me", "explain", "about", "under", "act", "law",
        "legal", "court",
    }
    q_tokens = {
        t for t in re.sub(r"[^a-z0-9\s]", " ", query.lower()).split()
        if t and t not in _STOP and (len(t) >= 3 or any(c.isdigit() for c in t))
    }

    scored: list[tuple[int, dict]] = []
    for chunk in chunks:
        blob = " ".join([
            chunk.get("text", ""),
            chunk.get("title", ""),
            chunk.get("act_name", ""),
            chunk.get("section", ""),
            chunk.get("article", ""),
        ]).lower()
        score = sum(1 for t in q_tokens if t in blob)
        scored.append((score, chunk))

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)

    # Keep chunks with at least min_overlap keyword hits; if none qualify keep top-3 anyway
    qualifying = [(s, c) for s, c in scored if s >= min_overlap]
    if not qualifying:
        qualifying = scored[:3]  # fallback: top-3 even if zero keyword overlap

    return [c for _, c in qualifying[:max_chunks]]


def build_rag_prompt(
    query: str,
    chunks: list[dict],
    citations: list[str],
    is_continuation: bool = False,
    last_user_query: str = "",
    last_assistant_response: str = "",
    knowledge_mode: str = "evidence_based",
) -> str:
    """Build the RAG prompt with retrieved evidence blocks.

    Filters chunks for relevance to the query before building the prompt so
    that unrelated statutory sections do not contaminate the evidence context.
    """
    # ── 1. Filter for relevance ──────────────────────────────────────────────────
    relevant_chunks = _filter_relevant_chunks(query, chunks)
    n_relevant = len(relevant_chunks)
    n_total = len(chunks)

    # ── 2. Build evidence blocks ───────────────────────────────────────────────
    evidence_blocks = []
    for chunk in relevant_chunks:
        cid    = chunk.get("citation_id", "?")
        source = chunk.get("source", "Unknown")
        title  = chunk.get("title", "Untitled")
        court  = chunk.get("court", "")
        date   = chunk.get("date", "")
        text   = chunk.get("text", "")

        header = f"[Source {cid}]"
        if court:
            header += f" {title} | {court}"
        else:
            header += f" {title}"
        if date:
            header += f" | {date}"
        header += f" (file: {source})"

        evidence_blocks.append(f"{header}\n{text[:850]}")

    evidence_section = "\n\n---\n\n".join(evidence_blocks) if evidence_blocks else "(No relevant passages retrieved.)"

    # ── 3. Evidence Coverage signal ───────────────────────────────────────────
    if n_relevant == 0:
        coverage_line = "Evidence Coverage: 0 relevant passages retrieved — answer from GENERAL LEGAL KNOWLEDGE and clearly state this."
        km_directive = "EVIDENCE INSTRUCTION: No relevant retrieved evidence exists for this query. You MUST answer using your general Indian legal knowledge. Mark all propositions [GENERAL LEGAL KNOWLEDGE]. Assign LOW confidence."
    elif n_relevant <= 2:
        coverage_line = f"Evidence Coverage: {n_relevant} partially relevant passage(s) retrieved (of {n_total} total)."
        km_directive = "EVIDENCE INSTRUCTION: Limited evidence retrieved. Answer what the retrieved passages support, mark those [VERIFIED SOURCE]. For gaps, use general legal knowledge and mark [GENERAL LEGAL KNOWLEDGE]. Assign MEDIUM confidence."
    else:
        coverage_line = f"Evidence Coverage: {n_relevant} relevant passage(s) retrieved (of {n_total} total)."
        km_directive = "EVIDENCE INSTRUCTION: Sufficient evidence retrieved. Prioritise retrieved passages, cite with [Source N]. Mark each proposition [VERIFIED SOURCE] or [GENERAL LEGAL KNOWLEDGE] as appropriate."

    # ── 4. Continuation / follow-up instruction ──────────────────────────────
    continuation_instruction = ""
    if is_continuation and last_user_query:
        summary_snippet = ""
        if last_assistant_response:
            clean_prior = strip_thinking(last_assistant_response).strip()
            summary_snippet = f"\nSUMMARY OF PREVIOUS ANSWER GIVEN:\n{clean_prior[:800]}...\n"

        continuation_instruction = f"""

SPECIAL INSTRUCTION FOR CONTINUATION:
The user previously asked: "{last_user_query}".
The user now says: "{query}" (requesting to continue / expand the analysis).{summary_snippet}
CRITICAL CONTINUATION DIRECTIVES:
1. CONTINUATION, NOT DEFINITION: Do NOT define the English word "continue" or discuss unrelated statutory contexts of the word "continue" (e.g. continuing attachment or continuing inquiries).
2. DEEPER SUBSTANTIVE ANALYSIS: Seamlessly continue the legal discussion on "{last_user_query}". Dive deeper into:
   - Specific statutory provisions, statutory exceptions, and provisos (e.g. Section 45(1) provisos under PMLA: women, persons under 16 years, sick or infirm, or money laundering under Rs. 1 Crore).
   - Landmark and recent Supreme Court judgments (e.g. Vijay Madanlal Choudhary v. Union of India (2022), Manish Sisodia v. ED (2024), Prem Prakash v. ED (2024), Ram Kripal Meena v. ED (2024), V. Senthil Balaji v. ED (2024)).
   - Interplay with constitutional guarantees: explain how Article 21 (right to personal liberty and speedy trial) operates when an undertrial faces prolonged incarceration without trial, overriding the Section 45 twin conditions.
   - Practical legal consequences and how the Special Court / High Court assesses "reasonable grounds for believing the accused is not guilty".
3. GROUNDING: Attribute evidence with [Source N] tags wherever supported.
"""

    # ── 5. Broad constitutional / Golden Triangle instruction ──────────────────
    broad_instruction = ""
    if is_broad_constitutional_query(query):
        broad_instruction = f"""

SPECIAL INSTRUCTION FOR BROAD CONSTITUTIONAL QUERIES:
The user is asking a broad constitutional query. You MUST organize your ## Answer section into EXACTLY these 10 topic subheadings:
### Preamble
### Fundamental Rights
### Directive Principles of State Policy
### Fundamental Duties
### Union and State Government
### Judiciary
### Federal structure
### Elections and constitutional bodies
### Emergency provisions
### Constitutional amendments

GROUNDING RULES:
1. Provide substantive legal analysis ONLY for topics supported by the retrieved evidence.
2. If any topic is not present in the retrieved evidence, you MUST write under that heading:
   "{UNSUPPORTED_TOPIC_MESSAGE}"
3. Do NOT claim unsupported topics as evidence-backed.
4. If some constitutional parts are not covered in the retrieved evidence, assign MEDIUM or LOW confidence (NEVER HIGH).
"""
    else:
        try:
            from constitutional_retrieval import is_golden_triangle_query
            if is_golden_triangle_query(query):
                broad_instruction = f"""

SPECIAL INSTRUCTION FOR GOLDEN TRIANGLE CONSTITUTIONAL QUERIES (ARTICLES 14, 19, 21):
The user is inquiring about the Golden Triangle of the Indian Constitution (Articles 14, 19, and 21), their interconnected doctrine, reasonable restrictions, and landmark Supreme Court judgments.
You MUST organize your substantive legal analysis in a clear legal order:
1. Direct Answer & Comparative Overview of Article 14, Article 19, and Article 21.
2. Distinct Scope & Individual Differences:
   - Article 14 (Equality before the law & equal protection; applies to all persons; intelligible differentia & non-arbitrariness).
   - Article 19 (Six fundamental freedoms; applies to citizens only; subject to enumerated reasonable restrictions under 19(2)-(6)).
   - Article 21 (Protection of life & personal liberty; applies to all persons; procedural & substantive due process).
3. The Interconnection & The Golden Triangle Doctrine:
   - Departure from the mutual exclusivity / siloed doctrine of A.K. Gopalan v. State of Madras (1950).
   - The landmark synthesis established in Maneka Gandhi v. Union of India (1978): Articles 14, 19, and 21 form a constitutional trinity / tapestry. A law depriving personal liberty under Article 21 must satisfy the non-arbitrariness test of Article 14 and the reasonableness test of Article 19.
4. Reasonable Restrictions:
   - Specific heads under Articles 19(2) through 19(6) (sovereignty, security, public order, decency, morality, contempt of court, etc.).
   - The test of reasonable classification under Article 14.
   - The "just, fair, and reasonable" standard governing procedure established by law under Article 21.
5. Landmark Supreme Court Precedents:
   - Maneka Gandhi v. Union of India (1978)
   - A.K. Gopalan v. State of Madras (1950)
   - E.P. Royappa v. State of Tamil Nadu (1974)
   - State of West Bengal v. Anwar Ali Sarkar (1952)
   - Romesh Thappar v. State of Madras (1950)
6. Grounding & Citations:
   - Attribute and cite retrieved evidence with [Source N] tags.
"""
        except Exception:
            pass

    # ── 6. Query header ────────────────────────────────────────────────────
    query_header = f"""USER QUERY:
{query}"""
    if is_continuation and last_user_query:
        query_header = f"""USER QUERY (CONTINUATION OF PREVIOUS TOPIC):
Previous Topic: "{last_user_query}"
Current Instruction: "{query}" (Continue and expand the legal analysis)"""

    return f"""{query_header}

{coverage_line}

{km_directive}

RETRIEVED LEGAL EVIDENCE:
{evidence_section}

CITATION REFERENCE LIST:
{chr(10).join(citations) if citations else '(none)'}{continuation_instruction}{broad_instruction}

Provide a structured legal answer using the EXACT headers from your system instructions."""



def strip_thinking(text: str) -> str:
    """
    Remove <think>…</think> blocks that Qwen3 may emit.
    Returns only the final answer portion.
    If stripping results in an empty string, returns the original text as-is
    so we never produce an empty response.
    """
    if not text:
        return text
    # Remove thinking tags (Qwen3 chain-of-thought)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<\|think\|>.*?<\|/think\|>", "", cleaned, flags=re.DOTALL)
    # Also strip unclosed think blocks that reach end of string
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    # Safety: if stripping left nothing, return the raw text so the answer is never empty
    if not cleaned:
        # Fall back to whatever came after the last </think> tag
        parts = re.split(r"</think>", text, flags=re.DOTALL)
        if len(parts) > 1:
            cleaned = parts[-1].strip()
        # Final fallback: return the raw text (the model produced only a think block)
        return cleaned if cleaned else text.strip()
    return cleaned


# ---------------------------------------------------------------------------
# Model loader (singleton pattern)
# ---------------------------------------------------------------------------

DEFAULT_ADAPTER_DIR = str(Path(__file__).resolve().parent.parent / "fine_tuning" / "adapters" / "lora-legal-v2")


class LegalMindModel:
    _instance = None

    def __init__(self, gpu: int, adapter_path: Optional[str] = None):
        self.gpu_id = gpu
        self.device = f"cuda:{gpu}"
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        import threading
        self._generate_lock = threading.Lock()

        log.info(f"Loading processor from {MODEL_PATH}")
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)

        log.info(f"Loading Qwen3.6-35B-A3B exclusively on {self.device} (BF16)…")
        t0 = time.time()
        from transformers import Qwen3_5MoeForConditionalGeneration

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": {"": self.device},
            "attn_implementation": "sdpa",
            "local_files_only": True,
            "trust_remote_code": True,
        }

        self.base_model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            **load_kwargs,
        )
        self.base_model.eval()

        # Force eager MoE experts to prevent 93GB memory spike caused by batched_mm weight expansion
        try:
            if hasattr(self.base_model, "set_experts_implementation"):
                self.base_model.set_experts_implementation("eager")
                log.info("MoE experts implementation set to 'eager' (prevents batched_mm memory explosion).")
        except Exception as e:
            log.warning(f"Could not set eager experts implementation: {e}")

        self.model = self.base_model
        elapsed = time.time() - t0
        log.info(f"Base model loaded in {elapsed:.1f}s exclusively on {self.device}")

        # Adapter status tracking
        self.adapter_attached = False
        self.adapter_name = None
        self.adapter_path = None
        self.adapter_status = "NONE"

        # Attempt to load adapter (fallback safely to Base + RAG if corrupt or absent)
        target_adapter = adapter_path if adapter_path is not None else os.getenv("LEGALMIND_ADAPTER_PATH", DEFAULT_ADAPTER_DIR)
        if target_adapter and os.path.exists(target_adapter):
            self.attach_adapter(target_adapter)
        else:
            log.info("No LoRA adapter configured. Running Base Model + RAG.")

    @classmethod
    def get_instance(cls, gpu: int = DEFAULT_GPU, adapter_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = cls(gpu, adapter_path=adapter_path)
        return cls._instance

    def attach_adapter(self, adapter_path: str) -> bool:
        """
        Audit and attach standard LoRA adapter.
        Guarantees automatic fallback to Base Model + RAG if corrupt or contains NaNs.
        """
        p = Path(adapter_path).resolve()
        if not p.is_dir():
            log.warning(f"LoRA adapter directory not found: {p}. Continuing with Base Model + RAG.")
            self.adapter_status = "NOT_FOUND"
            return False

        safetensors_file = p / "adapter_model.safetensors"
        if not safetensors_file.exists():
            log.warning(f"LoRA safetensors file not found at {safetensors_file}. Continuing with Base Model + RAG.")
            self.adapter_status = "FILE_MISSING"
            return False

        # Pre-load forensic safety audit
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fine_tuning"))
            from validate_adapter import inspect_safetensors_file
            is_valid, report = inspect_safetensors_file(safetensors_file)
            if not is_valid or report.get("nan_tensors_count", 0) > 0 or report.get("inf_tensors_count", 0) > 0:
                log.error(
                    f"CRITICAL: LoRA adapter at {p} failed safety audit: "
                    f"status={report.get('status')}, NaNs={report.get('nan_tensors_count')}, Infs={report.get('inf_tensors_count')}. "
                    "Refusing to load corrupt adapter. AUTOMATICALLY FALLING BACK TO BASE MODEL + RAG."
                )
                self.adapter_status = "CORRUPT_FALLBACK_BASE"
                return False
        except Exception as e:
            log.error(f"Error auditing LoRA adapter {p}: {e}. Falling back to Base Model + RAG.")
            self.adapter_status = f"AUDIT_ERROR_FALLBACK_BASE: {e}"
            return False

        # Load with PEFT
        try:
            from peft import PeftModel
            log.info(f"Attaching LoRA adapter from {p}...")
            peft_m = PeftModel.from_pretrained(self.base_model, str(p))
            peft_m.eval()

            # In-memory NaN check
            for name, param in peft_m.named_parameters():
                if "lora_" in name:
                    if torch.isnan(param.data).any().item() or torch.isinf(param.data).any().item():
                        log.error(f"CRITICAL: In-memory LoRA tensor '{name}' contains NaN/Inf! Rolling back to Base Model.")
                        self.model = self.base_model
                        self.adapter_attached = False
                        self.adapter_status = "CORRUPT_IN_MEMORY_FALLBACK_BASE"
                        return False

            self.model = peft_m
            self.adapter_attached = True
            self.adapter_name = p.name
            self.adapter_path = str(p)
            self.adapter_status = "ACTIVE"
            log.info(f"LoRA adapter '{self.adapter_name}' attached and verified clean (0 NaNs/Infs).")
            return True
        except Exception as e:
            log.error(f"Failed to attach LoRA adapter {p}: {e}. Falling back to Base Model + RAG.")
            self.model = self.base_model
            self.adapter_attached = False
            self.adapter_status = f"LOAD_ERROR_FALLBACK_BASE: {e}"
            return False

    def detach_adapter(self):
        """Detach LoRA adapter and revert to Base Model."""
        self.model = self.base_model
        self.adapter_attached = False
        self.adapter_status = "DETACHED"
        log.info("LoRA adapter detached. Using Base Model + RAG.")

    def is_lora_active(self) -> bool:
        """Returns True if a LoRA adapter is currently active."""
        return self.adapter_attached and self.adapter_status == "ACTIVE"

    def _prepare_chat_inputs(self, messages: list[dict], enable_thinking: bool = False) -> dict:
        """
        Format chat messages into tokenized model inputs, ensuring:
        1. The thinking tag is completely closed (<think>\n\n</think>\n\n) so the model
           does NOT waste 300+ tokens generating internal reasoning chains that get discarded.
        2. Tensor is placed on the appropriate model device.
        """
        try:
            prompt_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except Exception:
            prompt_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

        # Enforce closed thinking tag to immediately suppress chain-of-thought generation
        if prompt_text.endswith("<think>\n"):
            prompt_text = prompt_text[:-len("<think>\n")] + "<think>\n\n</think>\n\n"
        elif prompt_text.endswith("<think>"):
            prompt_text = prompt_text[:-len("<think>")] + "<think>\n\n</think>\n\n"
        elif "<|im_start|>assistant" in prompt_text:
            last_turn = prompt_text.split("<|im_start|>assistant")[-1]
            if "<think>" in last_turn and "</think>" not in last_turn:
                prompt_text = prompt_text + "\n</think>\n\n"
            elif "</think>" not in last_turn:
                prompt_text = prompt_text + "<think>\n\n</think>\n\n"

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            return_dict=True,
        )

        model_device = getattr(self.model, "device", None)
        if model_device is None:
            model_device = next(self.model.parameters()).device

        return {
            k: v.to(model_device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

    def _get_stop_token_ids(self) -> list[int]:
        """
        Return all valid EOS / Stop token IDs so generation stops immediately
        when the model finishes the turn (<|im_end|>, <|endoftext|>).
        """
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        stop_tokens = [248046, 248044]  # <|im_end|>, <|endoftext|>
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            if isinstance(tokenizer.eos_token_id, list):
                stop_tokens.extend(tokenizer.eos_token_id)
            else:
                stop_tokens.append(tokenizer.eos_token_id)
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            try:
                im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
                if im_end is not None and isinstance(im_end, int):
                    stop_tokens.append(im_end)
            except Exception:
                pass
        return list(set(tid for tid in stop_tokens if tid is not None and isinstance(tid, int)))

    def generate(
        self,
        query: str,
        retrieved_chunks: list[dict],
        citations: list[str],
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        conversation_history: list[dict] | None = None,
        is_continuation: bool = False,
        last_user_query: str = "",
        last_assistant_response: str = "",
    ) -> str:
        """Run RAG generation for a single query with optional conversation history."""

        rag_prompt = build_rag_prompt(
            query,
            retrieved_chunks,
            citations,
            is_continuation=is_continuation,
            last_user_query=last_user_query,
            last_assistant_response=last_assistant_response,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if conversation_history:
            for m in conversation_history[-4:]:
                r = m.get("role")
                c = m.get("content", "").strip()
                if r in ("user", "assistant") and c:
                    if r == "assistant" and len(c) > 1500:
                        c = c[:1500] + "..."
                    messages.append({"role": r, "content": c})

        messages.append({"role": "user", "content": rag_prompt})

        inputs = self._prepare_chat_inputs(messages)
        stop_token_ids = self._get_stop_token_ids()
        pad_token_id = getattr(getattr(self.processor, "tokenizer", self.processor), "eos_token_id", 248044) or 248044

        with self._generate_lock:
            try:
                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        top_p=top_p,
                        use_cache=True,
                        eos_token_id=stop_token_ids,
                        pad_token_id=pad_token_id,
                    )

                input_length = inputs["input_ids"].shape[1]
                new_tokens_count = generated_ids.shape[1] - input_length
                new_ids = generated_ids[:, input_length:]

                response = self.processor.batch_decode(
                    new_ids,
                    skip_special_tokens=True,
                )[0]
                response = strip_thinking(response)

                # Check if generation hit token limit before reaching an EOS stop token
                last_token_id = new_ids[0, -1].item() if new_ids.shape[1] > 0 else None
                if new_tokens_count >= max_new_tokens and last_token_id not in stop_token_ids:
                    log.info(f"Answer reached max_new_tokens ({max_new_tokens}) before EOS; completing legal analysis...")
                    cont_messages = messages + [
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": "Please complete your legal analysis. Finish any open sentence, state landmark Supreme Court authorities, and summarize the practical conclusions."}
                    ]
                    try:
                        cont_inputs = self._prepare_chat_inputs(cont_messages)
                        with torch.inference_mode():
                            cont_ids = self.model.generate(
                                **cont_inputs,
                                max_new_tokens=384,
                                do_sample=True if temperature > 0 else False,
                                temperature=temperature,
                                top_p=top_p,
                                use_cache=True,
                                eos_token_id=stop_token_ids,
                                pad_token_id=pad_token_id,
                            )
                        c_len = cont_inputs["input_ids"].shape[1]
                        cont_text = self.processor.batch_decode(cont_ids[:, c_len:], skip_special_tokens=True)[0]
                        response = response + "\n\n" + strip_thinking(cont_text)
                    except Exception as cont_err:
                        log.warning(f"Auto-completion step failed: {cont_err}")

                response = response.strip()
                if response and not response.endswith((".", "!", "?", "\n", "]", ")", '"')):
                    if response.endswith("[VERIFIED"):
                        response += " SOURCE].\n"
                    elif response.endswith("[VERIFIED "):
                        response += "SOURCE].\n"
                    elif response.endswith("[GENERAL LEGAL"):
                        response += " KNOWLEDGE].\n"
                    elif response.endswith("["):
                        response += "VERIFIED SOURCE].\n"
                    elif response.endswith("[VERIFIED SOURC"):
                        response += "E].\n"
                    else:
                        response += ".\n"

                # Safety: if response is empty after stripping, retry without think suppression
                if not response.strip():
                    log.warning("Response was empty after strip_thinking; retrying without think suppression...")
                    try:
                        messages_retry = messages.copy()
                        inputs_retry = self._prepare_chat_inputs(messages_retry, enable_thinking=True)
                        with torch.inference_mode():
                            retry_ids = self.model.generate(
                                **inputs_retry,
                                max_new_tokens=min(max_new_tokens, 512),
                                do_sample=False,
                                use_cache=True,
                                eos_token_id=stop_token_ids,
                                pad_token_id=pad_token_id,
                            )
                        r_len = inputs_retry["input_ids"].shape[1]
                        retry_text = self.processor.batch_decode(
                            retry_ids[:, r_len:], skip_special_tokens=True
                        )[0]
                        retry_text = strip_thinking(retry_text).strip()
                        if retry_text:
                            log.info("Retry without think suppression produced a valid response.")
                            return retry_text + "\n"
                    except Exception as retry_empty_err:
                        log.warning(f"Empty-response retry also failed: {retry_empty_err}")
                    # If all retries fail, return a graceful fallback
                    return "I was unable to generate a complete legal answer for your query. Please rephrase your question or try again.\n"

                return response
            except torch.OutOfMemoryError as oom_err:
                log.warning(f"CUDA OOM encountered during generation: {oom_err}. Clearing cache and retrying with reduced tokens...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    with torch.inference_mode():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=min(max_new_tokens, 256),
                            do_sample=False,
                            use_cache=True,
                            eos_token_id=stop_token_ids,
                            pad_token_id=pad_token_id,
                        )
                    input_length = inputs["input_ids"].shape[1]
                    generated_ids = generated_ids[:, input_length:]
                    response = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )[0]
                    return strip_thinking(response)
                except Exception as retry_err:
                    log.error(f"Generation failed after OOM retry: {retry_err}")
                    raise oom_err
            finally:
                try:
                    del inputs
                except Exception:
                    pass
                if "generated_ids" in locals():
                    del generated_ids

    def generate_stream(
        self,
        query: str,
        retrieved_chunks: list[dict],
        citations: list[str],
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        conversation_history: list[dict] | None = None,
        is_continuation: bool = False,
        last_user_query: str = "",
        last_assistant_response: str = "",
    ):
        """
        Stream tokens in real time using TextIteratorStreamer.
        Yields decoded text chunks as they are generated by the model.
        """
        from transformers import TextIteratorStreamer

        rag_prompt = build_rag_prompt(
            query,
            retrieved_chunks,
            citations,
            is_continuation=is_continuation,
            last_user_query=last_user_query,
            last_assistant_response=last_assistant_response,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if conversation_history:
            for m in conversation_history[-4:]:
                r = m.get("role")
                c = m.get("content", "").strip()
                if r in ("user", "assistant") and c:
                    if r == "assistant" and len(c) > 1500:
                        c = c[:1500] + "..."
                    messages.append({"role": r, "content": c})

        messages.append({"role": "user", "content": rag_prompt})

        inputs = self._prepare_chat_inputs(messages)
        stop_token_ids = self._get_stop_token_ids()
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        pad_token_id = getattr(tokenizer, "eos_token_id", 248044) or 248044

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
            eos_token_id=stop_token_ids,
            pad_token_id=pad_token_id,
        )

        def _worker():
            try:
                with torch.inference_mode():
                    self.model.generate(**generation_kwargs)
            except Exception as e:
                log.error(f"Stream generation error in worker thread: {e}")

        gen_thread = threading.Thread(target=_worker)

        with self._generate_lock:
            gen_thread.start()
            inside_think = False
            accumulated_chunks = []
            try:
                for new_text in streamer:
                    if "<think>" in new_text:
                        inside_think = True
                        new_text = new_text.split("<think>")[0]
                    if "</think>" in new_text:
                        inside_think = False
                        new_text = new_text.split("</think>")[-1]
                    if not inside_think and new_text:
                        accumulated_chunks.append(new_text)
                        yield new_text
            finally:
                gen_thread.join()

            # Safety: if the model only produced a think block and nothing was yielded,
            # emit a fallback so the stream is never completely empty.
            if not accumulated_chunks:
                fallback = "I was unable to generate a complete legal answer for your query. Please rephrase your question or try again.\n"
                log.warning("Stream produced no output (only think tokens); emitting fallback.")
                accumulated_chunks.append(fallback)
                yield fallback

            full_gen = "".join(accumulated_chunks).rstrip()
            if full_gen and not full_gen.endswith((".", "!", "?", "\n", "]", ")", '"')):
                if full_gen.endswith("[VERIFIED"):
                    patch = " SOURCE].\n"
                    yield patch
                elif full_gen.endswith("[VERIFIED "):
                    patch = "SOURCE].\n"
                    yield patch
                elif full_gen.endswith("[GENERAL LEGAL"):
                    patch = " KNOWLEDGE].\n"
                    yield patch
                elif full_gen.endswith("["):
                    patch = "VERIFIED SOURCE].\n"
                    yield patch
                elif full_gen.endswith("[VERIFIED SOURC"):
                    patch = "E].\n"
                    yield patch
                else:
                    patch = ".\n"
                    yield patch

    def generate_chat_completion(
        self,
        messages: list[dict],
        max_new_tokens: int = 450,
        temperature: float = 0.5,
        top_p: float = 0.9,
    ) -> str:
        """
        Direct chat completion interface for specialized agents (e.g. Moot Court Judge Mode).
        Thread-safe across concurrent calls with early EOS termination,
        KV-cache reuse, and thinking suppression.
        """
        inputs = self._prepare_chat_inputs(messages)
        stop_token_ids = self._get_stop_token_ids()
        pad_token_id = getattr(getattr(self.processor, "tokenizer", self.processor), "eos_token_id", 248044) or 248044

        with self._generate_lock:
            try:
                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True if temperature > 0 else False,
                        temperature=temperature,
                        top_p=top_p,
                        use_cache=True,
                        eos_token_id=stop_token_ids,
                        pad_token_id=pad_token_id,
                    )

                input_length = inputs["input_ids"].shape[1]
                generated_ids = generated_ids[:, input_length:]

                response = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]

                return strip_thinking(response)
            except torch.OutOfMemoryError as oom_err:
                log.warning(f"CUDA OOM in generate_chat_completion: {oom_err}. Retrying with reduced tokens...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                try:
                    with torch.inference_mode():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=min(max_new_tokens, 192),
                            do_sample=False,
                            use_cache=True,
                            eos_token_id=stop_token_ids,
                            pad_token_id=pad_token_id,
                        )
                    input_length = inputs["input_ids"].shape[1]
                    generated_ids = generated_ids[:, input_length:]
                    response = self.processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )[0]
                    return strip_thinking(response)
                except Exception as retry_err:
                    log.error(f"Retry failed after OOM: {retry_err}")
                    raise oom_err
            finally:
                try:
                    del inputs
                except Exception:
                    pass
                if "generated_ids" in locals():
                    del generated_ids


# ---------------------------------------------------------------------------
# Fast-Path Intent Matching for Conversational / Navigation Queries
# ---------------------------------------------------------------------------
def _match_conversational_query(q: str) -> Optional[str]:
    """
    Sub-millisecond intent router for non-substantive conversational turns.
    Bypasses 70GB model forward passes for greetings, courtesies, and meta-questions.
    """
    norm = re.sub(r"[^\w\s]", "", q.lower()).strip()
    if norm in ("hi", "hello", "hey", "namaste", "good morning", "good afternoon", "good evening", "greetings"):
        return (
            "## Answer\n"
            "Namaste! I am **LegalMind AI**, your specialized Indian legal research and adversarial intelligence assistant.\n\n"
            "You can query me on:\n"
            "- **Constitutional Law**: Fundamental rights, Golden Triangle (Articles 14, 19, 21), writ jurisdiction, and basic structure.\n"
            "- **Criminal Law & Procedure**: BNSS / CrPC provisions, BNS / IPC offenses, and bail standards (including Section 45 PMLA).\n"
            "- **Commercial & Regulatory Law**: Contractual remedies, arbitration appeals (S. 34/37), IBC priority, and environmental jurisprudence.\n"
            "- **Judicial Adversary Simulator**: Argue before simulated Supreme Court benches with live evaluation.\n\n"
            "Please state your legal inquiry or factual matrix to begin."
        )
    if norm in ("who are you", "what are you", "what can you do", "help", "how to use"):
        return (
            "## Answer\n"
            "I am **LegalMind AI**, an intelligent legal research platform fine-tuned on the Constitution of India, Central Legislation, and landmark Supreme Court of India judgments.\n\n"
            "### Core Capabilities:\n"
            "1. **Statutory & Precedential Research**: Evidence-grounded answers citing binding Supreme Court authorities.\n"
            "2. **Moot Court & Judicial Simulator**: Interactive appellate advocacy testing before 5 specialized corams.\n"
            "3. **Bail Assessment Matrix**: Twin condition analysis under PMLA, NDPS, and UAPA.\n"
            "4. **FIR Auditor**: Procedural offense audits.\n\n"
            "To begin, enter any legal query (e.g. *'Can bail be granted under Section 45 PMLA if there is unreasonable trial delay?'*)."
        )
    if norm in ("thank you", "thanks", "thanks a lot", "thank you so much", "bye", "goodbye"):
        return (
            "## Answer\n"
            "You are most welcome! Should you require further statutory analysis, judicial precedent research, or adversarial simulation, feel free to ask anytime. Wishing you successful advocacy!"
        )
    return None


# ---------------------------------------------------------------------------
# Full RAG pipeline (callable from FastAPI)
# ---------------------------------------------------------------------------
_fallback_retriever = None

def run_rag(
    query: str,
    retriever,             # HybridRetriever instance
    model: LegalMindModel,
    top_k: int = DEFAULT_TOP_K,
    conversation_history: list[dict] | None = None,
) -> dict:
    """
    Run the complete RAG pipeline for a single query.
    Supports multi-turn follow-ups, continuations, and context grounding.
    Returns structured result dict.
    """
    from retriever import format_citations

    # ── Fast-Path Intent Routing for Instant Response ───────────────────────
    fast_ans = _match_conversational_query(query)
    if fast_ans:
        return {
            "query":               query,
            "effective_query":     query,
            "is_continuation":     False,
            "answer":              fast_ans,
            "legal_provisions":    [],
            "judgments":           [],
            "legal_reasoning":     "",
            "citations":           [],
            "confidence":          "HIGH",
            "limitations":         "Conversational greeting or capability inquiry.",
            "raw_answer":          fast_ans,
            "retrieved_documents": [],
            "retrieval_latency":   0.0,
            "generation_latency":  0.001,
            "total_latency":       0.001,
            "question":            query,
            "answer_format":       "direct_answer",
            "confidence_score":    1.0,
            "confidence_label":    "high",
            "sources":             [],
            "legal_warnings":      [],
            "temporal_status":     "current",
            "reasoning_summary":   [],
            "related_cases":       [],
            "related_sections":    [],
            "article_coverage":    {},
            "is_golden_triangle":  False,
            "model":               "LegalMind-FastRouter",
            "created_at":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # ── Context Resolution for Continuations / Follow-ups ───────────────────
    if not conversation_history:
        conversation_history = _fetch_recent_history_from_db(query)

    last_user_query, last_assistant_text, retrieval_query = extract_conversation_context(
        query, conversation_history
    )
    is_cont = is_continuation_query(query) and bool(last_user_query)
    effective_query = last_user_query if is_cont else query

    # If the user says 'continue' in a brand new session with zero previous context
    if is_continuation_query(query) and not last_user_query:
        no_ctx_msg = (
            "You asked to continue, but there is no previous conversation topic in this session.\n\n"
            "Please ask a legal question or specify a topic you would like to explore "
            "(e.g., *'Can bail be denied under PMLA if twin conditions of Section 45 are not met?'* or "
            "*'What are the fundamental rights under the Indian Constitution?'*)."
        )
        return {
            "query":               query,
            "effective_query":     query,
            "is_continuation":     True,
            "answer":              no_ctx_msg,
            "legal_provisions":    [],
            "judgments":           [],
            "legal_reasoning":     "",
            "citations":           [],
            "confidence":          "HIGH",
            "limitations":         "",
            "raw_answer":          no_ctx_msg,
            "retrieved_documents": [],
            "retrieval_latency":   0.0,
            "generation_latency":  0.0,
            "total_latency":       0.0,
            "question":            query,
            "answer_format":       "direct_answer",
            "confidence_score":    1.0,
            "confidence_label":    "high",
            "sources":             [],
            "legal_warnings":      [],
            "temporal_status":     "current",
            "reasoning_summary":   [],
            "related_cases":       [],
            "related_sections":    [],
            "article_coverage":    {},
            "is_golden_triangle":  False,
            "model":               "Qwen/Qwen3.6-35B-A3B",
            "created_at":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    global _fallback_retriever
    if retriever is None:
        if _fallback_retriever is None:
            log.info("Retriever was None; initializing default HybridRetriever on demand…")
            try:
                from retriever import HybridRetriever
                _fallback_retriever = HybridRetriever(
                    chunks_dir="data/chunks/legislation/",
                    vector_db_dir="vector_db/legislation/",
                    embed_device="cuda:4",
                    rerank_device="cuda:4",
                )
            except Exception as e:
                log.warning(f"Could not initialize fallback retriever: {e}")
        retriever = _fallback_retriever

    t_start = time.time()

    # ── Retrieve ────────────────────────────────────────────────────────────
    article_coverage = {}
    is_gt = False
    is_const = False
    try:
        from constitutional_retrieval import is_constitutional_query, is_golden_triangle_query, get_constitutional_retriever
        is_const = is_constitutional_query(effective_query)
        is_gt = is_golden_triangle_query(effective_query)
    except Exception as e:
        log.warning(f"Could not check constitutional query status: {e}")

    if (is_const or is_gt) and not is_broad_constitutional_query(effective_query):
        log.info(f"Retrieving targeted constitutional evidence (GT={is_gt}) for: {retrieval_query[:80]}")
        t_r = time.time()
        try:
            cr = get_constitutional_retriever(base_retriever=retriever)
            chunks, article_coverage = cr.retrieve(retrieval_query, top_k=top_k)
            retrieval_time = time.time() - t_r
            log.info(f"ConstitutionalRetriever yielded {len(chunks)} chunks in {retrieval_time:.2f}s")
            citations = format_citations(chunks)
        except Exception as e:
            log.warning(f"ConstitutionalRetriever failed: {e}; falling back to standard retriever")
            chunks = retriever.retrieve(retrieval_query, top_k=top_k) if retriever else []
            retrieval_time = time.time() - t_r
            citations = format_citations(chunks) if chunks else []
    elif retriever is not None:
        log.info(f"Retrieving top-{top_k} chunks for: {retrieval_query[:80]}")
        t_r = time.time()
        chunks = retriever.retrieve(retrieval_query, top_k=top_k)
        retrieval_time = time.time() - t_r
        log.info(f"Retrieved {len(chunks)} chunks in {retrieval_time:.2f}s")
        citations = format_citations(chunks)
    else:
        log.warning("No retriever available; answering using base LLM knowledge only.")
        chunks = []
        citations = []
        retrieval_time = 0.0

    # ── Generate ────────────────────────────────────────────────────────────
    if is_gt:
        budget_tokens = 2048
    elif is_const or is_broad_constitutional_query(effective_query):
        budget_tokens = 2048
    elif len(effective_query.split()) > 10 or any(kw in effective_query.lower() for kw in ["explain", "difference", "compare", "detail", "discuss", "analyze", "procedure", "condition", "bail", "provision", "article", "section", "what", "how", "when", "why", "who"]):
        budget_tokens = 1600
    else:
        budget_tokens = 1200

    log.info(f"Generating answer with Qwen3.6 (continuation={is_cont}, budget={budget_tokens} tokens)…")
    t_g = time.time()
    answer_text = model.generate(
        query                   = query,
        retrieved_chunks        = chunks,
        citations               = citations,
        conversation_history    = conversation_history,
        is_continuation         = is_cont,
        last_user_query         = last_user_query,
        last_assistant_response = last_assistant_text,
        max_new_tokens          = budget_tokens,
    )
    generation_time = time.time() - t_g
    log.info(f"Generation complete in {generation_time:.2f}s")

    total_time = time.time() - t_start

    # ── Parse structured output ─────────────────────────────────────────────
    parsed = _parse_structured_answer(answer_text, query=effective_query, chunks=chunks)

    # ── Temporal Law Context ────────────────────────────────────────────────
    try:
        import temporal_law
        temporal_ctx = temporal_law.detect_temporal_context(effective_query)
        temporal_status = temporal_ctx.get("temporal_status", "current")
        temporal_warnings = temporal_ctx.get("warnings", [])
    except Exception as e:
        log.warning(f"Temporal law detection error: {e}")
        temporal_ctx = {}
        temporal_status = "unknown"
        temporal_warnings = []

    # ── Precedent Graph Service ─────────────────────────────────────────────
    try:
        import precedent_graph
        pg = precedent_graph.get_precedent_service()
        matched_precedents = pg.search_precedents(effective_query, limit=3)
        related_cases = [p.case_name for p in matched_precedents]
        precedent_nodes = [p.to_dict() for p in matched_precedents]
    except Exception as e:
        log.warning(f"Precedent graph error: {e}")
        related_cases = []
        precedent_nodes = []

    if not related_cases:
        related_cases = [j for j in (parsed.get("judgments") or []) if j != "None"]

    # ── Citation & Hallucination Verification ───────────────────────────────
    try:
        import citation_verifier
        verification = citation_verifier.verify_answer_citations(
            answer_text=parsed.get("answer") or answer_text,
            retrieved_chunks=chunks,
            query=effective_query,
        )
        citation_warnings = verification.legal_warnings
        conf_label = verification.confidence_label
        conf_score = verification.confidence_score
    except Exception as e:
        log.warning(f"Citation verification error: {e}")
        citation_warnings = []
        conf_label = "low" if ("LOW" in parsed.get("confidence", "").upper()) else "medium"
        conf_score = 0.50

    all_warnings = list(temporal_warnings) + list(citation_warnings)

    # ── Structured IRAC Reasoning Summary ───────────────────────────────────
    try:
        import legal_reasoning
        reasoning_summary = legal_reasoning.generate_structured_reasoning(
            query=effective_query,
            retrieved_chunks=chunks,
            parsed_sections=parsed,
        )
    except Exception as e:
        log.warning(f"Legal reasoning generator error: {e}")
        reasoning_summary = []

    # ── Response Formulation ──────────────────────────────────────────────
    # Routing logic:
    #   • Golden Triangle (GT) queries → full 14-section comparative synthesis.
    #   • General queries with a substantive LLM answer → use LLM answer as-is
    #     (the new system prompt already produces the 6-section structured output).
    #   • General queries where LLM answer is empty / trivially short → invoke
    #     build_fallback_legal_response() to build a clean 6-section response from
    #     the parsed sections, never returning an empty or "Insufficient" message.
    #
    # Knowledge mode is derived from the grounding assessment done at the API layer
    # (chat_api.py). Inside run_rag we approximate it from chunk count.
    _km = "evidence_based"
    if not chunks:
        _km = "general_knowledge"
    elif len(chunks) <= 2:
        _km = "partial_evidence"

    final_answer = answer_text.strip() if answer_text.strip() else (parsed.get("answer") or "")
    try:
        import response_generator
        if is_gt:
            # Full 14-section Golden Triangle comparative analysis
            final_answer = response_generator.build_detailed_legal_response(
                query=effective_query,
                base_answer=parsed.get("answer") or final_answer,
                retrieved_chunks=chunks,
                parsed_sections=parsed,
                citations_list=citations,
                temporal_info=temporal_ctx,
                precedent_nodes=precedent_nodes,
            )
        elif not final_answer or len(final_answer.strip()) < 100:
            # LLM produced an empty or near-empty answer — use the 6-section fallback builder
            log.info("LLM answer too short; invoking build_fallback_legal_response.")
            final_answer = response_generator.build_fallback_legal_response(
                query=effective_query,
                llm_answer=answer_text,
                parsed_sections=parsed,
                knowledge_mode=_km,
                retrieved_chunks=chunks,
                citations_list=citations,
            )
        # else: non-GT query with substantive LLM answer — use as-is
    except Exception as e:
        log.warning(f"Response generator error: {e}")
        if not final_answer:
            final_answer = answer_text or "(The model did not produce an answer. Please try again.)"

    # Align confidence string with verification
    final_confidence_str = parsed.get("confidence", "MEDIUM")
    if is_gt and article_coverage:
        all_found = all(c.get("found") for c in article_coverage.values())
        if all_found and len(chunks) >= 6:
            conf_label = "high"
            conf_score = 0.95
            final_confidence_str = "HIGH — Direct statutory texts for Articles 14, 19, and 21, Golden Triangle doctrine, and landmark Supreme Court precedents retrieved from the corpus."
    elif conf_label == "low" and "HIGH" in final_confidence_str.upper():
        final_confidence_str = "LOW — Retrieved evidence is insufficient or citations unverified."


    # ── Build retrieved_documents summary & extended sources ────────────────
    retrieved_docs = []
    sources = []
    related_sections_set = set()

    for chunk in chunks:
        c_dict = {
            "citation_id":    chunk.get("citation_id"),
            "document_id":    chunk.get("document_id"),
            "document_type":  chunk.get("document_type"),
            "title":          chunk.get("title"),
            "source":         chunk.get("source"),
            "court":          chunk.get("court"),
            "date":           chunk.get("date"),
            "citation":       chunk.get("citation"),
            "article":        chunk.get("article"),
            "section":        chunk.get("section"),
            "rrf_score":      chunk.get("rrf_score"),
            "rerank_score":   chunk.get("rerank_score"),
            "text_preview":   chunk.get("text", "")[:300],
        }
        retrieved_docs.append(c_dict)

        sources.append({
            "title":         chunk.get("title") or chunk.get("act_name"),
            "citation":      chunk.get("citation") or chunk.get("article") or chunk.get("section"),
            "court":         chunk.get("court"),
            "date":          chunk.get("date"),
            "source_id":     chunk.get("document_id") or chunk.get("source"),
            "excerpt":       chunk.get("text", "")[:300],
            "citation_id":   chunk.get("citation_id"),
            "document_type": chunk.get("document_type"),
        })

        if chunk.get("section"):
            related_sections_set.add(f"Section {chunk['section']}")
        if chunk.get("article"):
            related_sections_set.add(f"Article {chunk['article']}")

    for trans in temporal_ctx.get("suggested_transitions", []):
        if trans.get("modern_section"):
            related_sections_set.add(f"{trans['modern_act']} Section {trans['modern_section']}")

    return {
        # 100% Backward Compatible Keys
        "query":               query,
        "effective_query":     effective_query,
        "is_continuation":     is_cont,
        "answer":              final_answer,
        "legal_provisions":    parsed.get("legal_provisions", []),
        "judgments":           parsed.get("judgments", []),
        "legal_reasoning":     parsed.get("legal_reasoning", ""),
        "citations":           citations,
        "confidence":          final_confidence_str,
        "limitations":         parsed.get("limitations", ""),
        "raw_answer":          answer_text,
        "retrieved_documents": retrieved_docs,
        "retrieval_latency":   round(retrieval_time, 3),
        "generation_latency":  round(generation_time, 3),
        "total_latency":       round(total_time, 3),
        # Extended Fields (for Frontend/History Developer & API Contract)
        "question":            query,
        "answer_format":       "detailed_legal_explanation",
        "confidence_score":    conf_score,
        "confidence_label":    conf_label,
        "sources":             sources,
        "legal_warnings":      all_warnings,
        "temporal_status":     temporal_status,
        "jurisdiction":        "India",
        "reasoning_summary":   reasoning_summary,
        "related_cases":       related_cases,
        "related_sections":    sorted(list(related_sections_set)),
        "article_coverage":    article_coverage,
        "is_golden_triangle":  is_gt,
        "model":               "Qwen/Qwen3.6-35B-A3B",
        "created_at":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_rag_stream(
    query: str,
    retriever,
    model: LegalMindModel,
    top_k: int = DEFAULT_TOP_K,
    conversation_history: list[dict] | None = None,
):
    """
    Generator streaming RAG execution.
    Yields tuples of (event_name, data):
      - ("meta", {"retrieved_documents": [...], "citations": [...], "retrieval_latency": ...})
      - ("token", text_chunk)
      - ("done", full_result_dict)
    """
    from retriever import format_citations

    # 1. Fast-Path Intent Routing for Instant Response
    fast_ans = _match_conversational_query(query)
    if fast_ans:
        res = {
            "query":               query,
            "effective_query":     query,
            "is_continuation":     False,
            "answer":              fast_ans,
            "legal_provisions":    [],
            "judgments":           [],
            "legal_reasoning":     "",
            "citations":           [],
            "confidence":          "HIGH",
            "limitations":         "Conversational greeting or capability inquiry.",
            "raw_answer":          fast_ans,
            "retrieved_documents": [],
            "retrieval_latency":   0.0,
            "generation_latency":  0.001,
            "total_latency":       0.001,
            "question":            query,
            "answer_format":       "direct_answer",
            "confidence_score":    1.0,
            "confidence_label":    "high",
            "sources":             [],
            "legal_warnings":      [],
            "temporal_status":     "current",
            "reasoning_summary":   [],
            "related_cases":       [],
            "related_sections":    [],
            "article_coverage":    {},
            "is_golden_triangle":  False,
            "model":               "LegalMind-FastRouter",
            "created_at":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        yield ("meta", {"retrieved_documents": [], "citations": [], "retrieval_latency": 0.0})
        yield ("token", fast_ans)
        yield ("done", res)
        return

    # Context Resolution
    if not conversation_history:
        conversation_history = _fetch_recent_history_from_db(query)

    last_user_query, last_assistant_text, retrieval_query = extract_conversation_context(
        query, conversation_history
    )
    is_cont = is_continuation_query(query) and bool(last_user_query)
    effective_query = last_user_query if is_cont else query

    article_coverage = {}
    is_gt = False
    is_const = False
    try:
        from constitutional_retrieval import is_constitutional_query, is_golden_triangle_query, get_constitutional_retriever
        is_const = is_constitutional_query(effective_query)
        is_gt = is_golden_triangle_query(effective_query)
    except Exception as e:
        log.warning(f"Could not check constitutional query status in stream: {e}")

    t_start = time.time()
    t_r = time.time()
    if (is_const or is_gt) and not is_broad_constitutional_query(effective_query):
        log.info(f"Stream: Retrieving targeted constitutional evidence (GT={is_gt}) for: {retrieval_query[:80]}")
        try:
            cr = get_constitutional_retriever(base_retriever=retriever)
            chunks, article_coverage = cr.retrieve(retrieval_query, top_k=14 if is_gt else top_k)
            retrieval_time = time.time() - t_r
            log.info(f"Stream: ConstitutionalRetriever yielded {len(chunks)} chunks in {retrieval_time:.2f}s")
            citations = format_citations(chunks)
        except Exception as e:
            log.warning(f"Stream: ConstitutionalRetriever failed: {e}; falling back to standard retriever")
            chunks = retriever.retrieve(retrieval_query, top_k=top_k) if retriever else []
            retrieval_time = time.time() - t_r
            citations = format_citations(chunks) if chunks else []
    elif retriever is not None:
        chunks = retriever.retrieve(retrieval_query, top_k=top_k)
        retrieval_time = time.time() - t_r
        citations = format_citations(chunks) if chunks else []
    else:
        chunks = []
        retrieval_time = 0.0
        citations = []

    retrieved_docs = []
    for chunk in chunks:
        retrieved_docs.append({
            "citation_id":         chunk.get("citation_id"),
            "document_id":         chunk.get("document_id"),
            "document_type":       chunk.get("document_type"),
            "title":               chunk.get("title"),
            "source":              chunk.get("source"),
            "court":               chunk.get("court"),
            "date":                chunk.get("date"),
            "section":             chunk.get("section"),
            "article":             chunk.get("article"),
            "citation":            chunk.get("citation"),
            "text_preview":        (chunk.get("text") or "")[:200],
            "retrieval_latency_s": chunk.get("retrieval_latency_s", round(retrieval_time, 3)),
        })

    # Yield metadata first (Time To First Token < 0.2s!)
    yield ("meta", {
        "retrieved_documents": retrieved_docs,
        "citations":           citations,
        "retrieval_latency":   round(retrieval_time, 3),
        "is_golden_triangle":  is_gt,
        "article_coverage":    article_coverage,
    })

    # Determine generous token budget
    if is_gt:
        budget_tokens = 2048
    elif is_const or is_broad_constitutional_query(effective_query):
        budget_tokens = 1792
    elif len(effective_query.split()) > 10 or any(kw in effective_query.lower() for kw in ["explain", "difference", "compare", "detail", "discuss", "analyze", "procedure", "condition", "bail", "provision", "article", "section"]):
        budget_tokens = 1400
    else:
        budget_tokens = 896

    # Stream generated tokens in real time
    t_g = time.time()
    accumulated_tokens = []
    for chunk in model.generate_stream(
        query=query,
        retrieved_chunks=chunks,
        citations=citations,
        conversation_history=conversation_history,
        is_continuation=is_cont,
        last_user_query=last_user_query,
        last_assistant_response=last_assistant_text,
        max_new_tokens=budget_tokens,
    ):
        accumulated_tokens.append(chunk)
        yield ("token", chunk)

    generation_time = time.time() - t_g
    total_time = time.time() - t_start
    answer_text = "".join(accumulated_tokens)

    parsed = _parse_structured_answer(answer_text, query=effective_query, chunks=chunks)

    # Temporal Law Context
    try:
        import temporal_law
        temporal_ctx = temporal_law.detect_temporal_context(effective_query)
        temporal_status = temporal_ctx.get("temporal_status", "current")
        temporal_warnings = temporal_ctx.get("warnings", [])
    except Exception:
        temporal_ctx = {}
        temporal_status = "current"
        temporal_warnings = []

    # Precedent Graph Service
    try:
        import precedent_graph
        pg = precedent_graph.get_precedent_service()
        matched_precedents = pg.search_precedents(effective_query, limit=3)
        related_cases = [p.case_name for p in matched_precedents]
    except Exception:
        related_cases = []

    if not related_cases:
        related_cases = [j for j in (parsed.get("judgments") or []) if j != "None"]

    # Final answer handling for GT query fallback if needed
    final_answer = answer_text.strip()
    if is_gt and (not final_answer or len(final_answer) < 400):
        try:
            import response_generator
            final_answer = response_generator.build_detailed_legal_response(
                query=effective_query,
                base_answer=parsed.get("answer") or final_answer,
                retrieved_chunks=chunks,
                parsed_sections=parsed,
                citations_list=citations,
                temporal_info=temporal_ctx,
            )
        except Exception as e:
            log.warning(f"Response generator error in stream: {e}")

    # Citation Verification
    conf_score = 0.85 if chunks else 0.50
    conf_label = "high" if len(chunks) >= 3 else "medium"
    citation_warnings = []
    try:
        import citation_verifier
        verification = citation_verifier.verify_answer_citations(
            answer_text=parsed.get("answer") or final_answer,
            retrieved_chunks=chunks,
            query=effective_query,
        )
        citation_warnings = verification.legal_warnings
        conf_label = verification.confidence_label
        conf_score = verification.confidence_score
    except Exception as e:
        log.warning(f"Citation verification error in stream: {e}")

    final_confidence_str = parsed.get("confidence", "MEDIUM")
    if is_gt and article_coverage:
        all_found = all(c.get("found") for c in article_coverage.values())
        if all_found and len(chunks) >= 6:
            conf_label = "high"
            conf_score = 0.95
            final_confidence_str = "HIGH — Direct statutory texts for Articles 14, 19, and 21, Golden Triangle doctrine, and landmark Supreme Court precedents retrieved from the corpus."
    elif conf_label == "low" and "HIGH" in final_confidence_str.upper():
        final_confidence_str = "LOW — Retrieved evidence is insufficient or citations unverified."

    res = {
        "query":               query,
        "effective_query":     effective_query,
        "is_continuation":     is_cont,
        "answer":              final_answer,
        "legal_provisions":    parsed.get("legal_provisions", []),
        "judgments":           parsed.get("judgments", []),
        "legal_reasoning":     parsed.get("legal_reasoning", ""),
        "citations":           citations,
        "confidence":          final_confidence_str,
        "limitations":         parsed.get("limitations", ""),
        "raw_answer":          answer_text,
        "retrieved_documents": retrieved_docs,
        "retrieval_latency":   round(retrieval_time, 3),
        "generation_latency":  round(generation_time, 3),
        "total_latency":       round(total_time, 3),
        "question":            query,
        "answer_format":       "detailed_legal_explanation",
        "confidence_score":    conf_score,
        "confidence_label":    conf_label,
        "sources":             [],
        "legal_warnings":      temporal_warnings + citation_warnings,
        "temporal_status":     temporal_status,
        "jurisdiction":        "India",
        "reasoning_summary":   [],
        "related_cases":       related_cases,
        "related_sections":    [],
        "article_coverage":    article_coverage,
        "is_golden_triangle":  is_gt,
        "model":               "Qwen/Qwen3.6-35B-A3B",
        "created_at":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    yield ("done", res)


def organize_broad_constitutional_answer(
    raw_answer: str,
    chunks: list[dict] | None,
) -> str:
    """
    Organize an answer for a broad constitutional query into the 10 canonical topics:
    - Preamble
    - Fundamental Rights
    - Directive Principles of State Policy
    - Fundamental Duties
    - Union and State Government
    - Judiciary
    - Federal structure
    - Elections and constitutional bodies
    - Emergency provisions
    - Constitutional amendments

    Rules:
    - Only topics supported by retrieved evidence provide substantive legal details.
    - Any constitutional topic not present in retrieved evidence explicitly states:
      "The retrieved evidence does not cover this topic."
    """
    extracted_sections = {}

    # Extract existing topic sections if raw_answer already has subheadings
    for topic in CONSTITUTIONAL_TOPICS:
        topic_re = re.escape(topic)
        pattern = rf"(?:###|\*\*|(?:\d+\.))\s*{topic_re}[:\*\s]*(.*?)(?=(?:###|\*\*|(?:\d+\.))\s*(?:{'|'.join(re.escape(t) for t in CONSTITUTIONAL_TOPICS)})|$)"
        m = re.search(pattern, raw_answer, re.DOTALL | re.IGNORECASE)
        if m:
            content = m.group(1).strip()
            content = re.sub(r"^[:\-\*]\s*", "", content).strip()
            extracted_sections[topic] = content

    supported_status = {
        topic: check_topic_support(topic, chunks)
        for topic in CONSTITUTIONAL_TOPICS
    }

    output_sections = []

    # Preserve any high-level introduction before topic breakdown if present
    intro = ""
    first_topic_idx = len(raw_answer)
    for topic in CONSTITUTIONAL_TOPICS:
        m = re.search(rf"(?:###|\*\*)\s*{re.escape(topic)}", raw_answer, re.IGNORECASE)
        if m and m.start() < first_topic_idx:
            first_topic_idx = m.start()
    if 0 < first_topic_idx < len(raw_answer):
        possible_intro = raw_answer[:first_topic_idx].strip()
        if len(possible_intro) > 20 and not possible_intro.startswith("##"):
            intro = possible_intro

    if intro:
        output_sections.append(intro)

    for topic in CONSTITUTIONAL_TOPICS:
        is_sup = supported_status[topic]
        if not is_sup:
            # Unsupported topic: mandatory explicit statement
            section_body = UNSUPPORTED_TOPIC_MESSAGE
        else:
            existing_text = extracted_sections.get(topic, "").strip()
            if existing_text and not re.search(r"does\s+not\s+cover\b", existing_text, re.IGNORECASE):
                section_body = existing_text
            else:
                # If model did not separate into this topic heading, search for matching paragraphs
                patterns = CONSTITUTIONAL_TOPIC_PATTERNS.get(topic, [])
                matching_paras = []
                for p in raw_answer.split("\n\n"):
                    p_clean = p.strip()
                    if not p_clean:
                        continue
                    if any(p_clean.startswith(f"### {other}") for other in CONSTITUTIONAL_TOPICS if other != topic):
                        continue
                    if any(re.search(pat, p_clean, re.IGNORECASE) for pat in patterns):
                        matching_paras.append(p_clean)

                if matching_paras:
                    section_body = "\n\n".join(matching_paras)
                else:
                    # Summarize directly from retrieved evidence chunks
                    matched_chunks = [
                        c for c in (chunks or [])
                        if any(re.search(p, f"{c.get('title','')} {c.get('act_name','')} {c.get('text','')}", re.IGNORECASE)
                               for p in patterns)
                    ]
                    snippets = []
                    for c in matched_chunks:
                        cid = c.get("citation_id")
                        title = c.get("title") or c.get("act_name", "Constitution of India")
                        prefix = f"[{cid}] {title}" if cid else title
                        lines = [ln.strip() for ln in c.get("text", "").split("\n") if ln.strip() and not ln.startswith("Act:")]
                        sample = " ".join(lines[:3])[:300]
                        if sample:
                            snippets.append(f"{prefix}: {sample}")
                    section_body = "\n\n".join(snippets) if snippets else "Covered in retrieved evidence."

        output_sections.append(f"### {topic}\n{section_body}")

    return "\n\n".join(output_sections)


def enforce_grounding_and_confidence(parsed: dict, query: str = "", chunks: list[dict] | None = None) -> dict:
    """
    Enforce evidence grounding and confidence constraints:
    - If answer explicitly states evidence is insufficient/absent or relies on general knowledge,
      confidence MUST NOT be HIGH (downgraded to LOW or MEDIUM).
    - If a specific statutory/constitutional provision was requested (e.g. Article 21, Section 302)
      and is NOT present in the retrieved chunks, confidence MUST NOT be HIGH.
    - If query is a broad constitutional query and some constitutional topics are not covered
      in retrieved evidence, confidence MUST NOT be HIGH.
    - Ensure limitations section clearly states the evidence gap if insufficient.
    """
    answer = parsed.get("answer", "")
    confidence = parsed.get("confidence", "")

    # Evidence insufficiency indicators in answer text
    evidence_insufficient_patterns = [
        r"not\s+(?:mentioned|defined|present|found|included)\b",
        r"does\s+not\s+contain\b",
        r"insufficient\s+evidence\b",
        r"relying\s+on\s+general\s+(?:legal\s+)?knowledge\b",
        r"evidence\s+is\s+insufficient\b",
        r"no\s+relevant\s+retrieved\s+evidence\b",
        r"provided\s+(?:legal\s+)?evidence\s+is\s+insufficient\b",
    ]
    stated_insufficient = any(
        re.search(p, answer, re.IGNORECASE) for p in evidence_insufficient_patterns
    )

    # Missing target provision check (for specific article/section queries)
    target_missing = False
    if query:
        articles = re.findall(r"\b(?:article|art\.?)\s*(\d+[a-z]?)\b", query, re.IGNORECASE)
        sections = re.findall(r"\b(?:section|sec\.?)\s*(\d+[a-z]?)\b", query, re.IGNORECASE)
        requested_nums = [a.lower() for a in articles] + [s.lower() for s in sections]

        if requested_nums and chunks is not None:
            if not chunks:
                target_missing = True
            else:
                all_text = " ".join(c.get("text", "") for c in chunks)
                for num in requested_nums:
                    if not re.search(rf"\b{re.escape(num)}\b", all_text):
                        target_missing = True
                        break

    # Broad constitutional query grounding check
    broad_const_gap = False
    unsupported_topics = []
    supported_topics = []
    if is_broad_constitutional_query(query):
        unsupported_topics = [t for t in CONSTITUTIONAL_TOPICS if not check_topic_support(t, chunks)]
        supported_topics = [t for t in CONSTITUTIONAL_TOPICS if check_topic_support(t, chunks)]
        if unsupported_topics:
            broad_const_gap = True

    if broad_const_gap:
        # Broad constitutional query with unsupported topics -> confidence MUST NOT be HIGH
        if re.search(r"\bHIGH\b", confidence, re.IGNORECASE):
            if supported_topics:
                parsed["confidence"] = (
                    f"MEDIUM — Retrieved legal evidence directly supports {', '.join(supported_topics[:2])}; "
                    f"{len(unsupported_topics)} constitutional topics are not covered in the retrieved evidence."
                )
            else:
                parsed["confidence"] = "LOW — Retrieved legal evidence does not cover the requested constitutional topics."
        elif not confidence:
            if supported_topics:
                parsed["confidence"] = f"MEDIUM — Partial constitutional evidence retrieved ({', '.join(supported_topics[:2])})."
            else:
                parsed["confidence"] = "LOW — Retrieved legal evidence does not cover constitutional topics."

        limitations = parsed.get("limitations", "")
        unsupported_note = f"The retrieved evidence does not cover: {', '.join(unsupported_topics)}."
        if limitations:
            if "does not cover" not in limitations.lower():
                parsed["limitations"] = f"{limitations} Note: {unsupported_note}"
        else:
            parsed["limitations"] = unsupported_note

    elif stated_insufficient or target_missing or (chunks is not None and not chunks):
        # Downgrade HIGH to LOW for specific queries or general insufficiency
        if re.search(r"\bHIGH\b", confidence, re.IGNORECASE):
            parsed["confidence"] = "LOW — Retrieved legal evidence does not directly support the answer; relying on general knowledge."
        elif not confidence:
            parsed["confidence"] = "LOW — Retrieved legal evidence is insufficient."

        # Ensure limitations explicitly discloses insufficiency
        limitations = parsed.get("limitations", "")
        if limitations:
            if not any(w in limitations.lower() for w in ["insufficient", "not present", "general knowledge", "gap"]):
                parsed["limitations"] = limitations + " Note: Retrieved legal evidence does not directly contain the requested provision; answer relies on general legal knowledge."
        else:
            parsed["limitations"] = "Retrieved legal evidence does not directly contain the requested provision; answer relies on general legal knowledge."

    return parsed


def _parse_structured_answer(text: str, query: str = "", chunks: list[dict] | None = None) -> dict:
    """
    Parse the model's structured output into separate fields.
    Falls back gracefully if parsing fails.
    Enforces evidence grounding and confidence constraints.
    """
    sections = {
        "answer":           r"##\s+(?:Direct\s+)?Answer\s+(.*?)(?=(?:\r?\n)##(?!\#)|$)",
        "legal_provisions": r"##\s+(?:Relevant\s+(?:Legal\s+)?(?:Provisions|Authorities)|Statutory\s+Provisions)\s+(.*?)(?=(?:\r?\n)##(?!\#)|$)",
        "judgments":        r"##\s+(?:Relevant\s+)?(?:Judgments(?:\s*/\s*Precedents)?|Landmark\s+Judgments|Case\s+Law|Precedents)\s+(.*?)(?=(?:\r?\n)##(?!\#)|$)",
        "legal_reasoning":  r"##\s+(?:Legal\s+Reasoning|Legal\s+Position|Analysis|Application(?:\s+to\s+the\s+Question)?)\s+(.*?)(?=(?:\r?\n)##(?!\#)|$)",
        "confidence":       r"##\s+Confidence(?:\s+Level)?\s+(.*?)(?=(?:\r?\n)##(?!\#)|$)",
        "limitations":      r"##\s+Limitations?(?:\s*(?:&|and)\s*Gaps?)?\s+(.*?)(?=(?:\r?\n)##(?!\#)|$)",
    }

    result = {}
    for key, pattern in sections.items():
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            content = m.group(1).strip()
            if key in ("legal_provisions", "judgments"):
                # Split into list items
                items = [
                    line.strip().lstrip("•-*0123456789. ").strip()
                    for line in content.split("\n")
                    if line.strip() and len(line.strip()) > 3
                ]
                result[key] = items
            else:
                result[key] = content
        else:
            result[key] = [] if key in ("legal_provisions", "judgments") else ""

    # Fallback if "answer" section was not matched
    if not result.get("answer"):
        first_header = re.search(r"^\s*##\s+", text, re.MULTILINE)
        if first_header and first_header.start() > 0:
            pre_text = text[:first_header.start()].strip()
            if pre_text:
                result["answer"] = pre_text
        elif not any(result.values()):
            result["answer"] = text.strip()

    if is_broad_constitutional_query(query):
        result["answer"] = organize_broad_constitutional_answer(result.get("answer", ""), chunks)

    return enforce_grounding_and_confidence(result, query, chunks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LegalMind AI — RAG Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query", "-q", default=None,
                        help="Single query (skip interactive mode)")
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU,
                        help=f"CUDA device for Qwen model (default: {DEFAULT_GPU})")
    parser.add_argument("--embed-gpu", type=int, default=4,
                        help="CUDA device for embedding model (default: 4)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Exit after single query (use with --query)")
    parser.add_argument("--no-retriever", action="store_true",
                        help="Skip retrieval, use model directly (baseline test)")
    parser.add_argument("--chunks-dir",    default="data/chunks/legislation/")
    parser.add_argument("--vector-db-dir", default="vector_db/legislation/")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Load model ──────────────────────────────────────────────────────────
    print("\n⚖️  LegalMind AI — Initialising…\n")
    lm = LegalMindModel.get_instance(gpu=args.gpu)

    # ── Load retriever ──────────────────────────────────────────────────────
    retriever = None
    if not args.no_retriever:
        try:
            from retriever import HybridRetriever
            retriever = HybridRetriever(
                chunks_dir    = args.chunks_dir,
                vector_db_dir = args.vector_db_dir,
                embed_device  = f"cuda:{args.embed_gpu}",
            )
        except Exception as e:
            log.warning(f"Could not initialise retriever: {e}")
            log.warning("Running in base model mode (no retrieval).")

    cli_history = []

    def ask(query: str):
        print(f"\n{'='*70}")
        print(f"QUERY: {query}")
        print(f"{'='*70}")

        if retriever:
            result = run_rag(query, retriever, lm, top_k=args.top_k, conversation_history=cli_history)
            cli_history.append({"role": "user", "content": query})
            cli_history.append({"role": "assistant", "content": result.get("raw_answer", "")})
            print(f"\n{result['raw_answer']}")
            print(f"\n{'─'*70}")
            print("CITATIONS:")
            for c in result["citations"]:
                print(f"  {c}")
            print(f"\nLatency: retrieve={result['retrieval_latency']}s  "
                  f"generate={result['generation_latency']}s  "
                  f"total={result['total_latency']}s")
        else:
            # Base model (no RAG)
            messages = [
                {"role": "system", "content": "You are LegalMind AI, an expert in Indian law."},
                {"role": "user",   "content": query},
            ]
            inputs = lm.processor.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_tensors="pt", return_dict=True,
            )
            inputs = {k: v.to(lm.model.device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}
            with torch.inference_mode():
                out = lm.model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                )
            resp = lm.processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0]
            print(f"\n{strip_thinking(resp)}")

        print(f"\n{'─'*70}\n")

    if args.query:
        ask(args.query)
        if args.no_interactive:
            return

    # ── Interactive loop ────────────────────────────────────────────────────
    print("\n⚖️  LegalMind AI is ready.")
    print("Type your legal question and press Enter. Type 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting…")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Exiting…")
            break

        ask(query)


if __name__ == "__main__":
    main()
