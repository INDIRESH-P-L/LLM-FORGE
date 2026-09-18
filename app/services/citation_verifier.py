#!/usr/bin/env python3
"""
app/services/citation_verifier.py
=================================
Check every case, citation and statutory provision an answer asserts against
the source corpus, before the user sees it.

Why: a fabricated citation is the defining failure mode of a legal assistant —
lawyers have been sanctioned for filing AI-invented authorities. A model that
"sounds right" is worthless here; the only thing that makes an answer usable is
whether each authority it names actually exists and actually says what is
claimed.

Three verdicts, deliberately distinct:

    grounded    the authority appears in the passages retrieved for THIS answer.
                Strongest: the model had the text in front of it.
    in_corpus   the authority exists in our source data but was not retrieved
                for this question. Real, but the model recalled it from weights
                rather than reading it — so the proposition attached to it is
                not evidenced.
    unverified  not found at all. Treat as fabricated until shown otherwise.

The verifier never rewrites an answer silently: it returns findings, and the
caller decides whether to annotate or withhold.

    from app.services.citation_verifier import CitationVerifier
    v = CitationVerifier()
    report = v.verify(answer_text, retrieved_chunks)
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path

log = logging.getLogger("legalmind.verify")

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = Path(os.environ.get("LEGALMIND_AUTHORITY_DB",
                              ROOT / "data/verification/authorities.db"))

# ── extraction patterns ─────────────────────────────────────────────────────
# Indian reporters, written with or without periods and with either bracket.
RE_CITATION = re.compile(
    r"[\[\(]?\s*(?:19|20)\d{2}\s*[\]\)]?\s*"
    r"(?:\(\d+\)\s*)?\d{0,3}\s*"
    r"(?:S\.?\s?C\.?\s?C\.?(?:\s*OnLine)?|A\.?\s?I\.?\s?R\.?|S\.?\s?C\.?\s?R\.?|"
    r"INSC|Cri\.?\s?L\.?\s?J\.?|SUPP\.?\s*\d*\s*S\.?\s?C\.?\s?R\.?)"
    r"[\s.]*(?:S\.?C\.?|SC)?\s*\d{1,5}",
    re.I)

# A party name is a run of Capitalised words, optionally joined by lowercase
# connectors ("of", "and", "the"). Anchoring on capitalisation is what stops
# the match running into the following verb — an earlier lazy pattern captured
# "State of Kerala establi" from "...State of Kerala established...", which
# then failed to match the real judgment and reported a landmark case as
# unverified.
_PARTY = (r"[A-Z][A-Za-z'’.&\-]*"
          r"(?:\s+(?:of|and|the|&|for|in)\s+[A-Z][A-Za-z'’.&\-]*"
          r"|\s+[A-Z][A-Za-z'’.&\-]*){0,9}")
RE_CASE_NAME = re.compile(
    rf"({_PARTY})\s+(?:v\.?s?\.?|versus|Versus|VERSUS)\s+({_PARTY})")

RE_SECTION = re.compile(
    r"\b[Ss]ection\s+(\d+[A-Za-z]{0,2})\b"
    # The act name, when given, runs to the first sentence break. Without the
    # [^.]* guard this swallowed following sentences ("Section 3 of Act. See
    # also Article 21") and produced nonsense act names.
    r"(?:\s+[Oo]f\s+(?:[Tt]he\s+)?((?:[A-Z][\w'()\-]*(?:\s+(?![Ss]ee\b|[Tt]he\b)[A-Z\d][\w'()\-]*){0,7})"
    r"(?:\s+[Aa]ct(?:,?\s+\d{4})?)?))?")
RE_ARTICLE = re.compile(r"\bArticle\s+(\d+[A-Z]{0,2})\b", re.I)

# Sentences that merely name a statute in passing are not citations to verify.
STOP_NAMES = {"", "india", "the state", "state", "union of india", "the union of india"}


def norm_citation(c: str) -> str:
    return re.sub(r"[\[\](){}.\s]", "", (c or "").lower())


ACT_ACRONYMS = {
    "pmla": "prevention of money laundering act",
    "ipc": "indian penal code",
    "crpc": "code of criminal procedure",
    "cpc": "code of civil procedure",
    "ndps": "narcotic drugs and psychotropic substances act",
    "uapa": "unlawful activities prevention act",
    "mva": "motor vehicles act",
    "pocso": "protection of children from sexual offences act",
    "sc st act": "scheduled castes and the scheduled tribes prevention of atrocities act",
    "ni act": "negotiable instruments act",
    "rti": "right to information act",
    "it act": "information technology act"
}

def norm_name(n: str) -> str:
    n = (n or "").lower()
    n = re.sub(r"\b(?:m/s|shri|smt|sri|the state of|state of)\b", " ", n)
    n = re.sub(r"\s+(?:v|vs|versus)\.?\s+", " v ", n)
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    n = re.sub(r"\b(?:ors|anr|others|another|etc|dead|thr|lrs|d)\b", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return ACT_ACRONYMS.get(n, n)


# Words that sit next to a case name in prose but are not part of it. Left
# uncleaned, "In Maneka Gandhi v. Union of India the Court held…" yields the
# party "Union of India the Court", which matches nothing.
_LEAD_NOISE = {"in", "see", "per", "cf", "but", "and", "also", "citing", "following",
               "held", "the", "e.g", "eg", "viz", "accord", "compare", "whereas"}
_TRAIL_NOISE = {"the court", "this court", "the supreme court", "the hon'ble court",
                "court", "the bench", "the appellant", "the respondent", "the petitioner",
                "the case", "it", "held", "the judgment", "supra"}


def _clean_party(p: str) -> str:
    p = " ".join((p or "").split()).strip(" ,.;:")
    words = p.split()
    while words and words[0].lower().strip(".") in _LEAD_NOISE:
        words.pop(0)
    p = " ".join(words)
    low = p.lower()
    for tail in sorted(_TRAIL_NOISE, key=len, reverse=True):
        if low.endswith(" " + tail):
            p = p[: len(p) - len(tail) - 1].rstrip(" ,.;:")
            low = p.lower()
    return p.strip(" ,.;:")


def _name_overlap(a: str, b: str) -> float:
    """Token Jaccard between two normalised case names."""
    ta = {t for t in a.split() if len(t) > 3}
    tb = {t for t in b.split() if len(t) > 3}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# Reporter series our corpus can actually adjudicate. Everything in the
# judgments table is S.C.R.; Indian practice more often cites S.C.C. or A.I.R.
# Marking an S.C.C. citation "unverified" would tell a user that a real
# authority is fabricated — a different but equally serious failure — so those
# get their own status.
SUPPORTED_REPORTERS = ("scr",)
RE_REPORTER = re.compile(r"(scconline|scc|air|scr|insc|crilj|supp)", re.I)


def reporter_of(citation: str) -> str | None:
    m = RE_REPORTER.search(re.sub(r"[.\s]", "", citation or ""))
    return m.group(1).lower() if m else None


@dataclass
class Finding:
    kind: str            # citation | case | section | article
    text: str            # exactly as the answer wrote it
    # grounded | in_corpus | unverified | source_gap
    #   source_gap = our corpus cannot adjudicate this one either way
    status: str
    matched: str | None = None     # what it matched in the corpus
    detail: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class VerificationReport:
    findings: list[Finding] = field(default_factory=list)
    checked: int = 0
    grounded: int = 0
    in_corpus: int = 0
    unverified: int = 0
    source_gap: int = 0

    @property
    def adjudicable(self) -> int:
        """Authorities our corpus can actually rule on."""
        return self.checked - self.source_gap

    @property
    def hallucination_rate(self) -> float:
        """
        Share of ADJUDICABLE authorities that do not exist in the corpus.

        Deliberately excludes source_gap: counting "we cannot check this
        reporter series" as a hallucination would inflate the number and hide
        the real one.
        """
        return (self.unverified / self.adjudicable) if self.adjudicable else 0.0

    @property
    def grounded_rate(self) -> float:
        return (self.grounded / self.checked) if self.checked else 0.0

    def to_dict(self):
        return {
            "checked": self.checked, "grounded": self.grounded,
            "in_corpus": self.in_corpus, "unverified": self.unverified,
            "source_gap": self.source_gap, "adjudicable": self.adjudicable,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "grounded_rate": round(self.grounded_rate, 4),
            "findings": [f.to_dict() for f in self.findings],
        }


class CitationVerifier:
    """Thread-safe; holds one SQLite connection per thread."""

    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self._local = threading.local()
        self.available = self.db_path.exists()
        if not self.available:
            log.warning(f"authority DB missing at {self.db_path}; "
                        "citations cannot be verified — run scripts/07_build_authority_table.py")

    def _conn(self) -> sqlite3.Connection | None:
        if not self.available:
            return None
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(str(self.db_path), check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA busy_timeout=5000")
            self._local.conn = c
        return c

    # ── extraction ──────────────────────────────────────────────────────
    @staticmethod
    def extract(text: str) -> list[tuple[str, str]]:
        """Every authority the answer asserts, as (kind, verbatim text)."""
        out: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()

        def add(kind, val):
            val = " ".join((val or "").split())
            key = (kind, val.lower())
            if val and key not in seen:
                seen.add(key)
                out.append((kind, val))

        for m in RE_CITATION.finditer(text or ""):
            add("citation", m.group(0))
        for m in RE_CASE_NAME.finditer(text or ""):
            a, b = _clean_party(m.group(1)), _clean_party(m.group(2))
            if a.lower() in STOP_NAMES and b.lower() in STOP_NAMES:
                continue
            if len(a) < 3 or len(b) < 3:
                continue
            add("case", f"{a} v. {b}")
        for m in RE_SECTION.finditer(text or ""):
            num, act = m.group(1), (m.group(2) or "").strip(" ,.")
            add("section", f"Section {num}" + (f" of {act}" if act else ""))
        for m in RE_ARTICLE.finditer(text or ""):
            add("article", f"Article {m.group(1)}")
        return out

    # ── verification ────────────────────────────────────────────────────
    def verify(self, answer: str, retrieved_chunks: list[dict] | None = None) -> VerificationReport:
        chunks = retrieved_chunks or []
        blob = " ".join((c.get("text") or "") for c in chunks).lower()
        chunk_cites = {norm_citation(c.get("citation") or "") for c in chunks if c.get("citation")}
        chunk_names = {norm_name(c.get("title") or "") for c in chunks if c.get("title")}
        chunk_cites.discard("")
        chunk_names.discard("")

        report = VerificationReport()
        for kind, text in self.extract(answer):
            f = self._verify_one(kind, text, blob, chunk_cites, chunk_names)
            report.findings.append(f)
            report.checked += 1
            setattr(report, f.status, getattr(report, f.status) + 1)
        return report

    def _verify_one(self, kind, text, blob, chunk_cites, chunk_names) -> Finding:
        conn = self._conn()

        if kind == "citation":
            n = norm_citation(text)
            if n and any(n in cc or cc in n for cc in chunk_cites):
                return Finding(kind, text, "grounded", detail="cited in a retrieved passage")
            if conn and n:
                row = conn.execute(
                    "SELECT title, citation, decision_date FROM judgments "
                    "WHERE citation_norm = ? LIMIT 1", (n,)).fetchone()
                if not row:
                    row = conn.execute(
                        "SELECT title, citation, decision_date FROM judgments "
                        "WHERE citation_norm LIKE ? LIMIT 1", (f"%{n}%",)).fetchone()
                if row:
                    return Finding(kind, text, "in_corpus",
                                   matched=f"{row['title']} — {row['citation']}",
                                   detail="exists in the corpus but was not retrieved")
            rep = reporter_of(text)
            if rep and rep not in SUPPORTED_REPORTERS:
                return Finding(kind, text, "source_gap",
                               detail=f"our corpus indexes S.C.R. citations only, so a "
                                      f"{rep.upper()} citation cannot be checked either way")
            return Finding(kind, text, "unverified",
                           detail="no judgment in the corpus carries this citation")

        if kind == "case":
            n = norm_name(text)
            if n and any(n in cn or cn in n for cn in chunk_names):
                return Finding(kind, text, "grounded", detail="name of a retrieved judgment")
            if conn and n:
                # BOTH parties must match, and the distinctive one must not be
                # a ubiquitous respondent. Matching on the first party alone
                # blessed "Sharma v. Union of India" against an unrelated
                # judgment that merely had a Sharma in it — a false "verified"
                # on a fabricated case is the exact failure this guards.
                parts = [p.strip() for p in n.split(" v ") if p.strip()]
                if len(parts) == 2:
                    left, right = parts
                    generic = {"union of india", "india", "state", "the state",
                               "union", "govt of india", "government of india"}
                    lt = [t for t in left.split() if len(t) > 3]
                    rt = [t for t in right.split() if len(t) > 3]
                    # at least one side must carry a distinctive (non-generic) name
                    if (left in generic and right in generic) or not (lt or rt):
                        pass
                    else:
                        like_l = "%" + "%".join(lt[:3]) + "%" if lt else None
                        like_r = "%" + "%".join(rt[:3]) + "%" if rt else None
                        # Fetch several candidates and pick the best-scoring one.
                        # LIMIT 1 returned whichever row SQLite happened to hit
                        # first — for "Maneka Gandhi v. Union of India" that was
                        # "Buffalo Traders v. Maneka Gandhi", which scored 0.25
                        # and made a real landmark case look fabricated.
                        cands = []
                        if like_l and like_r and left not in generic and right not in generic:
                            cands = conn.execute(
                                "SELECT title, citation FROM judgments "
                                "WHERE title_norm LIKE ? AND title_norm LIKE ? LIMIT 25",
                                (like_l, like_r)).fetchall()
                        if not cands and like_l and left not in generic:
                            cands = conn.execute(
                                "SELECT title, citation FROM judgments "
                                "WHERE title_norm LIKE ? LIMIT 25", (like_l,)).fetchall()
                        if not cands and like_r and right not in generic:
                            cands = conn.execute(
                                "SELECT title, citation FROM judgments "
                                "WHERE title_norm LIKE ? LIMIT 25", (like_r,)).fetchall()
                        best, best_score = None, 0.0
                        for c in cands:
                            sc = _name_overlap(n, norm_name(c["title"]))
                            if sc > best_score:
                                best, best_score = c, sc
                        if best is not None and best_score >= 0.45:
                            return Finding(kind, text, "in_corpus",
                                           matched=f"{best['title']}"
                                                   + (f" — {best['citation']}" if best["citation"] else ""),
                                           detail=f"exists in the corpus (name match {best_score:.2f}) "
                                                  "but was not retrieved")
            return Finding(kind, text, "unverified",
                           detail="no judgment in the corpus matches this case name")

        # section / article
        m = re.search(r"(\d+[A-Z]{0,2})", text, re.I)
        num = (m.group(1).upper() if m else "")
        if num and re.search(rf"\b(?:section|article)\s+{re.escape(num)}\b", blob, re.I):
            return Finding(kind, text, "grounded", detail="appears in a retrieved passage")
        if conn and num:
            # If the answer named an act, the provision must exist in THAT act.
            # Matching on the number alone reported "Article 512" as real
            # because some unrelated text contained that token.
            act_m = re.search(r"\bof\s+(?:the\s+)?(.+)$", text, re.I)
            act_norm = norm_name(act_m.group(1)) if act_m else None
            if act_norm:
                toks = [t for t in act_norm.split() if len(t) > 3][:3]
                like = "%" + "%".join(toks) + "%" if toks else "%"
                row = conn.execute(
                    "SELECT act, kind, number FROM provisions "
                    "WHERE kind = ? AND number = ? AND act_norm LIKE ? LIMIT 1",
                    (kind, num, like)).fetchone()
                if row:
                    return Finding(kind, text, "in_corpus",
                                   matched=f"{row['act']} {row['kind']} {row['number']}",
                                   detail="exists in the corpus but was not retrieved")
                return Finding(kind, text, "unverified",
                               detail=f"no '{kind} {num}' found in the act named")
            if kind == "article":
                # Unqualified "Article N" means the Constitution of India.
                row = conn.execute(
                    "SELECT act, number FROM provisions WHERE kind='article' AND number=? "
                    "AND act_norm LIKE '%constitution%' LIMIT 1", (num,)).fetchone()
                if row:
                    return Finding(kind, text, "in_corpus", matched=f"{row['act']} Article {num}",
                                   detail="exists in the corpus but was not retrieved")
                held = conn.execute(
                    "SELECT COUNT(DISTINCT number) n FROM provisions "
                    "WHERE kind='article' AND act_norm LIKE '%constitution%'").fetchone()["n"]
                # We hold a small fraction of the Constitution's ~395 articles,
                # so absence here is not evidence of non-existence.
                if held < 200 and num.isdigit() and 1 <= int(num) <= 395:
                    return Finding(kind, text, "source_gap",
                                   detail=f"our corpus holds only {held} Constitution articles, "
                                          "so this one cannot be checked either way")
                return Finding(kind, text, "unverified",
                               detail="not an Article of the Constitution in our source data")
            row = conn.execute(
                "SELECT act, kind, number FROM provisions WHERE kind = ? AND number = ? LIMIT 1",
                (kind, num)).fetchone()
            if row:
                return Finding(kind, text, "in_corpus", matched=f"{row['act']} {row['kind']} {row['number']}",
                               detail="exists in the corpus but was not retrieved (act not named)")
        if conn:
            held = conn.execute("SELECT COUNT(*) n FROM provisions WHERE kind=?",
                                (kind,)).fetchone()["n"]
            if held < 5000:
                return Finding(kind, text, "source_gap",
                               detail=f"our provisions table holds only {held} {kind}s, "
                                      "so this one cannot be checked either way")
        return Finding(kind, text, "unverified",
                       detail="this provision was not found in the source material")

    # ── presentation ────────────────────────────────────────────────────
    @staticmethod
    def annotate(answer: str, report: VerificationReport) -> str:
        """
        Append an explicit verification notice. The answer text itself is left
        untouched — silently editing a legal answer would be its own hazard.
        """
        if not report.checked:
            return answer
        bad = [f for f in report.findings if f.status == "unverified"]
        weak = [f for f in report.findings if f.status == "in_corpus"]
        gap = [f for f in report.findings if f.status == "source_gap"]
        lines = ["", "---", "**Citation check**"]
        lines.append(
            f"- {report.grounded}/{report.checked} authorities are supported by the "
            f"passages retrieved for this answer.")
        if weak:
            lines.append("- Present in our source data but **not retrieved for this question**, "
                         "so the proposition attached to them is not evidenced here: "
                         + ", ".join(f"`{f.text}`" for f in weak[:6]))
        if gap:
            lines.append("- Outside what our source data can check (we index S.C.R. citations "
                         "and a subset of statutory provisions) — **not** a finding of error, "
                         "but verify independently: "
                         + ", ".join(f"`{f.text}`" for f in gap[:6]))
        if bad:
            lines.append("- ⚠️ **Could not be verified against our source judgments** — treat as "
                         "unreliable and check the official reports before relying on them: "
                         + ", ".join(f"`{f.text}`" for f in bad[:6]))
        return answer + "\n".join(lines)


_verifier: CitationVerifier | None = None
_lock = threading.Lock()


def get_verifier() -> CitationVerifier:
    """Process-wide singleton."""
    global _verifier
    with _lock:
        if _verifier is None:
            _verifier = CitationVerifier()
    return _verifier
