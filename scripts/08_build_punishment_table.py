#!/usr/bin/env python3
"""
scripts/08_build_punishment_table.py
====================================
Extract the maximum punishment for penal sections, into authorities.db.

Why this is needed: the Arnesh Kumar guidelines (and s.35 BNSS / s.41A CrPC)
apply to offences punishable with imprisonment of **up to seven years**. The
FIR auditor previously asserted "Arrest violates Arnesh Kumar guidelines
(punishment <= 7 years)" for every FIR where the arrest box was ticked —
without ever determining what the invoked sections are punishable with. That
is a legal conclusion stated with no basis, which is exactly the kind of thing
this tool must not do.

This builds `section_punishments`, so the auditor can say "s.318(4) BNS carries
up to 7 years, so Arnesh Kumar applies" — or decline to reach that conclusion
when the punishment is unknown.

    .venv/bin/python scripts/08_build_punishment_table.py
"""
from __future__ import annotations
import json, logging, re, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data/verification/authorities.db"
LEG = ROOT / "data/chunks/legislation/chunks.jsonl"

log = logging.getLogger("punishment")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")

WORD_YEARS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "twelve": 12, "fourteen": 14, "twenty": 20,
}

# Ordered: the gravest formulation present in the section wins.
PAT_LIFE = re.compile(r"imprisonment\s+for\s+life|remainder\s+of\s+that\s+person'?s?\s+natural\s+life", re.I)
PAT_DEATH = re.compile(r"\bpunish\w*\s+with\s+death\b|\bwith\s+death\b", re.I)
PAT_YEARS = re.compile(
    r"imprisonment[^.]{0,120}?(?:which\s+may\s+extend\s+to|for\s+a\s+term\s+of|not\s+less\s+than)?\s*"
    r"(\d{1,2}|" + "|".join(WORD_YEARS) + r")\s*years?", re.I)

#: Sections in a procedure code (BNSS/CrPC) are not offences; skip them.
PROCEDURAL_ACTS = ("nagarik suraksha", "criminal procedure", "civil procedure", "evidence")


#: A punishment named as the *subject* of an offence rather than its penalty.
#: BNS s.351(3) reads "...to cause an offence punishable with death or
#: imprisonment for life..." — that describes the THREATENED offence, not the
#: penalty for criminal intimidation. Reading it as the penalty recorded s.351
#: (2-7 years) as a life offence.
DESCRIPTIVE = re.compile(
    r"(?:an?\s+offence\s+punishable\s+with|offence\s+punishable\s+with|"
    r"punishable\s+with\s+death\s+or\s+imprisonment\s+for\s+life\s*,?\s*or)", re.I)


def _is_descriptive(text: str, start: int) -> bool:
    """True if the match sits inside a description of some other offence."""
    window = text[max(0, start - 90):start + 10]
    return bool(DESCRIPTIVE.search(window))


def max_years(text: str) -> tuple[int | None, str | None, str]:
    """
    Return (max years, evidence phrase, confidence).

    life/death -> 1000. confidence is 'high' when the figure came from an
    operative punishment clause, 'low' when it was the largest number found
    without one — the FIR auditor refuses to draw conclusions from 'low'.
    """
    if not text:
        return None, None, "none"

    for pat in (PAT_DEATH, PAT_LIFE):
        for m in pat.finditer(text):
            if _is_descriptive(text, m.start()):
                continue
            # operative clauses read "shall be punished with ..."
            lead = text[max(0, m.start() - 60):m.start()]
            conf = "high" if re.search(r"shall\s+be\s+punish\w*|be\s+punished\s+with", lead, re.I) \
                else "medium"
            return 1000, m.group(0)[:90], conf

    best, phrase, conf = None, None, "low"
    for m in PAT_YEARS.finditer(text):
        if _is_descriptive(text, m.start()):
            continue
        tok = m.group(1).lower()
        y = int(tok) if tok.isdigit() else WORD_YEARS.get(tok)
        if y is None:
            continue
        lead = text[max(0, m.start() - 70):m.start()]
        this_conf = "high" if re.search(r"shall\s+be\s+punish\w*|be\s+punished\s+with|punishable\s+with",
                                        lead, re.I) else "low"
        if best is None or y > best:
            best, phrase, conf = y, m.group(0)[:90], this_conf
    return best, phrase, conf


def build():
    conn = sqlite3.connect(str(DB))
    conn.executescript("""
        DROP TABLE IF EXISTS section_punishments;
        CREATE TABLE section_punishments (
            act        TEXT NOT NULL,
            act_norm   TEXT NOT NULL,
            section    TEXT NOT NULL,
            max_years  INTEGER,      -- 1000 = life or death
            phrase     TEXT,         -- the words the figure came from
            confidence TEXT,         -- high | medium | low
            source     TEXT,
            PRIMARY KEY (act_norm, section)
        );
        CREATE INDEX idx_pun_section ON section_punishments(section);
    """)

    rows: dict[tuple[str, str], tuple] = {}
    if LEG.exists():
        with LEG.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                act = (d.get("title") or "").strip()
                if not act or any(p in act.lower() for p in PROCEDURAL_ACTS):
                    continue
                text = d.get("text") or ""
                # Attribution must be UNAMBIGUOUS. A chunk that spans several
                # sections cannot tell us which one carries which punishment:
                # an early version read "imprisonment for life" from a
                # neighbouring section and recorded BNS s.351 (criminal
                # intimidation, 2-7 years) as a life offence. An overstated
                # punishment silently disables the Arnesh Kumar check, so only
                # single-section chunks are trusted.
                headings = set(re.findall(r"Section\s+(\d{1,3}[A-Z]?)\b", text, re.I))
                explicit = d.get("section")
                secs = {str(explicit)} if explicit else set()
                if not secs:
                    if len(headings) != 1:
                        continue                 # ambiguous or none: skip
                    secs = headings
                elif headings - {str(explicit)}:
                    continue                     # other sections also present
                y, phrase, conf = max_years(text)
                if y is None:
                    continue
                for s in secs:
                    s = s.upper()
                    key = (re.sub(r"[^a-z0-9 ]", " ", act.lower()).strip(), s)
                    if key not in rows or (rows[key][3] or 0) < y:
                        rows[key] = (act, key[0], s, y, phrase, conf, d.get("source"))

    conn.executemany("INSERT OR REPLACE INTO section_punishments "
                     "(act, act_norm, section, max_years, phrase, confidence, source) "
                     "VALUES (?,?,?,?,?,?,?)", list(rows.values()))
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM section_punishments").fetchone()[0]
    le7 = conn.execute("SELECT COUNT(*) FROM section_punishments WHERE max_years<=7").fetchone()[0]
    log.info(f"section_punishments: {n:,} rows ({le7:,} punishable with <= 7 years)")
    for r in conn.execute("SELECT act, section, max_years, phrase FROM section_punishments "
                          "ORDER BY RANDOM() LIMIT 5"):
        log.info(f"   {r[0][:34]:<34} s.{r[1]:<5} max={r[2]:<5} {str(r[3])[:52]!r}")
    conn.close()
    return n


if __name__ == "__main__":
    build()
