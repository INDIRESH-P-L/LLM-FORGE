#!/usr/bin/env python3
"""
storage/exporters.py
====================
Render a stored conversation for download — Markdown or plain text.

Law students paste these straight into case notes, so the authorities cited
across the whole conversation are collected, de-duplicated and numbered at the
bottom of the document rather than left scattered inline.

No SQL here: this module takes rows from chat_store and returns text.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from storage.chat_store import ConversationSummary, Message, slugify

#: Citation fields we render, in order, with their display labels.
_AUTHORITY_FIELDS = [
    ("court", "Court"),
    ("date", "Date"),
    ("judges", "Judges"),
    ("paragraph_no", "Paragraph"),
    ("source_id", "Source"),
]

_ROLE_LABELS = {
    "user": "Question",
    "assistant": "LegalMind AI",
    "system": "System",
}

_STATUS_NOTES = {
    "partial": "answer incomplete — generation was interrupted",
    "error": "generation failed",
}


def _normalise_citation(cite) -> dict | None:
    """
    Accept either shape of stored citation.

    /api/chat writes structured authorities (see chat_api.citations_from_result).
    Other writers may pass `run_rag()["citations"]` straight through, which is a
    list of pre-formatted strings like "[1] Bharatiya Nagarik Suraksha Sanhita".
    Both are legitimate rows in the database, so rendering must handle both
    rather than raising half-way through somebody's export.
    """
    if isinstance(cite, dict):
        return cite
    if isinstance(cite, str):
        text = cite.strip()
        if not text:
            return None
        # Drop a leading "[n] " marker; the list is renumbered on output anyway.
        return {"case_name": re.sub(r"^\[\d+\]\s*", "", text)}
    return None


def export_filename(conversation: ConversationSummary, fmt: str = "md") -> str:
    """`<date>-<slugified-title>.md` — date is the conversation's creation day."""
    day = (conversation.created_at or "")[:10] or datetime.now(timezone.utc).date().isoformat()
    ext = "txt" if fmt == "txt" else "md"
    return f"{day}-{slugify(conversation.title)}.{ext}"


def collect_authorities(messages: list[Message]) -> list[dict]:
    """
    Every distinct judgment or provision relied on, in first-cited order.

    De-duplication is by (case_name, citation, paragraph_no) because the same
    judgment can legitimately be cited at different paragraphs, and those are
    distinct authorities in a note.
    """
    seen: set[tuple] = set()
    authorities: list[dict] = []
    for msg in messages:
        for raw in msg.citations or []:
            cite = _normalise_citation(raw)
            if cite is None:
                continue
            key = (
                str(cite.get("case_name") or "").strip().lower(),
                str(cite.get("citation") or "").strip().lower(),
                str(cite.get("paragraph_no") or "").strip().lower(),
            )
            if key == ("", "", "") or key in seen:
                continue
            seen.add(key)
            authorities.append(cite)
    return authorities


def _authority_label(cite: dict) -> str:
    name = (cite.get("case_name") or "").strip()
    citation = (cite.get("citation") or "").strip()
    if name and citation and citation.lower() not in name.lower():
        return f"{name}, {citation}"
    return name or citation or (cite.get("source_id") or "Unattributed source")


def to_markdown(conversation: ConversationSummary, messages: list[Message]) -> str:
    out: list[str] = [f"# {conversation.title}", ""]
    out.append(f"- **Exported:** {_now_human()}")
    out.append(f"- **Started:** {_human(conversation.created_at)}")
    out.append(f"- **Last updated:** {_human(conversation.updated_at)}")
    if conversation.model_name:
        out.append(f"- **Model:** {conversation.model_name}")
    out.append(f"- **Messages:** {len(messages)}")
    out.append("")
    out.append("---")
    out.append("")

    for msg in messages:
        note = _STATUS_NOTES.get(msg.status)
        heading = f"## {_ROLE_LABELS.get(msg.role, msg.role.title())}"
        if note:
            heading += f" *({note})*"
        out.append(heading)
        out.append("")
        out.append(msg.content.strip() or "_(empty)_")
        out.append("")
        if msg.role == "assistant" and msg.citations:
            labels = [_authority_label(c) for c in
                      filter(None, (_normalise_citation(x) for x in msg.citations))]
            if labels:
                out.append(f"*Relied on:* {', '.join(labels)}")
                out.append("")

    authorities = collect_authorities(messages)
    out.append("---")
    out.append("")
    out.append("## Authorities")
    out.append("")
    if not authorities:
        out.append("_No authorities were recorded for this conversation._")
        out.append("")
    else:
        for i, cite in enumerate(authorities, start=1):
            out.append(f"{i}. **{_authority_label(cite)}**")
            details = [
                f"{label}: {str(cite.get(key)).strip()}"
                for key, label in _AUTHORITY_FIELDS
                if cite.get(key) not in (None, "", [])
            ]
            if details:
                out.append(f"   {' · '.join(details)}")
            excerpt = (cite.get("excerpt") or "").strip()
            if excerpt:
                out.append(f"   > {_collapse(excerpt)}")
            out.append("")

    out.append("---")
    out.append("")
    out.append("*Generated by LegalMind AI. Not a substitute for professional legal advice.*")
    out.append("")
    return "\n".join(out)


def to_text(conversation: ConversationSummary, messages: list[Message]) -> str:
    rule = "=" * 72
    out: list[str] = [rule, conversation.title, rule, ""]
    out.append(f"Exported:     {_now_human()}")
    out.append(f"Started:      {_human(conversation.created_at)}")
    out.append(f"Last updated: {_human(conversation.updated_at)}")
    if conversation.model_name:
        out.append(f"Model:        {conversation.model_name}")
    out.append(f"Messages:     {len(messages)}")
    out.append("")

    for msg in messages:
        label = _ROLE_LABELS.get(msg.role, msg.role.title()).upper()
        note = _STATUS_NOTES.get(msg.status)
        out.append("-" * 72)
        out.append(f"{label}{f'  [{note}]' if note else ''}  ({_human(msg.created_at)})")
        out.append("-" * 72)
        out.append(msg.content.strip() or "(empty)")
        out.append("")

    authorities = collect_authorities(messages)
    out.append(rule)
    out.append("AUTHORITIES")
    out.append(rule)
    out.append("")
    if not authorities:
        out.append("No authorities were recorded for this conversation.")
        out.append("")
    else:
        for i, cite in enumerate(authorities, start=1):
            out.append(f"[{i}] {_authority_label(cite)}")
            for key, label in _AUTHORITY_FIELDS:
                value = cite.get(key)
                if value not in (None, "", []):
                    out.append(f"    {label}: {str(value).strip()}")
            excerpt = (cite.get("excerpt") or "").strip()
            if excerpt:
                out.append(f'    "{_collapse(excerpt)}"')
            out.append("")

    out.append(rule)
    out.append("Generated by LegalMind AI. Not a substitute for professional legal advice.")
    out.append("")
    return "\n".join(out)


def render(conversation: ConversationSummary, messages: list[Message],
           fmt: str = "md") -> tuple[str, str, str]:
    """Return (filename, media_type, body) for the requested format."""
    if fmt == "txt":
        return export_filename(conversation, "txt"), "text/plain; charset=utf-8", \
            to_text(conversation, messages)
    return export_filename(conversation, "md"), "text/markdown; charset=utf-8", \
        to_markdown(conversation, messages)


def _collapse(text: str, limit: int = 400) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "…"


def _human(iso: str | None) -> str:
    if not iso:
        return "unknown"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return iso


def _now_human() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
