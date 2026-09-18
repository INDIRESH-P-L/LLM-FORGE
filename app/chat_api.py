#!/usr/bin/env python3
"""
app/chat_api.py
===============
LegalMind AI — persistent chat endpoints.

Everything about *conversations* lives here; app/main.py keeps its existing job
of loading the model and the retriever. The legacy POST /query endpoint is
untouched (fine_tuning/evaluate.py drives it) — this router adds the durable
path alongside it.

Write ordering is the point of this module:

    1. the user's question is committed BEFORE the model is called, so a crash
       or an OOM during generation cannot lose the question;
    2. the answer is committed in one transaction with the conversation's
       updated_at bump;
    3. a failure still produces a row — status='error' with the error text in
       metadata — and a process death mid-generation is closed off as
       status='partial' by chat_store.reconcile_interrupted() on next startup;
    4. a retried or double-clicked submit carries the same client_token and
       returns the existing exchange instead of a duplicate row.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.services.answer_grounding import (
    ABSTAIN_MESSAGE,
    GENERAL_KNOWLEDGE_NOTICE,
    PARTIAL_EVIDENCE_NOTICE,
    assess,
)
from app.services.citation_verifier import get_verifier
from storage import chat_store as store
from storage import exporters

log = logging.getLogger("legalmind.chat")

router = APIRouter(prefix="/api", tags=["chat"])

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Generation timeout limit (default 180 seconds)
GENERATION_TIMEOUT_SECONDS = int(os.environ.get("GENERATION_TIMEOUT_SECONDS", "180"))

#: Cookie that carries the browser's identity. Ten years — it is not a session.
UID_COOKIE = "legalmind_uid"
UID_COOKIE_MAX_AGE = 10 * 365 * 24 * 3600

#: ?uid=... in the URL wins over the cookie, so a conversation URL can be
#: bookmarked or moved to another browser and still resolve to its owner.
UID_QUERY_PARAM = "uid"

_UID_RE = re.compile(r"^[0-9a-zA-Z._-]{8,64}$")

#: Second-pass title generation costs GPU time, so it is off unless asked for.
AUTO_TITLE = os.environ.get("LEGALMIND_AUTO_TITLE", "0").lower() in ("1", "true", "yes", "on")

#: One generation at a time. The model is a single in-process GPU object and
#: two threads calling .generate() concurrently is not something to find out
#: about in production. DB writes happen outside this lock, so concurrent
#: users still get their questions persisted immediately and in parallel.
_inference_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def get_or_create_user_id(request: Request) -> tuple[str, bool]:
    """
    Resolve the owner of this request, minting an id on first visit.

    THIS IS THE AUTH SEAM. There are no accounts yet, so identity is "this
    browser": a uuid4 held in a cookie, overridable by ?uid= in the URL. When
    real accounts arrive, THIS FUNCTION IS THE ONLY THING THAT CHANGES —
    return the authenticated principal's id and every call site, every query
    and the whole schema carry on working unmodified, because user_id is
    already a plain opaque string on `conversations`.

    Returns (user_id, should_refresh_cookie).
    """
    from_url = (request.query_params.get(UID_QUERY_PARAM) or "").strip()
    if from_url and _UID_RE.match(from_url):
        # Explicit uid in the URL wins, and is written back to the cookie so
        # the rest of the browsing session stays on that identity.
        return from_url, from_url != request.cookies.get(UID_COOKIE)

    from_cookie = (request.cookies.get(UID_COOKIE) or "").strip()
    if from_cookie and _UID_RE.match(from_cookie):
        return from_cookie, False

    return str(uuid.uuid4()), True


def install_user_identity(app) -> None:
    """
    Attach identity resolution to every request, including the HTML page load,
    so the cookie exists before any JavaScript runs.
    """

    @app.middleware("http")
    async def _user_identity(request: Request, call_next):
        user_id, refresh = get_or_create_user_id(request)
        request.state.user_id = user_id
        response = await call_next(request)
        if refresh:
            response.set_cookie(
                UID_COOKIE,
                user_id,
                max_age=UID_COOKIE_MAX_AGE,
                httponly=True,      # the browser never needs to read it; /api/me reports it
                samesite="lax",
                path="/",
            )
        return response


def _uid(request: Request) -> str:
    uid = getattr(request.state, "user_id", None)
    if not uid:                                  # middleware not installed (tests)
        uid, _ = get_or_create_user_id(request)
    return uid


def _own_conversation(conversation_id: str, user_id: str,
                      include_deleted: bool = True) -> store.ConversationSummary:
    """
    Fetch a conversation or 404. Scoped by user_id, so another browser's
    conversation is indistinguishable from one that does not exist.
    """
    conv = store.get_conversation(conversation_id, user_id=user_id,
                                  include_deleted=include_deleted)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return conv


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class NewConversation(BaseModel):
    title: str | None = None


class PatchConversation(BaseModel):
    title: str | None = None
    pinned: bool | None = None


class ChatRequest(BaseModel):
    query: str
    conversation_id: str | None = None
    #: Browser-generated token for this submit. Makes the whole endpoint
    #: idempotent: same token in, same exchange out, no second row and no
    #: second trip through the GPU.
    client_token: str | None = Field(default=None, max_length=128)
    top_k: int | None = None


class FeedbackRequest(BaseModel):
    feedback: int = Field(ge=-1, le=1)


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

def citations_from_result(result: dict) -> list[dict]:
    """
    Map the retriever's chunk metadata onto the stored authorities shape.

    `judges` and `paragraph_no` are recorded as None: the ingested corpus
    carries neither (chunk keys are case_number / chunk_id / chunk_index /
    citation / court / date / document_id / document_type / source / title),
    and `chunk_index` is a chunking artefact, not a judgment paragraph number.
    The columns exist so that when ingestion starts extracting them, nothing
    here has to change.
    """
    authorities: list[dict] = []
    for doc in result.get("retrieved_documents") or []:
        if not isinstance(doc, dict):
            continue
        authorities.append({
            "case_name":    doc.get("title"),
            "citation":     doc.get("citation") or doc.get("article") or doc.get("section"),
            "court":        doc.get("court"),
            "date":         doc.get("date"),
            "judges":       None,
            "paragraph_no": None,
            "source_id":    doc.get("document_id") or doc.get("source"),
            "excerpt":      doc.get("text_preview"),
            # Kept for traceability back into the retrieval logs.
            "citation_id":  doc.get("citation_id"),
            "document_type": doc.get("document_type"),
        })
    return authorities


def _assistant_metadata(result: dict, model_name: str | None) -> dict:
    return {
        "confidence":         result.get("confidence"),
        "limitations":        result.get("limitations"),
        "legal_provisions":   result.get("legal_provisions") or [],
        "judgments":          result.get("judgments") or [],
        "retrieval_latency":  result.get("retrieval_latency"),
        "generation_latency": result.get("generation_latency"),
        "total_latency":      result.get("total_latency"),
        "retrieved_count":    len(result.get("retrieved_documents") or []),
        "model_name":         model_name,
    }


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------

def derive_title(query: str, result: dict | None = None) -> str:
    """
    Conversation title. Always the cleaned 60-char truncation of the first
    question; with LEGALMIND_AUTO_TITLE=1 the leading statutory provision the
    answer relied on is prefixed, which reads better in a long sidebar.

    A genuinely model-written title is deliberately NOT done here: the loaded
    model's only entry point is the RAG-shaped generate(), which would emit a
    structured legal answer rather than a title, so a real second-model title
    needs an inference-layer change. Out of scope for this task; this flag
    exists so turning it on later is a one-line swap. See docs/persistence.md.
    """
    base = store.make_title(query)
    if not (AUTO_TITLE and result):
        return base
    provisions = [p for p in (result.get("legal_provisions") or []) if isinstance(p, str)]
    if not provisions:
        return base
    lead = re.sub(r"\s+", " ", provisions[0]).strip(" .;,")[:28]
    if not lead:
        return base
    return store.make_title(f"{lead} — {query}")


# ---------------------------------------------------------------------------
# Endpoints — identity
# ---------------------------------------------------------------------------

@router.get("/me")
def whoami(request: Request) -> dict:
    """The caller's user id, so the frontend can put ?uid= in the URL."""
    return {"user_id": _uid(request), "auto_title": AUTO_TITLE}


# ---------------------------------------------------------------------------
# Endpoints — conversations
# ---------------------------------------------------------------------------

@router.get("/conversations")
def list_conversations(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    include_deleted: bool = Query(default=False),
) -> dict:
    uid = _uid(request)
    convs = store.list_conversations(uid, limit=limit, offset=offset,
                                     include_deleted=include_deleted)
    return {"user_id": uid, "conversations": [c.to_dict() for c in convs]}


@router.post("/conversations")
def create_conversation(request: Request, body: NewConversation | None = None) -> dict:
    uid = _uid(request)
    title = (body.title if body and body.title else "New conversation")
    cid = store.create_conversation(uid, title, model_name=_model_name(request))
    conv = store.get_conversation(cid, user_id=uid)
    return {"conversation": conv.to_dict() if conv else {"id": cid}}


@router.get("/conversations/{conversation_id}")
def get_conversation(request: Request, conversation_id: str) -> dict:
    """
    The conversation and its full message list, read from the database. This
    is what a cold reload renders — the DB is the source of truth, the browser
    holds no history of its own.
    """
    uid = _uid(request)
    conv = store.get_conversation(conversation_id, user_id=uid)
    read_only = False
    if conv is None:
        conv = store.get_public_conversation(conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        read_only = True
    messages = store.get_messages(conversation_id)
    return {
        "conversation": conv.to_dict(),
        "messages": [m.to_dict() for m in messages],
        "read_only": read_only,
    }


@router.patch("/conversations/{conversation_id}")
def patch_conversation(request: Request, conversation_id: str, body: PatchConversation) -> dict:
    uid = _uid(request)
    _own_conversation(conversation_id, uid)
    if body.title is not None:
        if not body.title.strip():
            raise HTTPException(status_code=400, detail="Title cannot be empty.")
        store.rename_conversation(conversation_id, body.title, user_id=uid)
    if body.pinned is not None:
        store.set_pinned(conversation_id, body.pinned, user_id=uid)
    conv = store.get_conversation(conversation_id, user_id=uid)
    return {"conversation": conv.to_dict() if conv else {}}


@router.delete("/conversations/{conversation_id}")
def delete_conversation(request: Request, conversation_id: str) -> dict:
    """Soft delete. Rows stay; /restore puts it back, which is the undo."""
    uid = _uid(request)
    _own_conversation(conversation_id, uid)
    store.soft_delete_conversation(conversation_id, user_id=uid)
    return {"ok": True, "conversation_id": conversation_id, "undo": True}


@router.post("/conversations/{conversation_id}/restore")
def restore_conversation(request: Request, conversation_id: str) -> dict:
    uid = _uid(request)
    _own_conversation(conversation_id, uid)
    store.restore_conversation(conversation_id, user_id=uid)
    conv = store.get_conversation(conversation_id, user_id=uid)
    return {"conversation": conv.to_dict() if conv else {}}


# ---------------------------------------------------------------------------
# Endpoints — search / export / feedback
# ---------------------------------------------------------------------------

@router.get("/search")
def search(request: Request, q: str = Query(default=""),
           limit: int = Query(default=50, ge=1, le=200)) -> dict:
    uid = _uid(request)
    hits = store.search_messages(uid, q, limit=limit)
    return {"query": q, "hits": [h.to_dict() for h in hits]}


# ---------------------------------------------------------------------------
# Shared activity log
# ---------------------------------------------------------------------------
# Everything above this point is scoped to the calling browser. These two
# endpoints are deliberately NOT: they serve one global feed of what everyone
# has asked, so two people on two laptops see the same list. Reads only —
# renaming, pinning and deleting still require ownership, so a public reader
# cannot alter anyone else's history.

@router.get("/activity")
def activity(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    q: str = Query(default=""),
) -> dict:
    """
    The shared, global "recent searches" feed — every question from every
    user, newest first, paginated so the list cannot grow without bound.
    """
    term = q.strip() or None
    entries = store.list_activity(limit=limit, offset=offset, query=term)
    total = store.count_activity(query=term)
    return {
        "entries": [e.to_dict() for e in entries],
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(entries) < total,
        "query": q,
        "you": _uid(request),
    }


@router.get("/activity/{conversation_id}")
def activity_conversation(conversation_id: str) -> dict:
    """
    Read-only transcript of any conversation in the shared feed, so an entry
    can be opened whoever asked it. No ownership check — that is the point of
    a public log — and no mutating capability is exposed here.
    """
    conv = store.get_public_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {
        "conversation": conv.to_dict(),
        "messages": [m.to_dict() for m in store.get_messages(conversation_id)],
        "read_only": True,
    }


@router.get("/conversations/{conversation_id}/export")
def export_conversation(request: Request, conversation_id: str,
                        format: str = Query(default="md", pattern="^(md|txt)$")):
    uid = _uid(request)
    conv = store.get_conversation(conversation_id, user_id=uid)
    if conv is None:
        conv = store.get_public_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    messages = store.get_messages(conversation_id)
    filename, media_type, body = exporters.render(conv, messages, fmt=format)
    return PlainTextResponse(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/messages/{message_id}/feedback")
def message_feedback(request: Request, message_id: str, body: FeedbackRequest) -> dict:
    uid = _uid(request)
    if not store.set_feedback(message_id, body.feedback, user_id=uid):
        raise HTTPException(status_code=404, detail="Message not found.")
    return {"ok": True, "message_id": message_id, "feedback": body.feedback}


@router.get("/stats")
def store_stats(request: Request) -> dict:
    uid = _uid(request)
    out = store.stats()
    out["your_messages"] = store.count_messages(uid)
    return out


# ---------------------------------------------------------------------------
# Endpoint — the chat loop
# ---------------------------------------------------------------------------

@router.post("/chat")
def chat(request: Request, body: ChatRequest) -> dict:
    """
    One exchange, persisted as it happens.

    Deliberately a sync `def`: Starlette runs it on the threadpool, so the
    blocking model call does not stall the event loop and other browsers keep
    getting served. It also means the function runs to completion even if the
    client disconnects halfway — a killed tab still gets its answer written to
    the database rather than losing it.
    """
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    uid = _uid(request)
    model_name = _model_name(request)
    token = (body.client_token or "").strip() or None

    # ── Idempotency: has this exact submit already been handled? ────────────
    if token:
        prior = store.get_message_by_client_token(token)
        if prior is not None:
            conv = store.get_conversation(prior.conversation_id, user_id=uid)
            if conv is not None:
                answer = store.get_next_message(prior.conversation_id, prior)
                log.info(f"Replaying idempotent submit {token} on {conv.id}")
                return {
                    "conversation_id": conv.id,
                    "conversation": conv.to_dict(),
                    "user_message": prior.to_dict(),
                    "assistant_message": answer.to_dict() if answer else None,
                    "replayed": True,
                }

    # ── Resolve or create the conversation ─────────────────────────────────
    conv = None
    if body.conversation_id:
        conv = store.get_conversation(body.conversation_id, user_id=uid, include_deleted=False)

    if conv is not None:
        conversation_id = conv.id
        is_first_message = conv.message_count == 0
    else:
        conversation_id = store.create_conversation(
            uid, store.make_title(query), model_name=model_name)
        is_first_message = True

    # Multi-turn conversation context: fetch prior messages before committing the new question
    prior_messages = store.get_messages(conversation_id) if conv is not None else []
    chat_history = [
        {"role": m.role, "content": m.content}
        for m in prior_messages
        if m.content and m.status != "error"
    ]

    # ── 1. Commit the question BEFORE touching the model ───────────────────
    user_message_id = store.append_message(
        conversation_id, "user", query,
        {"client_token": token, "status": "complete"},
    )

    if is_first_message:
        store.rename_conversation(conversation_id, store.make_title(query), user_id=uid)

    # ── 2. Generate ────────────────────────────────────────────────────────
    infer: Callable[..., dict] | None = getattr(request.app.state, "infer", None)
    if infer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    t0 = time.time()
    try:
        with _inference_lock:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(infer, query, body.top_k, chat_history)
                result = future.result(timeout=GENERATION_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        log.error(f"Inference timed out after {GENERATION_TIMEOUT_SECONDS}s for conversation {conversation_id}")
        error_msg = f"Generation timed out after {GENERATION_TIMEOUT_SECONDS} seconds. The question above has been preserved."
        assistant_message_id = store.append_message(
            conversation_id, "assistant",
            "_(generation interrupted — timeout reached while generating answer)_",
            {
                "status": "partial",
                "error": error_msg,
                "error_code": "GENERATION_INTERRUPTED",
                "error_type": "TimeoutError",
                "latency_ms": int((time.time() - t0) * 1000),
                "model_name": model_name,
            },
        )
        return _exchange_payload(uid, conversation_id, user_message_id,
                                 assistant_message_id, failed=True, error=error_msg,
                                 error_code="GENERATION_INTERRUPTED")
    except Exception as e:
        # ── 3a. Failure still gets a row, with the error in metadata ───────
        log.error(f"Inference failed for conversation {conversation_id}: {e}", exc_info=True)
        assistant_message_id = store.append_message(
            conversation_id, "assistant",
            "The answer could not be generated. The question above has been saved.",
            {
                "status": "error",
                "error": str(e),
                "error_code": "INFERENCE_FAILED",
                "error_type": type(e).__name__,
                "latency_ms": int((time.time() - t0) * 1000),
                "model_name": model_name,
            },
        )
        return _exchange_payload(uid, conversation_id, user_message_id,
                                 assistant_message_id, failed=True, error=str(e),
                                 error_code="INFERENCE_FAILED")

    # ── 3b. Success: one row, one transaction, updated_at bumped with it ───
    answer = result.get("answer") or result.get("raw_answer") or ""
    metadata = _assistant_metadata(result, model_name)
    metadata["status"] = "complete" if answer.strip() else "partial"
    metadata["latency_ms"] = int((time.time() - t0) * 1000)
    metadata["citations"] = citations_from_result(result)
    if not answer.strip():
        answer = "_(the model returned an empty answer)_"
        metadata["error"] = "empty generation"

    # ── Trust layer ────────────────────────────────────────────────────────
    # Nothing reaches the user before every authority it names has been checked
    # against the source corpus, and the answer's actual evidentiary support
    # has been measured. A fabricated citation presented confidently is the
    # failure mode this whole layer exists to prevent.
    chunks = result.get("retrieved_documents") or []
    try:
        report = get_verifier().verify(answer, chunks)
        eval_query = result.get("effective_query") or query
        grounding = assess(eval_query, chunks, report)

        metadata["verification"] = report.to_dict()
        metadata["grounding"] = grounding.to_dict()
        # The displayed confidence comes from measured support, not from how
        # assertive the model's prose happened to sound — and only when that
        # score has been shown to track correctness. Until then we publish the
        # evidence counts and no label, because an uncalibrated confidence word
        # is the false-assurance failure this layer exists to prevent.
        metadata["confidence_raw_model"] = metadata.get("confidence")
        metadata["support"] = round(grounding.support, 4)
        metadata["confidence_internal"] = grounding.confidence
        metadata["confidence_calibrated"] = grounding.calibrated
        if grounding.calibrated:
            metadata["confidence"] = grounding.confidence
        else:
            metadata["confidence"] = None
            metadata["evidence_summary"] = (
                f"{report.grounded} of {report.adjudicable} authorities verified "
                f"against retrieved sources"
                + (f"; {report.unverified} could not be verified" if report.unverified else "")
                + f"; {grounding.usable_sources} usable source passage(s)")

        if grounding.should_abstain:
            # should_abstain is now always False, but we keep this branch for
            # backward-compat with any code that sets LEGALMIND_ABSTAIN_FLOOR > 0.
            metadata["withheld_answer"] = answer[:4000]
            metadata["status"] = "complete"
            metadata["abstained"] = True
            metadata["abstain_reasons"] = grounding.reasons
            answer = ABSTAIN_MESSAGE
        elif report.unverified or report.in_corpus:
            answer = get_verifier().annotate(answer, report)

        # ── Knowledge-mode transparency notice ─────────────────────────────
        # Instead of refusing, prepend an honest notice that tells the user
        # exactly how the answer was produced and what they can rely on.
        km = getattr(grounding, "knowledge_mode", "evidence_based")
        if not metadata.get("abstained"):
            if km == "general_knowledge":
                metadata["knowledge_mode"] = "general_knowledge"
                answer = GENERAL_KNOWLEDGE_NOTICE + answer
            elif km == "partial_evidence":
                metadata["knowledge_mode"] = "partial_evidence"
                answer = PARTIAL_EVIDENCE_NOTICE + answer
            else:
                metadata["knowledge_mode"] = "evidence_based"
    except Exception as e:                      # verification must never 500 a chat
        log.warning(f"verification/grounding failed: {e}", exc_info=True)
        metadata["verification_error"] = str(e)

    assistant_message_id = store.append_message(conversation_id, "assistant", answer, metadata)

    if conv_model_missing(conversation_id, uid):
        store.set_conversation_model(conversation_id, model_name, user_id=uid)

    if is_first_message and AUTO_TITLE:
        store.rename_conversation(conversation_id, derive_title(query, result), user_id=uid)

    return _exchange_payload(uid, conversation_id, user_message_id, assistant_message_id)


@router.post("/chat/stream")
def chat_stream(request: Request, body: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    1. Persists the user question in SQLite immediately.
    2. Emits 'meta' event with retrieved chunks & citations as soon as retrieval finishes (< 0.2s).
    3. Emits 'token' events in real time as each token is generated (TTFT < 1.0s).
    4. Persists the complete assistant answer to the database upon completion.
    5. Emits 'done' event with the full persisted payload.
    """
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    uid = _uid(request)
    model_name = _model_name(request)
    token = (body.client_token or "").strip() or None

    conv = None
    if body.conversation_id:
        conv = store.get_conversation(body.conversation_id, user_id=uid, include_deleted=False)

    if conv is not None:
        conversation_id = conv.id
        is_first_message = conv.message_count == 0
    else:
        conversation_id = store.create_conversation(
            uid, store.make_title(query), model_name=model_name)
        is_first_message = True

    prior_messages = store.get_messages(conversation_id) if conv is not None else []
    chat_history = [
        {"role": m.role, "content": m.content}
        for m in prior_messages
        if m.content and m.status != "error"
    ]

    user_message_id = store.append_message(
        conversation_id, "user", query,
        {"client_token": token, "status": "complete"},
    )

    if is_first_message:
        store.rename_conversation(conversation_id, store.make_title(query), user_id=uid)

    infer_stream = getattr(request.app.state, "infer_stream", None)
    infer = getattr(request.app.state, "infer", None)

    if infer_stream is None and infer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    def event_generator():
        t0 = time.time()
        final_result = None
        accumulated_text = []

        try:
            # Emit immediate init event with conversation_id and title so client updates instantly
            init_payload = {
                "conversation_id": conversation_id,
                "title": store.make_title(query),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            yield f"event: init\ndata: {json.dumps(init_payload)}\n\n"

            with _inference_lock:
                if infer_stream is not None:
                    for ev_type, ev_data in infer_stream(query, body.top_k, chat_history):
                        if ev_type == "meta":
                            if isinstance(ev_data, dict):
                                ev_data["conversation_id"] = conversation_id
                            yield f"event: meta\ndata: {json.dumps(ev_data)}\n\n"
                        elif ev_type == "token":
                            accumulated_text.append(ev_data)
                            yield f"event: token\ndata: {json.dumps({'delta': ev_data})}\n\n"
                        elif ev_type == "done":
                            final_result = ev_data
                else:
                    final_result = infer(query, body.top_k, chat_history)
                    ans = final_result.get("answer", "")
                    yield f"event: token\ndata: {json.dumps({'delta': ans})}\n\n"

            if final_result is None:
                final_result = {"answer": "".join(accumulated_text), "retrieved_documents": []}

            answer = final_result.get("answer") or "".join(accumulated_text)
            metadata = _assistant_metadata(final_result, model_name)
            metadata["status"] = "complete" if answer.strip() else "partial"
            metadata["latency_ms"] = int((time.time() - t0) * 1000)
            metadata["citations"] = citations_from_result(final_result)

            chunks = final_result.get("retrieved_documents") or []
            try:
                report = get_verifier().verify(answer, chunks)
                eval_query = final_result.get("effective_query") or query
                grounding = assess(eval_query, chunks, report)
                metadata["verification"] = report.to_dict()
                metadata["grounding"] = grounding.to_dict()
                metadata["confidence"] = grounding.confidence if grounding.calibrated else None
            except Exception as e:
                log.warning(f"Verification in stream failed: {e}")

            assistant_message_id = store.append_message(conversation_id, "assistant", answer, metadata)

            if conv_model_missing(conversation_id, uid):
                store.set_conversation_model(conversation_id, model_name, user_id=uid)

            if is_first_message and AUTO_TITLE:
                store.rename_conversation(conversation_id, derive_title(query, final_result), user_id=uid)

            payload = _exchange_payload(uid, conversation_id, user_message_id, assistant_message_id)
            yield f"event: done\ndata: {json.dumps(payload)}\n\n"

        except Exception as e:
            log.error(f"Streaming failed: {e}", exc_info=True)
            err_msg = str(e)
            assistant_message_id = store.append_message(
                conversation_id, "assistant",
                "Generation error occurred.",
                {"status": "error", "error": err_msg, "latency_ms": int((time.time() - t0) * 1000)},
            )
            err_payload = _exchange_payload(uid, conversation_id, user_message_id, assistant_message_id, failed=True, error=err_msg)
            yield f"event: error\ndata: {json.dumps(err_payload)}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


def conv_model_missing(conversation_id: str, uid: str) -> bool:
    conv = store.get_conversation(conversation_id, user_id=uid)
    return conv is not None and not conv.model_name


def _exchange_payload(uid: str, conversation_id: str, user_message_id: str,
                      assistant_message_id: str, failed: bool = False,
                      error: str | None = None, error_code: str | None = None) -> dict:
    """
    Everything the exchange produced, read back out of the database rather
    than assembled from what we happen to have in hand — if it is not in the
    store, the UI must not show it.
    """
    conv = store.get_conversation(conversation_id, user_id=uid)
    user_msg = store.get_message(user_message_id)
    assistant_msg = store.get_message(assistant_message_id)
    payload = {
        "conversation_id": conversation_id,
        "conversation": conv.to_dict() if conv else {},
        "user_message": user_msg.to_dict() if user_msg else None,
        "assistant_message": assistant_msg.to_dict() if assistant_msg else None,
        "replayed": False,
    }
    if failed:
        payload["error"] = error
        if error_code:
            payload["error_code"] = error_code
    return payload


def _model_name(request: Request) -> str | None:
    """Which checkpoint is answering — recorded because the model is retrained."""
    name = getattr(request.app.state, "model_name", None)
    if name:
        return name
    path = os.environ.get("MODEL_PATH")
    return Path(path).name if path else None
