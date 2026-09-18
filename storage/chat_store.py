#!/usr/bin/env python3
"""
storage/chat_store.py
=====================
LegalMind AI — durable chat history store.

Single point of contact with the chat database. Nothing outside this module
writes SQL; callers use the functions at the bottom of the file. That is what
makes the Postgres move cheap later: swap `_connect()` / `_DSN` and the schema
DDL, leave every call site alone. See docs/persistence.md.

Engine:     stdlib sqlite3 (no ORM, no extra dependency)
Location:   $LEGALMIND_DB_PATH, default ./data/legalmind.db
Schema:     versioned via the `schema_version` table + MIGRATIONS below

Concurrency model
-----------------
The API runs single-process (`uvicorn --workers 1`, required by the GPU model)
but multi-threaded: Starlette dispatches sync endpoints onto a threadpool, so
several requests touch the DB at once. Therefore:

  * one connection per thread (threading.local), each opened with
    check_same_thread=False and the four PRAGMAs applied;
  * WAL journalling so readers never block the writer and vice versa;
  * busy_timeout=5000 so a contended writer waits instead of raising
    "database is locked";
  * write transactions use BEGIN IMMEDIATE — acquiring the write lock up front
    is what lets busy_timeout actually do its job (a deferred transaction that
    upgrades to a write mid-way can fail instantly instead of waiting);
  * an in-process write lock serialises writers before they reach SQLite,
    which keeps contention off the file entirely for the common case.

That combination also makes the DB safe to read from a second process while
the server runs (e.g. `sqlite3 data/legalmind.db "select count(*) ..."`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("legalmind.store")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = "./data/legalmind.db"

#: Longest auto-generated conversation title, in characters.
TITLE_MAX_CHARS = 60

#: Body written for an answer that was never committed, so the transcript has
#: something honest to render where the answer should have been.
_RECONCILE_PLACEHOLDER = "_(interrupted — the server stopped while this answer was being generated)_"

_ROLES = ("user", "assistant", "system")
_STATUSES = ("complete", "partial", "error")

#: Metadata keys that `append_message` promotes into real typed columns.
#: Anything else in the metadata dict is JSON-encoded into `messages.metadata`.
_PROMOTED_KEYS = ("token_count", "latency_ms", "status", "citations", "feedback", "client_token")


def db_path() -> Path:
    """Resolved on every call so tests can repoint LEGALMIND_DB_PATH freely."""
    return Path(os.environ.get("LEGALMIND_DB_PATH", DEFAULT_DB_PATH)).expanduser()


# ---------------------------------------------------------------------------
# Row types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Message:
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str
    token_count: int | None = None
    latency_ms: int | None = None
    status: str = "complete"
    citations: list[dict] = field(default_factory=list)
    feedback: int | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "token_count": self.token_count,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "citations": self.citations,
            "feedback": self.feedback,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ConversationSummary:
    id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    pinned: bool = False
    deleted_at: str | None = None
    model_name: str | None = None
    message_count: int = 0
    preview: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "pinned": self.pinned,
            "deleted_at": self.deleted_at,
            "model_name": self.model_name,
            "message_count": self.message_count,
            "preview": self.preview,
        }


@dataclass(frozen=True)
class ActivityEntry:
    """
    One entry in the shared, public activity log: a question somebody asked.

    Deliberately NOT scoped to a user — this is the global "recent searches"
    feed every visitor sees, whichever device or browser they are on.
    `who` is a short non-identifying label derived from the browser id, so the
    feed reads as a stream of distinct people without publishing the cookie.
    """
    message_id: str
    conversation_id: str
    conversation_title: str
    query: str
    created_at: str
    who: str
    answered: bool = False
    answer_status: str | None = None

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "conversation_id": self.conversation_id,
            "conversation_title": self.conversation_title,
            "query": self.query,
            "created_at": self.created_at,
            "who": self.who,
            "answered": self.answered,
            "answer_status": self.answer_status,
        }


@dataclass(frozen=True)
class SearchHit:
    conversation_id: str
    conversation_title: str
    message_id: str
    role: str
    created_at: str
    snippet: str

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "conversation_title": self.conversation_title,
            "message_id": self.message_id,
            "role": self.role,
            "created_at": self.created_at,
            "snippet": self.snippet,
        }


# ---------------------------------------------------------------------------
# Connection handling
# ---------------------------------------------------------------------------

_local = threading.local()
_write_lock = threading.RLock()


def _connect(path: Path) -> sqlite3.Connection:
    """
    The one function that knows how to reach the database.

    Moving to Postgres replaces this body (and the DDL in MIGRATIONS); no
    caller changes. isolation_level=None puts sqlite3 in autocommit mode so
    that `transaction()` below controls transactions explicitly rather than
    the driver guessing where they start.
    """
    conn = sqlite3.connect(
        str(path),
        check_same_thread=False,   # connections are handed to threadpool workers
        isolation_level=None,      # explicit BEGIN IMMEDIATE in transaction()
        timeout=5.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def get_connection() -> sqlite3.Connection:
    """
    Per-thread singleton connection. Opened once and reused for the life of the
    thread — never reopened per request.
    """
    path = db_path()
    cached = getattr(_local, "conn", None)
    if cached is not None and getattr(_local, "path", None) == str(path):
        return cached
    if cached is not None:          # LEGALMIND_DB_PATH changed under us (tests)
        try:
            cached.close()
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(path)
    _local.conn = conn
    _local.path = str(path)
    _local.depth = 0
    return conn


def close_connection() -> None:
    """Close this thread's connection. Tests use it between temp databases."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
    _local.conn = None
    _local.path = None
    _local.depth = 0


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """
    Write transaction. Re-entrant: nested uses join the outermost transaction,
    so a composite operation commits once, all-or-nothing.
    """
    conn = get_connection()
    depth = getattr(_local, "depth", 0)
    if depth:                                   # already inside one
        _local.depth = depth + 1
        try:
            yield conn
        finally:
            _local.depth -= 1
        return

    _write_lock.acquire()
    try:
        conn.execute("BEGIN IMMEDIATE")
        _local.depth = 1
        try:
            yield conn
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")
        finally:
            _local.depth = 0
    finally:
        _write_lock.release()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utcnow() -> str:
    """ISO8601 UTC, second-and-microsecond precision, sorts lexicographically."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def new_id() -> str:
    return str(uuid.uuid4())


def make_title(text: str, max_chars: int = TITLE_MAX_CHARS) -> str:
    """Cleaned truncation of the first user message. Never returns empty."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    cleaned = re.sub(r"^[\s\-*>#`]+", "", cleaned)
    if not cleaned:
        return "Untitled conversation"
    if len(cleaned) <= max_chars:
        return cleaned
    cut = cleaned[:max_chars]
    if " " in cut[int(max_chars * 0.6):]:        # avoid chopping mid-word
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;:—-") + "…"


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (slug[:max_len].rstrip("-") or "conversation")


def _json_loads(raw: Any, fallback):
    if raw in (None, ""):
        return fallback
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return fallback


def _row_to_message(row: sqlite3.Row) -> Message:
    return Message(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
        token_count=row["token_count"],
        latency_ms=row["latency_ms"],
        status=row["status"] or "complete",
        citations=_json_loads(row["citations"], []),
        feedback=row["feedback"],
        metadata=_json_loads(row["metadata"], {}),
    )


def _row_to_conversation(row: sqlite3.Row) -> ConversationSummary:
    keys = row.keys()
    return ConversationSummary(
        id=row["id"],
        user_id=row["user_id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        pinned=bool(row["pinned"]),
        deleted_at=row["deleted_at"],
        model_name=row["model_name"],
        message_count=row["message_count"] if "message_count" in keys else 0,
        preview=row["preview"] if "preview" in keys else None,
    )


# ---------------------------------------------------------------------------
# Schema + migrations
# ---------------------------------------------------------------------------
# Numbered, forward-only, applied in order inside one transaction each. To
# change the schema later: append a new (version, [statements]) entry. Never
# edit a shipped migration — existing databases have already run it, and the
# whole point is that nobody has to wipe their history.

_MIGRATION_1 = [
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id          TEXT PRIMARY KEY,              -- uuid4
        user_id     TEXT NOT NULL,
        title       TEXT NOT NULL,
        created_at  TEXT NOT NULL,                 -- ISO8601 UTC
        updated_at  TEXT NOT NULL,                 -- ISO8601 UTC
        pinned      INTEGER NOT NULL DEFAULT 0,
        deleted_at  TEXT,                          -- soft delete; NULL when live
        model_name  TEXT                           -- checkpoint that answered
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id              TEXT PRIMARY KEY,          -- uuid4
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role            TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
        content         TEXT NOT NULL,
        created_at      TEXT NOT NULL,
        token_count     INTEGER,
        latency_ms      INTEGER,
        status          TEXT NOT NULL DEFAULT 'complete'
                        CHECK (status IN ('complete','partial','error')),
        citations       TEXT,                      -- JSON array of authorities
        feedback        INTEGER,                   -- -1 / 0 / +1, for eval work
        client_token    TEXT,                      -- submit token; makes writes idempotent
        metadata        TEXT                       -- JSON; anything not promoted above
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_messages_conv_created ON messages(conversation_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC)",
    # Idempotency guard: a retried or double-clicked submit carries the same
    # token and cannot become a second row.
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_client_token ON messages(client_token) WHERE client_token IS NOT NULL",
    # Full-text search over message bodies. External-content table: the index
    # stores terms only and reads text back from `messages`, kept in step by
    # the three triggers below.
    "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content, content='messages', content_rowid='rowid')",
    """
    CREATE TRIGGER IF NOT EXISTS messages_fts_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_fts_ad AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_fts_au AFTER UPDATE OF content ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
        INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
    END
    """,
]

MIGRATIONS: list[tuple[int, list[str]]] = [
    (1, _MIGRATION_1),
]

_init_lock = threading.Lock()
_initialised: set[str] = set()


def init_db(force: bool = False) -> int:
    """
    Create or upgrade the schema. Idempotent — safe to call on every startup
    and from every test. Returns the schema version now in force.
    """
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with _init_lock:
        if not force and str(path) in _initialised:
            return _current_version()

        conn = get_connection()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_version ("
            " version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
        )
        current = _current_version()
        for version, statements in MIGRATIONS:
            if version <= current:
                continue
            log.info(f"Applying chat-store migration {version}…")
            with transaction() as c:
                for stmt in statements:
                    c.execute(stmt)
                c.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (version, utcnow()),
                )
            current = version

        _initialised.add(str(path))
        log.info(f"Chat store ready at {path} (schema v{current})")
        return current


def _current_version() -> int:
    conn = get_connection()
    try:
        row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row["v"]) if row and row["v"] is not None else 0


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def create_conversation(
    user_id: str,
    title: str = "New conversation",
    model_name: str | None = None,
    conversation_id: str | None = None,
) -> str:
    """Create a conversation and return its id (uuid4 unless one is supplied)."""
    if not user_id:
        raise ValueError("user_id is required")
    cid = conversation_id or new_id()
    now = utcnow()
    with transaction() as conn:
        conn.execute(
            "INSERT INTO conversations"
            " (id, user_id, title, created_at, updated_at, pinned, deleted_at, model_name)"
            " VALUES (?, ?, ?, ?, ?, 0, NULL, ?)",
            (cid, user_id, make_title(title) if title else "New conversation", now, now, model_name),
        )
    return cid


def get_conversation(conversation_id: str, user_id: str | None = None,
                     include_deleted: bool = True) -> ConversationSummary | None:
    """
    Fetch one conversation. Pass `user_id` to scope the lookup to its owner —
    every HTTP handler does, so one user can never address another's chat.
    """
    sql = [
        "SELECT c.*,",
        " (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count,",
        " (SELECT m.content FROM messages m WHERE m.conversation_id = c.id"
        "  ORDER BY m.created_at DESC, m.rowid DESC LIMIT 1) AS preview",
        "FROM conversations c WHERE c.id = ?",
    ]
    params: list[Any] = [conversation_id]
    if user_id is not None:
        sql.append("AND c.user_id = ?")
        params.append(user_id)
    if not include_deleted:
        sql.append("AND c.deleted_at IS NULL")
    row = get_connection().execute(" ".join(sql), params).fetchone()
    return _row_to_conversation(row) if row else None


def list_conversations(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    include_deleted: bool = False,
) -> list[ConversationSummary]:
    """
    That user's conversations, pinned first then newest-updated first — the
    order the sidebar renders in.
    """
    sql = (
        "SELECT c.*,"
        " (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count,"
        " (SELECT m.content FROM messages m WHERE m.conversation_id = c.id"
        "  ORDER BY m.created_at DESC, m.rowid DESC LIMIT 1) AS preview"
        " FROM conversations c"
        " WHERE c.user_id = ?"
        + ("" if include_deleted else " AND c.deleted_at IS NULL") +
        " ORDER BY c.pinned DESC, c.updated_at DESC"
        " LIMIT ? OFFSET ?"
    )
    rows = get_connection().execute(sql, (user_id, int(limit), int(offset))).fetchall()
    return [_row_to_conversation(r) for r in rows]


def rename_conversation(conversation_id: str, title: str, user_id: str | None = None) -> bool:
    """Rename. Returns False if the conversation does not exist (or isn't theirs)."""
    clean = re.sub(r"\s+", " ", (title or "").strip())[:200]
    if not clean:
        raise ValueError("title cannot be empty")
    return _update_conversation(conversation_id, user_id, "title = ?", [clean])


def set_pinned(conversation_id: str, pinned: bool, user_id: str | None = None) -> bool:
    return _update_conversation(conversation_id, user_id, "pinned = ?", [1 if pinned else 0])


def soft_delete_conversation(conversation_id: str, user_id: str | None = None) -> bool:
    """Mark deleted. Rows stay put, so undo is just clearing the stamp."""
    return _update_conversation(conversation_id, user_id, "deleted_at = ?", [utcnow()])


def restore_conversation(conversation_id: str, user_id: str | None = None) -> bool:
    """Undo a soft delete."""
    return _update_conversation(conversation_id, user_id, "deleted_at = NULL", [])


def set_conversation_model(conversation_id: str, model_name: str | None,
                           user_id: str | None = None) -> bool:
    return _update_conversation(conversation_id, user_id, "model_name = ?", [model_name])


def _update_conversation(conversation_id: str, user_id: str | None,
                         assignment: str, params: list[Any]) -> bool:
    sql = f"UPDATE conversations SET {assignment} WHERE id = ?"
    args = list(params) + [conversation_id]
    if user_id is not None:
        sql += " AND user_id = ?"
        args.append(user_id)
    with transaction() as conn:
        cur = conn.execute(sql, args)
        return cur.rowcount > 0


def purge_conversation(conversation_id: str) -> bool:
    """
    Hard delete, for retention jobs and tests. Messages go with it via
    ON DELETE CASCADE (foreign_keys=ON is set on every connection).
    """
    with transaction() as conn:
        return conn.execute("DELETE FROM conversations WHERE id = ?",
                            (conversation_id,)).rowcount > 0


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def append_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> str:
    """
    Append one message and bump the conversation's updated_at **in the same
    transaction**, so a reader never sees a message whose conversation looks
    stale.

    `metadata` carries both typed columns and free-form extras:

        token_count, latency_ms, status, citations, feedback, client_token
            -> promoted into their own columns
        anything else
            -> JSON-encoded into messages.metadata (this is where a failure's
               error text lands)

    Idempotency: if `metadata['client_token']` has already been written, the
    existing message id comes back and nothing is inserted. That is the guard
    against a double submit or a retried request creating a duplicate row —
    it keys on the caller's token, never on request ordering.
    """
    if role not in _ROLES:
        raise ValueError(f"role must be one of {_ROLES}, got {role!r}")
    meta = dict(metadata or {})

    status = meta.pop("status", "complete") or "complete"
    if status not in _STATUSES:
        raise ValueError(f"status must be one of {_STATUSES}, got {status!r}")

    token_count = meta.pop("token_count", None)
    latency_ms = meta.pop("latency_ms", None)
    feedback = meta.pop("feedback", None)
    client_token = meta.pop("client_token", None)
    citations = meta.pop("citations", None)
    citations_json = json.dumps(citations, ensure_ascii=False) if citations else None
    meta_json = json.dumps(meta, ensure_ascii=False, default=str) if meta else None

    now = utcnow()
    mid = new_id()

    with transaction() as conn:
        if client_token:
            existing = conn.execute(
                "SELECT id FROM messages WHERE client_token = ?", (client_token,)
            ).fetchone()
            if existing:
                return existing["id"]

        conn.execute(
            "INSERT INTO messages"
            " (id, conversation_id, role, content, created_at, token_count, latency_ms,"
            "  status, citations, feedback, client_token, metadata)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (mid, conversation_id, role, content, now,
             _as_int(token_count), _as_int(latency_ms), status,
             citations_json, _as_int(feedback), client_token, meta_json),
        )
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    return mid


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_messages(conversation_id: str, limit: int | None = None) -> list[Message]:
    """Full message list for a conversation, oldest first."""
    sql = ("SELECT * FROM messages WHERE conversation_id = ?"
           " ORDER BY created_at ASC, rowid ASC")
    params: list[Any] = [conversation_id]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    rows = get_connection().execute(sql, params).fetchall()
    return [_row_to_message(r) for r in rows]


def get_message(message_id: str) -> Message | None:
    row = get_connection().execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
    return _row_to_message(row) if row else None


def get_message_by_client_token(client_token: str) -> Message | None:
    """
    Look up a message by its submit token. The chat endpoint calls this first
    so a retried or double-submitted request returns the exchange that already
    exists instead of inserting a second one — and, more expensively, instead
    of running the model twice.
    """
    if not client_token:
        return None
    row = get_connection().execute(
        "SELECT * FROM messages WHERE client_token = ?", (client_token,)
    ).fetchone()
    return _row_to_message(row) if row else None


def get_next_message(conversation_id: str, after: Message) -> Message | None:
    """The message immediately following `after` in its conversation, if any."""
    row = get_connection().execute(
        "SELECT * FROM messages WHERE conversation_id = ?"
        "   AND (created_at > ? OR (created_at = ? AND rowid > ("
        "        SELECT rowid FROM messages WHERE id = ?)))"
        " ORDER BY created_at ASC, rowid ASC LIMIT 1",
        (conversation_id, after.created_at, after.created_at, after.id),
    ).fetchone()
    return _row_to_message(row) if row else None


def update_message(
    message_id: str,
    content: str | None = None,
    status: str | None = None,
    metadata: dict | None = None,
    citations: list[dict] | None = None,
    token_count: int | None = None,
    latency_ms: int | None = None,
) -> bool:
    """
    Patch an existing message. Used to finalise a row that was written before
    the answer was known (the streaming seam — see docs/persistence.md).
    """
    sets: list[str] = []
    args: list[Any] = []
    if content is not None:
        sets.append("content = ?")
        args.append(content)
    if status is not None:
        if status not in _STATUSES:
            raise ValueError(f"status must be one of {_STATUSES}")
        sets.append("status = ?")
        args.append(status)
    if citations is not None:
        sets.append("citations = ?")
        args.append(json.dumps(citations, ensure_ascii=False))
    if token_count is not None:
        sets.append("token_count = ?")
        args.append(_as_int(token_count))
    if latency_ms is not None:
        sets.append("latency_ms = ?")
        args.append(_as_int(latency_ms))
    if metadata is not None:
        sets.append("metadata = ?")
        args.append(json.dumps(metadata, ensure_ascii=False, default=str))
    if not sets:
        return False
    args.append(message_id)
    with transaction() as conn:
        cur = conn.execute(f"UPDATE messages SET {', '.join(sets)} WHERE id = ?", args)
        if cur.rowcount:
            conn.execute(
                "UPDATE conversations SET updated_at = ?"
                " WHERE id = (SELECT conversation_id FROM messages WHERE id = ?)",
                (utcnow(), message_id),
            )
        return cur.rowcount > 0


def set_feedback(message_id: str, feedback: int, user_id: str | None = None) -> bool:
    """Record -1 / 0 / +1 against an assistant message, for later eval work."""
    if feedback not in (-1, 0, 1):
        raise ValueError("feedback must be -1, 0 or 1")
    sql = "UPDATE messages SET feedback = ? WHERE id = ?"
    args: list[Any] = [feedback, message_id]
    if user_id is not None:
        sql += (" AND conversation_id IN (SELECT id FROM conversations WHERE user_id = ?)")
        args.append(user_id)
    with transaction() as conn:
        return conn.execute(sql, args).rowcount > 0


def count_messages(user_id: str | None = None) -> int:
    """Total message rows, optionally for one user. Mirrors the UI's count."""
    if user_id is None:
        row = get_connection().execute("SELECT COUNT(*) AS n FROM messages").fetchone()
    else:
        row = get_connection().execute(
            "SELECT COUNT(*) AS n FROM messages m"
            " JOIN conversations c ON c.id = m.conversation_id WHERE c.user_id = ?",
            (user_id,),
        ).fetchone()
    return int(row["n"])


# ---------------------------------------------------------------------------
# Search (FTS5)
# ---------------------------------------------------------------------------

def _fts_query(raw: str) -> str:
    """
    Turn arbitrary user input into a safe FTS5 MATCH expression.

    Raw input cannot go straight to MATCH: bare `-`, `*`, `"` or the words
    AND/OR/NOT are operators there and a stray one is a syntax error, not a
    no-result. So each word becomes a quoted phrase, ANDed implicitly, with a
    prefix wildcard on the last one so partial typing still matches.
    """
    tokens = re.findall(r"[0-9A-Za-z_]+", raw or "")
    if not tokens:
        return ""
    quoted = [f'"{t}"' for t in tokens]
    quoted[-1] += "*"
    return " ".join(quoted)


def search_messages(user_id: str, query: str, limit: int = 50,
                    include_deleted: bool = False) -> list[SearchHit]:
    """
    Full-text search across that user's own messages. Soft-deleted
    conversations are excluded unless asked for.
    """
    match = _fts_query(query)
    if not match:
        return []
    sql = (
        "SELECT m.id AS message_id, m.role, m.created_at,"
        "       c.id AS conversation_id, c.title AS conversation_title,"
        "       snippet(messages_fts, 0, '<mark>', '</mark>', '…', 14) AS snippet"
        " FROM messages_fts"
        " JOIN messages m ON m.rowid = messages_fts.rowid"
        " JOIN conversations c ON c.id = m.conversation_id"
        " WHERE messages_fts MATCH ? AND c.user_id = ?"
        + ("" if include_deleted else " AND c.deleted_at IS NULL") +
        " ORDER BY rank LIMIT ?"
    )
    try:
        rows = get_connection().execute(sql, (match, user_id, int(limit))).fetchall()
    except sqlite3.OperationalError as e:      # malformed MATCH must not 500
        log.warning(f"history search failed for {query!r}: {e}")
        return []
    return [
        SearchHit(
            conversation_id=r["conversation_id"],
            conversation_title=r["conversation_title"],
            message_id=r["message_id"],
            role=r["role"],
            created_at=r["created_at"],
            snippet=r["snippet"] or "",
        )
        for r in rows
    ]


def _who(user_id: str | None) -> str:
    """Short, stable, non-identifying label for the public feed."""
    if not user_id:
        return "anon"
    tail = re.sub(r"[^0-9a-zA-Z]", "", user_id)[-4:].lower()
    return f"user-{tail}" if tail else "anon"


def list_activity(limit: int = 50, offset: int = 0, query: str | None = None,
                  include_deleted: bool = False) -> list[ActivityEntry]:
    """
    The shared activity log: every question asked by anyone, newest first.

    This is the deliberately UNSCOPED counterpart to list_conversations() —
    no user_id filter, so two people on two laptops see one combined feed.
    Paginated because it grows without bound; the UI pages through it rather
    than rendering the whole table.

    `query` filters via the same FTS5 index used by search_messages().
    """
    params: list[Any] = []
    where = ["m.role = 'user'"]
    join = ""

    if not include_deleted:
        where.append("c.deleted_at IS NULL")

    if query:
        match = _fts_query(query)
        if not match:
            return []
        join = " JOIN messages_fts ON messages_fts.rowid = m.rowid"
        where.append("messages_fts MATCH ?")
        params.append(match)

    sql = (
        "SELECT m.id AS message_id, m.content AS query, m.created_at,"
        "       c.id AS conversation_id, c.title AS conversation_title, c.user_id,"
        "       (SELECT a.status FROM messages a"
        "         WHERE a.conversation_id = c.id AND a.role = 'assistant'"
        "           AND (a.created_at > m.created_at"
        "                OR (a.created_at = m.created_at AND a.rowid > m.rowid))"
        "         ORDER BY a.created_at ASC, a.rowid ASC LIMIT 1) AS answer_status"
        " FROM messages m"
        " JOIN conversations c ON c.id = m.conversation_id"
        + join +
        " WHERE " + " AND ".join(where) +
        " ORDER BY m.created_at DESC, m.rowid DESC"
        " LIMIT ? OFFSET ?"
    )
    params.extend([int(limit), int(offset)])

    try:
        rows = get_connection().execute(sql, params).fetchall()
    except sqlite3.OperationalError as e:          # a malformed MATCH must not 500
        log.warning(f"activity feed query failed for {query!r}: {e}")
        return []

    return [
        ActivityEntry(
            message_id=r["message_id"],
            conversation_id=r["conversation_id"],
            conversation_title=r["conversation_title"],
            query=r["query"],
            created_at=r["created_at"],
            who=_who(r["user_id"]),
            answered=r["answer_status"] is not None,
            answer_status=r["answer_status"],
        )
        for r in rows
    ]


def count_activity(query: str | None = None, include_deleted: bool = False) -> int:
    """Total rows in the shared feed, so the UI can page through it."""
    params: list[Any] = []
    where = ["m.role = 'user'"]
    join = ""
    if not include_deleted:
        where.append("c.deleted_at IS NULL")
    if query:
        match = _fts_query(query)
        if not match:
            return 0
        join = " JOIN messages_fts ON messages_fts.rowid = m.rowid"
        where.append("messages_fts MATCH ?")
        params.append(match)
    sql = ("SELECT COUNT(*) AS n FROM messages m"
           " JOIN conversations c ON c.id = m.conversation_id" + join +
           " WHERE " + " AND ".join(where))
    try:
        return int(get_connection().execute(sql, params).fetchone()["n"])
    except sqlite3.OperationalError:
        return 0


def get_public_conversation(conversation_id: str) -> ConversationSummary | None:
    """
    Read a conversation without an ownership check, for opening an entry from
    the shared feed. Read-only by design: every mutating function still takes
    a user_id, so a public reader can look but cannot rename, pin or delete.
    """
    return get_conversation(conversation_id, user_id=None, include_deleted=False)


def rebuild_search_index() -> None:
    """Rebuild the FTS index from `messages`. For recovery, not normal use."""
    with transaction() as conn:
        conn.execute("INSERT INTO messages_fts(messages_fts) VALUES ('rebuild')")


# ---------------------------------------------------------------------------
# Startup reconciliation
# ---------------------------------------------------------------------------

def reconcile_interrupted(placeholder: str = _RECONCILE_PLACEHOLDER) -> int:
    """
    Find exchanges the previous process died in the middle of and close them
    off honestly, so the UI has a row to render instead of a question whose
    answer vanished.

    Generation here is one blocking call (`run_rag`), so there is no partial
    text on disk to recover — what we can say for certain is that a question
    was asked and no answer was ever committed. Each such trailing user
    message gets an assistant row with status='partial'. Called from startup;
    returns how many it closed.
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT m.id, m.conversation_id, m.created_at FROM messages m"
        " WHERE m.role = 'user'"
        "   AND NOT EXISTS ("
        "     SELECT 1 FROM messages later"
        "      WHERE later.conversation_id = m.conversation_id"
        "        AND later.role IN ('assistant','system')"
        "        AND (later.created_at > m.created_at"
        "             OR (later.created_at = m.created_at AND later.rowid > m.rowid))"
        "   )"
    ).fetchall()
    if not rows:
        return 0
    for row in rows:
        append_message(
            row["conversation_id"],
            "assistant",
            placeholder,
            {
                "status": "partial",
                "error": "server stopped before the answer was committed",
                "interrupted_user_message_id": row["id"],
                "reconciled_at": utcnow(),
            },
        )
    log.warning(f"Reconciled {len(rows)} interrupted exchange(s) as status='partial'.")
    return len(rows)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def stats() -> dict:
    """Cheap counters for /stats and for eyeballing the store."""
    conn = get_connection()
    row = conn.execute(
        "SELECT"
        " (SELECT COUNT(*) FROM conversations WHERE deleted_at IS NULL) AS conversations,"
        " (SELECT COUNT(*) FROM conversations WHERE deleted_at IS NOT NULL) AS deleted_conversations,"
        " (SELECT COUNT(*) FROM messages) AS messages,"
        " (SELECT COUNT(*) FROM messages WHERE status != 'complete') AS incomplete_messages,"
        " (SELECT COUNT(DISTINCT user_id) FROM conversations) AS users"
    ).fetchone()
    out = {k: int(row[k]) for k in row.keys()}
    out["db_path"] = str(db_path())
    out["schema_version"] = _current_version()
    return out
