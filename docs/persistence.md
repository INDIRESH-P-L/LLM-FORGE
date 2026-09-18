# Chat persistence

How LegalMind AI stores conversations, where the data lives, how to back it up,
and how to move to Postgres when real accounts arrive.

- Store module: `storage/chat_store.py` (the only place SQL is written)
- Export rendering: `storage/exporters.py`
- HTTP surface: `app/chat_api.py`
- Tests: `tests/test_chat_store.py`

---

## What changed, and why

Before this, chat history did not exist anywhere. Messages were appended
straight into the DOM by `app/static/app.js` — no JavaScript array, no
`localStorage`, no server-side session — and `POST /query` was stateless. A
refresh, a closed tab or a navigation destroyed the conversation, because the
DOM *was* the conversation.

Now every message is a row in SQLite before the user can navigate away from it,
and the browser renders history *from* the database rather than remembering it.

---

## Where the database lives

| | |
|---|---|
| Path | `$LEGALMIND_DB_PATH`, default `./data/legalmind.db` |
| Created by | `chat_store.init_db()` on API startup (parent dirs included) |
| Journal | WAL — expect `legalmind.db-wal` and `legalmind.db-shm` beside it |
| Git | all three are in `.gitignore`; this is user data and must never be committed |

Point it somewhere else with an env var:

```bash
export LEGALMIND_DB_PATH=/var/lib/legalmind/legalmind.db
uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
```

`db_path()` is read on every call, so tests and one-off scripts can repoint it
freely.

### SQLite configuration

Applied to **every** connection, in `chat_store._connect()`:

```sql
PRAGMA journal_mode=WAL;      -- readers never block the writer, or vice versa
PRAGMA busy_timeout=5000;     -- a contended writer waits instead of failing
PRAGMA foreign_keys=ON;       -- ON DELETE CASCADE actually cascades
PRAGMA synchronous=NORMAL;    -- the right durability/throughput point under WAL
```

Connections are opened with `check_same_thread=False` and `isolation_level=None`.
`journal_mode` persists in the file; the other three are per-connection, which
is why they are set on open rather than once at install time.

---

## Schema

`schema_version` records which migrations have run. Migrations are numbered,
forward-only, and applied in one transaction each by `init_db()`, which is
idempotent — safe on every startup.

### `conversations`

| column | type | notes |
|---|---|---|
| `id` | TEXT PK | uuid4, not autoincrement — safe to merge across databases |
| `user_id` | TEXT NOT NULL | opaque string; see *Ownership* below |
| `title` | TEXT NOT NULL | cleaned 60-char truncation of the first question |
| `created_at` | TEXT NOT NULL | ISO8601 UTC, `Z`-suffixed, sorts lexicographically |
| `updated_at` | TEXT NOT NULL | bumped on every append, in the same transaction |
| `pinned` | INTEGER | 0/1; pinned sorts above everything in the sidebar |
| `deleted_at` | TEXT | soft delete; NULL when live. Undo just clears it |
| `model_name` | TEXT | which checkpoint answered — matters as the model is retrained |

### `messages`

| column | type | notes |
|---|---|---|
| `id` | TEXT PK | uuid4 |
| `conversation_id` | TEXT NOT NULL | `REFERENCES conversations(id) ON DELETE CASCADE` |
| `role` | TEXT NOT NULL | `CHECK (role IN ('user','assistant','system'))` |
| `content` | TEXT NOT NULL | |
| `created_at` | TEXT NOT NULL | ISO8601 UTC |
| `token_count` | INTEGER | currently NULL — see *Known gaps* |
| `latency_ms` | INTEGER | wall-clock for the whole exchange |
| `status` | TEXT | `CHECK (status IN ('complete','partial','error'))`, default `complete` |
| `citations` | TEXT | JSON array of authorities, or NULL |
| `feedback` | INTEGER | −1 / 0 / +1, for later eval work |
| `client_token` | TEXT | submit token; `UNIQUE` where non-NULL. The duplicate-row guard |
| `metadata` | TEXT | JSON for everything not promoted to a column (confidence, error text, …) |

`client_token` and `metadata` are additions to the originally specified shape.
`metadata` is what `append_message(..., metadata)` writes once the typed keys
(`token_count`, `latency_ms`, `status`, `citations`, `feedback`, `client_token`)
have been promoted into their own columns; `client_token` is what makes writes
idempotent per interaction.

### Indexes

```sql
messages(conversation_id, created_at)              -- transcript reads
conversations(user_id, updated_at DESC)            -- sidebar reads
messages(client_token) WHERE client_token NOT NULL -- UNIQUE; the dedup guard
messages_fts                                        -- FTS5 over messages.content
```

`messages_fts` is an **external-content** FTS5 table: it indexes terms only and
reads the text back from `messages`, kept in step by three triggers
(`messages_fts_ai` / `_ad` / `_au`). If it ever drifts,
`chat_store.rebuild_search_index()` repairs it without data loss.

### `citations` JSON

Written alongside each assistant message that relied on retrieval:

```json
[{"case_name": "...", "citation": "...", "court": "...", "date": "...",
  "judges": null, "paragraph_no": null, "source_id": "...", "excerpt": "...",
  "citation_id": 1, "document_type": "case_law"}]
```

Mapped in `app/chat_api.py:citations_from_result()` from the
`retrieved_documents` the RAG pipeline already returns —
`case_name ← title`, `source_id ← document_id`, `excerpt ← text_preview`.

**Two shapes reach this column.** `POST /api/chat` writes the structured form
above. Any writer that passes `run_rag()["citations"]` straight through instead
stores a list of pre-formatted **strings** (`"[1] Bharatiya Nagarik Suraksha
Sanhita"`), because that is what `format_citations()` returns — a different
thing from `retrieved_documents`. `storage/exporters.py` normalises both, so
neither crashes an export, but the string form loses court, date, source id and
excerpt, and an export can then only list bare names.

If you are persisting from your own endpoint, pass the structured form:

```python
from app.chat_api import citations_from_result
chat_store.append_message(conv_id, "assistant", result["answer"],
                          {"citations": citations_from_result(result),  # not result["citations"]
                           "status": "complete"})
```

`judges` and `paragraph_no` are written as `null` on purpose: the ingested
corpus carries neither (chunk keys are `case_number`, `chunk_id`,
`chunk_index`, `citation`, `court`, `date`, `document_id`, `document_type`,
`source`, `title`), and `chunk_index` is a chunking artefact, not a judgment
paragraph number — recording it as one would be a fabricated citation. The
columns exist so that when ingestion starts extracting them, only the mapping
function changes.

---

## The shared activity log (global, all users)

Alongside the per-browser history there is one **global feed**: every question
asked by anyone, visible to everyone, on any device. No login involved.

### Why nothing "needed replacing"

The storage was already shared and persistent — one SQLite file on the server,
written by every request. Chat history was never in `st.session_state` (there
is no Streamlit here), never in `localStorage`, never in an in-memory list.
What made one laptop's searches invisible on another was purely the **read
path**: `list_conversations()` and `search_messages()` filter on `user_id`,
which comes from the per-browser `legalmind_uid` cookie.

So the shared feed is a second, deliberately **unscoped** read over the same
rows — not a second store. There is exactly one source of truth.

| | private view | shared feed |
|---|---|---|
| store function | `list_conversations(user_id, …)` | `list_activity(limit, offset, query)` |
| endpoint | `GET /api/conversations` | `GET /api/activity` |
| filter | `WHERE c.user_id = ?` | none |
| UI | sidebar → **My chats** | sidebar → **Everyone** (default) |

### Shape

`GET /api/activity?limit=&offset=&q=` returns newest-first entries of
`{message_id, conversation_id, conversation_title, query, created_at, who,
answered, answer_status}` plus `total` and `has_more`.

- **`who`** is a short non-identifying label (`user-3f50`) derived from the
  browser id — the feed reads as distinct people without publishing cookies.
- **`q`** filters through the same FTS5 index as private search, with the same
  token-quoting so operator characters cannot become a syntax error.
- **Pagination is mandatory**, not cosmetic: the feed grows without bound, so
  the UI pages through it (`limit`/`offset`, default 40 per page) rather than
  rendering the whole table.
- Soft-deleted conversations drop out of the feed and stop being publicly
  readable.

### Read-only by construction

`GET /api/activity/{conversation_id}` returns any conversation's transcript
with no ownership check — that is what makes the log public. Every **mutating**
function still takes `user_id` and every mutating endpoint still calls
`_own_conversation()`, so a reader who opens somebody else's conversation can
look but cannot rename, pin, delete or export it. The UI reflects this with a
"shared — read only" badge and hides those buttons.

### Freshness

Another person's search happens in a different process, so there is nothing to
push into an open tab. The UI re-reads `/api/activity` on every page load, when
the tab regains focus, immediately after you send your own question, and on a
20-second poll while the **Everyone** tab is visible.

### Concurrency

No new machinery: WAL journalling, `busy_timeout=5000` and `BEGIN IMMEDIATE`
already cover several people searching at once, and readers never block the
writer. `tests/test_chat_store.py` exercises the feed under six concurrent
writer threads and under continuous reads during writes.

### Privacy note

This is a public log by design — every visitor sees every question, including
questions typed by people who did not realise they were public. For a legal
research tool that is a real consideration. The narrowing lever, if it is ever
wanted, is one function: `get_or_create_user_id()` in `app/chat_api.py`.

---

## Ownership (no accounts yet)

`get_or_create_user_id()` in `app/chat_api.py` is the **single** auth seam:

1. `?uid=` in the URL, if present and well-formed — so a conversation link is
   bookmarkable and portable between browsers;
2. otherwise the `legalmind_uid` cookie (HttpOnly, SameSite=Lax, 10 years);
3. otherwise a fresh uuid4, written back as a cookie.

A middleware runs it on every request — including the HTML page load, so the
identity exists before any JavaScript does — and puts the result on
`request.state.user_id`.

**When real accounts arrive, that function is the only thing that changes.**
Return the authenticated principal's id and everything else keeps working:
`user_id` is already an opaque string on `conversations`, every handler is
already scoped by it, and every conversation-scoped query already filters on
it, so one browser cannot read, rename, delete, export or rate another's chat
(a foreign id returns 404, indistinguishable from "does not exist").

---

## Write ordering

The guarantee is *the question is never lost*, so writes happen when each
thing becomes known rather than at the end of the exchange:

1. **On submit** — the user message is committed **before** the model is
   called. An OOM, a crash or a killed process during generation cannot lose
   the question.
2. **On completion** — the assistant row is written with `status='complete'`,
   its citations, latencies and confidence, in one transaction with the
   `updated_at` bump.
3. **On failure** — a row is still written, `status='error'`, with the error
   text and type in `metadata`. The UI renders it as a failed answer.
4. **On a dropped client** — `/api/chat` is a sync `def`, so Starlette runs it
   on the threadpool and it runs to completion even after the browser goes
   away. A killed tab still gets its answer committed; it appears on next load.
5. **On process death mid-generation** — `reconcile_interrupted()` runs at
   startup, finds any question with no answer after it, and closes it off with
   an assistant row at `status='partial'`. Idempotent across restarts.

### On transactions and "no partially-written exchanges"

Each append is atomic — message row plus `updated_at`, one transaction, and
`transaction()` is re-entrant so a composite operation commits once or not at
all. The two halves of an exchange are deliberately **not** one transaction:
holding a write transaction open across a multi-second GPU generation would
block every other writer on the box for the duration. Rule 1 (the question is
durable before inference) wins over a single enclosing transaction, and the
in-between state is not corruption — it is `status='partial'`, a state the
schema names, reconciliation closes and the UI renders.

### Idempotency

The browser generates a `client_token` (uuid4) per submit. `/api/chat` looks it
up first: a token already on disk returns the stored exchange, writing nothing
and **not** re-running the model. A `UNIQUE` index enforces it at the SQL level
even if a caller bypasses that check. The guard is the token, never position in
a script or request ordering.

---

## Reads

`GET /api/conversations/{id}` returns the conversation and its messages **from
the database**. The frontend keeps no history of its own; `state` in
`app/static/app.js` is a cache for the current paint. The active conversation
and identity live in the URL (`?uid=…&c=…`), so a refresh lands in the same
chat and a link can be reopened later.

---

## HTTP surface

| method | path | purpose |
|---|---|---|
| GET | `/api/me` | resolved `user_id` |
| GET | `/api/conversations` | sidebar list (pinned first, newest-updated first) |
| POST | `/api/conversations` | new empty conversation |
| GET | `/api/conversations/{id}` | conversation + full message list |
| PATCH | `/api/conversations/{id}` | rename and/or pin |
| DELETE | `/api/conversations/{id}` | soft delete |
| POST | `/api/conversations/{id}/restore` | undo the soft delete |
| GET | `/api/conversations/{id}/export?format=md\|txt` | download |
| GET | `/api/search?q=` | FTS5 search over your own messages |
| POST | `/api/messages/{id}/feedback` | −1 / 0 / +1 |
| GET | `/api/stats` | store counters |
| POST | `/api/chat` | **the persistent chat loop** |

`POST /query` is unchanged and still stateless — `fine_tuning/evaluate.py`
drives it, so it was deliberately left alone.

### Export

`<date>-<slugified-title>.md` (or `.txt`), with every authority cited in the
conversation de-duplicated and numbered at the bottom under **Authorities** —
law students paste that straight into case notes. `partial` and `error` answers
are labelled as such in the export rather than passed off as complete.

---

## Configuration

| variable | default | meaning |
|---|---|---|
| `LEGALMIND_DB_PATH` | `./data/legalmind.db` | database file |
| `LEGALMIND_AUTO_TITLE` | `0` | prefix the leading statutory provision onto the title |

`LEGALMIND_AUTO_TITLE` is off by default. See *Known gaps* for what it does and
does not do.

---

## Backup and restore

Use SQLite's online backup — **never** `cp` a live WAL database, which can
capture a torn page plus a stale `-wal`:

```bash
# Consistent snapshot, safe while the server is running and serving
sqlite3 data/legalmind.db ".backup '/backups/legalmind-$(date +%F-%H%M).db'"

# Verify the snapshot before trusting it
sqlite3 /backups/legalmind-2026-09-12-2200.db "PRAGMA integrity_check; select count(*) from messages;"

# Restore: stop the API, put the file back, restart
sqlite3 /backups/legalmind-2026-09-12-2200.db ".backup 'data/legalmind.db'"
```

A plain-text dump, for archival or moving between machines:

```bash
sqlite3 data/legalmind.db .dump | gzip > /backups/legalmind-$(date +%F).sql.gz
```

Nightly, via cron:

```cron
15 3 * * * cd /home/sece2026-student12/LegalMindAI && \
  sqlite3 data/legalmind.db ".backup '/backups/legalmind-$(date +\%F).db'"
```

Reading a live database from a second process is fine — WAL plus
`busy_timeout` is what makes this safe:

```bash
sqlite3 data/legalmind.db "select count(*) from messages;"
```

---

## Moving to Postgres

Deliberately a small job: no call site knows it is talking to SQLite. All SQL
is inside `storage/chat_store.py`, and every caller goes through the functions
at the bottom of that file.

1. **Add the driver** — `psycopg[binary]==3.2.*` in `requirements.txt`. Still
   no ORM and no Alembic; the numbered-migration mechanism already here is
   enough.

2. **Add a DSN** — `LEGALMIND_DB_DSN`, e.g.
   `postgresql://legalmind@localhost/legalmind`. Keep `LEGALMIND_DB_PATH` as
   the SQLite fallback so nothing breaks mid-migration.

3. **Change `_connect()`** — the one function that knows how to reach the
   database. Return a `psycopg` connection; set `autocommit=True` so
   `transaction()` keeps owning transactions. Drop the four PRAGMAs (WAL,
   busy_timeout, foreign_keys and synchronous are all either default or
   meaningless there) and drop `check_same_thread`. Swap the thread-local
   connection for `psycopg_pool.ConnectionPool` — `get_connection()` is the
   only other function that changes, and now more than one process can serve.

4. **Translate placeholders** — `?` becomes `%s`. A ~20-line pass over the
   module; nothing outside it.

5. **Port the DDL in `MIGRATIONS`**, as a new migration for Postgres:
   - `TEXT PRIMARY KEY` stays `TEXT PRIMARY KEY` (still uuid4 strings — do not
     switch to `SERIAL`; ids stay portable);
   - `INTEGER` booleans (`pinned`) become `BOOLEAN`, or keep them `INTEGER` to
     avoid touching call sites;
   - `CHECK` constraints and `ON DELETE CASCADE` carry over unchanged;
   - timestamps: keep them as `TEXT` ISO8601 UTC for a like-for-like move, or
     go to `TIMESTAMPTZ` and convert in `_row_to_*` — those two functions are
     the only readers;
   - the partial unique index on `client_token` is native:
     `CREATE UNIQUE INDEX ... WHERE client_token IS NOT NULL`.

6. **Replace FTS5**, which has no Postgres equivalent. Add a `tsvector` column
   with a GIN index and a trigger, then rewrite `search_messages()` to
   `to_tsquery` / `ts_headline`. `_fts_query()`'s job — quoting each token so
   raw user input cannot be a syntax error — stays the same idea with
   `plainto_tsquery`. Drop the three FTS triggers and
   `rebuild_search_index()`.

7. **Migrate the data** — dump conversations and messages to CSV and `COPY`
   them in (tables first, then rebuild the search index). The uuid4 primary
   keys mean no id remapping is needed.

8. **Wire up accounts at the same time** — the natural moment. Change
   `get_or_create_user_id()` to return the authenticated principal's id, add a
   real `users` table, and add `conversations.user_id REFERENCES users(id)`.
   Existing rows keep their cookie-derived ids, so either backfill a claim
   step ("this browser's history is now yours") or leave them as anonymous
   users.

`tests/test_chat_store.py` is the acceptance test for the port: it drives only
the public functions, so a Postgres implementation that passes it is behaviour-
compatible. Its four SQLite-specific tests (`test_required_pragmas_are_set…`,
`test_fts_table_exists`, `test_rebuild_index_recovers_search`,
`test_concurrent_writers_in_separate_processes`) are the ones to replace.

---

## Tests

```bash
cd /home/sece2026-student12/LegalMindAI
.venv/bin/python -m unittest tests.test_chat_store -v
```

73 tests against a real temp database file — not `:memory:`, because WAL,
busy-timeout behaviour and cross-process locking are exactly what needs
testing and none of them exist in memory. Covers the schema and PRAGMAs,
message ordering, the `updated_at` bump, citation round-tripping, metadata
promotion, idempotency, sidebar operations, per-user isolation, FTS search
(including operator characters that would otherwise be a syntax error),
reconciliation, export, rollback, and concurrent writers across both threads
and separate OS processes.

stdlib `unittest`, matching `tests/test_pipeline.py` — pytest is not a
dependency of this project.

---

## Known gaps

- **`token_count` is always NULL.** `run_rag()` does not return token counts,
  and producing one means reaching into the tokenizer — an inference-layer
  change, out of scope here. The column exists; a fabricated estimate would be
  worse than a null.
- **`judges` and `paragraph_no` are always null.** Not in the corpus. See the
  *`citations` JSON* section.
- **`LEGALMIND_AUTO_TITLE` does not call a model.** A model-written title needs
  a non-RAG prompt path in `scripts/05_rag.py`: the loaded model's only entry
  point is `generate(query, retrieved_chunks, citations)`, which emits a
  structured legal answer, not a title. With the flag on, the title is built
  from the leading statutory provision the answer relied on — no extra GPU
  time. The flag and its call site exist so swapping in a real second-model
  call later is a one-line change.
- **No streaming.** `run_rag()` blocks and returns a finished string, so there
  is no partial text to persist mid-answer. `status='partial'` is reached via
  reconciliation (rule 5 above) rather than a truncated stream.
  `update_message()` is the seam a future streaming path would use: insert the
  assistant row at `status='partial'` on first chunk, patch it to `complete`
  at the end. No schema change needed.
- **Single process.** `--workers 1` is required by the GPU model, not by this
  store. WAL and `busy_timeout` already make the database safe across
  processes (the test suite proves it with four concurrent writers), so the
  store is not what blocks scaling out.
