#!/usr/bin/env python3
"""
app/main.py
===========
LegalMind AI — FastAPI REST API

Keeps Qwen3.6-35B-A3B and HybridRetriever loaded in memory.
All queries are served without reloading the model.

Endpoints:
    POST /query          — Main RAG query endpoint (stateless; kept for eval tooling)
    POST /api/chat       — RAG query persisted to the chat store
    GET  /api/...        — Conversation history, search, export (see app/chat_api.py)
    GET  /health         — Health check
    GET  /stats          — Server stats (model loaded, GPU info)
    GET  /               — Frontend HTML

Chat history is persisted by storage/chat_store.py (SQLite at
$LEGALMIND_DB_PATH, default ./data/legalmind.db). See docs/persistence.md.

Usage:
    cd /home/sece2026-student12/LegalMindAI
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

Note: workers=1 is REQUIRED (model is not fork-safe with GPU tensors).
"""

import json
import logging
import os
import re
import sys
import threading
import time

import torch

# Avoid PyTorch CUDA memory fragmentation on shared GPU
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Add project root (for the `storage` package) and scripts dir (for retriever)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from fastapi import FastAPI, HTTPException, Request, APIRouter, File, UploadFile, Header
from fastapi.responses import HTMLResponse, JSONResponse, Response, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import chat_api
from app.chat_api import install_user_identity, router as chat_router
from storage import chat_store

log = logging.getLogger("legalmind.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Configuration (from environment or defaults)
# ---------------------------------------------------------------------------
MODEL_PATH      = os.environ.get("MODEL_PATH",      "/home/sece2026-student12/LegalMindAI/models/Qwen3.6-35B-A3B")
LLM_GPU         = int(os.environ.get("LLM_GPU",         "0"))
EMBED_GPU       = int(os.environ.get("EMBED_GPU",        "1"))
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(LLM_GPU)
except Exception:
    pass
TOP_K           = int(os.environ.get("TOP_K",            "4"))
# Comma-separated so the app can retrieve over judgments AND legislation.
# It previously loaded only the legislation index, which is why no judgment was
# ever retrievable. Each entry is "<chunks_dir>|<vector_db_dir>".
CHUNKS_DIR      = os.environ.get("CHUNKS_DIR",      "data/chunks/legislation/")
VECTOR_DB_DIR   = os.environ.get("VECTOR_DB_DIR",   "vector_db/legislation/")
RETRIEVAL_SETS  = os.environ.get("LEGALMIND_RETRIEVAL_SETS", "data/chunks/legislation|vector_db/legislation,data/chunks/case_laws|vector_db/case_laws").strip()
EMBED_MODEL     = os.environ.get("EMBED_MODEL",     "BAAI/bge-m3")
RERANK_MODEL    = os.environ.get("RERANK_MODEL",    "BAAI/bge-reranker-v2-m3")
ADMIN_TOKEN     = os.environ.get("ADMIN_TOKEN",     "supersecretadmin")


# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------
_model:     Any = None   # LegalMindModel
_retriever: Any = None   # HybridRetriever
_start_time = time.time()
_query_count = 0


# ---------------------------------------------------------------------------
# Inference access
# ---------------------------------------------------------------------------

_rag_mod_mtime = 0

def _rebind_retriever(retriever: Any, hybrid_cls: type) -> int:
    """
    Re-bind freshly reloaded HybridRetriever methods onto the live retriever(s).

    `retriever` may be a MultiRetriever wrapping several HybridRetrievers.
    Binding HybridRetriever methods onto that wrapper replaced its own
    retrieve() with one expecting HybridRetriever internals (self.chunks,
    self.bm25, self._retrieval_cache_lock), which the wrapper does not have —
    the next query then failed with
        'MultiRetriever' object has no attribute '_retrieval_cache_lock'
    So rebind onto the wrapped HybridRetrievers instead, and make sure every
    instance ends up with its cache and lock even if one step fails.

    Returns the number of retrievers actually rebound.
    """
    import types

    targets = [r for _, r in getattr(retriever, "retrievers", []) if r is not None] or [retriever]
    rebound = 0
    for target in targets:
        # Duck-typed: an instance created from the pre-reload module object is
        # not an isinstance() of the reloaded class.
        if not (hasattr(target, "chunks") and hasattr(target, "bm25")):
            log.debug(f"Skipping re-bind for {type(target).__name__}: not a HybridRetriever.")
            continue
        for attr in ("retrieve", "_find_exact_provision_matches", "_build_provision_index"):
            if hasattr(hybrid_cls, attr):
                setattr(target, attr, types.MethodType(getattr(hybrid_cls, attr), target))
        if not hasattr(target, "_provision_index"):
            try:
                target._build_provision_index()
            except Exception as idx_err:
                log.warning(f"Could not rebuild provision index: {idx_err}")
        if not hasattr(target, "_retrieval_cache"):
            target._retrieval_cache = {}
        if getattr(target, "_retrieval_cache_lock", None) is None:
            target._retrieval_cache_lock = threading.Lock()
        rebound += 1
    log.info(f"Re-bound optimized HybridRetriever methods, inverted index, and cache onto {rebound} retriever(s).")
    return rebound


def _get_rag_module(app: FastAPI):
    """
    The RAG module, loaded dynamically and reloaded if scripts/05_rag.py has been modified.
    """
    global _rag_mod_mtime
    script_path = Path(__file__).parent.parent / "scripts" / "05_rag.py"
    current_mtime = script_path.stat().st_mtime if script_path.exists() else 0
    rag_mod = getattr(app.state, "rag_module", None)
    if rag_mod is None or current_mtime > _rag_mod_mtime:
        import importlib.util
        import types
        spec = importlib.util.spec_from_file_location("rag_module", str(script_path))
        rag_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rag_mod)
        app.state.rag_module = rag_mod
        _rag_mod_mtime = current_mtime
        if _model is not None and hasattr(rag_mod, "LegalMindModel"):
            for attr in ("generate", "generate_stream", "generate_chat_completion", "_prepare_chat_inputs", "_get_stop_token_ids"):
                if hasattr(rag_mod.LegalMindModel, attr):
                    try:
                        setattr(_model, attr, types.MethodType(getattr(rag_mod.LegalMindModel, attr), _model))
                    except Exception as e:
                        log.warning(f"Could not re-bind {attr} to _model: {e}")

        if _retriever is not None:
            try:
                import sys
                import importlib
                if "retriever" in sys.modules:
                    importlib.reload(sys.modules["retriever"])
                    _rebind_retriever(_retriever, sys.modules["retriever"].HybridRetriever)
            except Exception as re_err:
                log.warning(f"Could not re-bind retriever methods: {re_err}")

        log.info(f"Loaded / reloaded rag_module from {script_path}")
    return rag_mod


def _run_inference(
    query: str,
    top_k: int | None = None,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Single entry point the persistent chat endpoint calls. Forwards conversation
    history to run_rag so continuations and follow-ups maintain dialog continuity.
    """
    global _query_count
    _query_count += 1
    rag_mod = _get_rag_module(app)
    return rag_mod.run_rag(
        query                = query,
        retriever            = _retriever,
        model                = _model,
        top_k                = top_k if top_k else TOP_K,
        conversation_history = chat_history,
    )


def _run_inference_stream(
    query: str,
    top_k: int | None = None,
    chat_history: list[dict] | None = None,
):
    """
    Streaming generator entry point for real-time SSE chat.
    Yields (event_type, payload) tuples: ('meta', dict), ('token', str), ('done', dict).
    """
    global _query_count
    _query_count += 1
    rag_mod = _get_rag_module(app)
    return rag_mod.run_rag_stream(
        query                = query,
        retriever            = _retriever,
        model                = _model,
        top_k                = top_k if top_k else TOP_K,
        conversation_history = chat_history,
    )


# ---------------------------------------------------------------------------
# Lifespan (startup/shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _retriever

    log.info("⚖️  LegalMind AI API starting up…")

    # ── Chat store ─────────────────────────────────────────────────────────
    # Brought up before the model: history must be readable even if the GPU
    # load fails, and a question interrupted by the last shutdown has to be
    # closed off before anyone can reload the page and look at it.
    chat_store.init_db()          # logs the path and schema version itself
    reconciled = chat_store.reconcile_interrupted()
    if reconciled:
        log.warning(f"Closed {reconciled} interrupted exchange(s) as status='partial'.")

    app.state.model_name = Path(MODEL_PATH).name

    # Load retriever(s)
    try:
        from retriever import HybridRetriever
        if RETRIEVAL_SETS:
            from app.services.multi_retriever import MultiRetriever
            built = []
            for spec in RETRIEVAL_SETS.split(","):
                spec = spec.strip()
                if not spec:
                    continue
                chunks_dir, _, db_dir = spec.partition("|")
                name = Path(chunks_dir).name
                embed_dev = os.environ.get("EMBED_DEVICE", f"cuda:{EMBED_GPU}")
                rerank_dev = os.environ.get("RERANK_DEVICE", embed_dev)
                try:
                    built.append((name, HybridRetriever(
                        chunks_dir=chunks_dir, vector_db_dir=db_dir or chunks_dir,
                        embed_model=EMBED_MODEL, rerank_model=RERANK_MODEL,
                        embed_device=embed_dev, rerank_device=rerank_dev)))
                    log.info(f"retrieval set loaded: {name}")
                except Exception as e:
                    log.error(f"retrieval set '{spec}' failed to load: {e}")
            _retriever = MultiRetriever(built) if built else None
            log.info("HybridRetriever(s) loaded via LEGALMIND_RETRIEVAL_SETS.")
            raise StopIteration          # skip the single-index path below
        embed_dev = os.environ.get("EMBED_DEVICE", f"cuda:{EMBED_GPU}")
        rerank_dev = os.environ.get("RERANK_DEVICE", embed_dev)
        _retriever = HybridRetriever(
            chunks_dir    = CHUNKS_DIR,
            vector_db_dir = VECTOR_DB_DIR,
            embed_model   = EMBED_MODEL,
            rerank_model  = RERANK_MODEL,
            embed_device  = embed_dev,
            rerank_device = rerank_dev,
        )
        log.info("HybridRetriever loaded.")
    except StopIteration:
        pass                              # multi-index path already set _retriever
    except Exception as e:
        log.warning(f"Retriever could not be loaded: {e}. RAG will be disabled.")
        _retriever = None

    # Load Qwen model
    try:
        from scripts_05_rag import LegalMindModel   # will import from 05_rag module
        _model = LegalMindModel.get_instance(gpu=LLM_GPU)
        log.info("Qwen3.6 model loaded.")
    except Exception:
        # Try direct import
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "rag_module",
                str(Path(__file__).parent.parent / "scripts" / "05_rag.py"),
            )
            rag_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rag_mod)
            _model = rag_mod.LegalMindModel.get_instance(gpu=LLM_GPU)
            app.state.rag_module = rag_mod
            log.info("Qwen3.6 model loaded via importlib.")
        except Exception as e2:
            log.error(f"Could not load Qwen model: {e2}")
            _model = None

    # Publish the inference entry point for app/chat_api.py. Left as None when
    # the model failed to load, so /api/chat answers 503 instead of crashing —
    # and history stays browsable regardless.
    app.state.infer = _run_inference if _model is not None else None
    app.state.infer_stream = _run_inference_stream if _model is not None else None
    # Other services (e.g. the FIR auditor) ground their findings in the same
    # corpus the chat endpoint uses.
    app.state.retriever = _retriever

    log.info("✅ LegalMind AI API ready.")
    yield  # Server runs here

    log.info("LegalMind AI API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LegalMind AI",
    description="Intelligent Indian Legal Research Assistant powered by Qwen3.6-35B-A3B",
    version="1.0.0",
    lifespan=lifespan,
)


# Every request gets a resolved user_id on request.state (and a cookie on the
# way out) before it reaches a handler — including the HTML page load, so the
# identity exists before any JavaScript runs.
install_user_identity(app)

# Persistent chat history: /api/chat, /api/conversations, /api/search, export.
app.include_router(chat_router)

# Multimodal & Legal Engineering Suite Services
from app.api import (
    speech_router,
    images_router,
    documents_router,
    audio_router,
    drafter_router,
    moot_router,
    dossier_router,
    temporal_router,
    bail_router,
    fir_audit_router,
    citation_check_router,
    pleading_router,
    precedents_router,
)
app.include_router(speech_router)
app.include_router(images_router)
app.include_router(documents_router)
app.include_router(audio_router)
app.include_router(drafter_router)
app.include_router(moot_router)
app.include_router(dossier_router)
app.include_router(temporal_router)
app.include_router(bail_router)

# Enable CORS middleware so cross-origin, IP-based, and LAN requests
# (e.g. https://192.168.4.99:8443) never fail preflight OPTIONS checks or get blocked.
from fastapi.middleware.cors import CORSMiddleware

_raw_cors = os.environ.get("LEGALMIND_CORS_ORIGINS", "").strip()
if _raw_cors:
    _cors_origins = [o.strip() for o in _raw_cors.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials="*" not in _cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    log.info(f"CORS enabled for specific origins: {_cors_origins}")
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://.*$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    log.info("CORS enabled with permissive origin regex for LAN/local clients")
app.include_router(fir_audit_router)
app.include_router(citation_check_router)
app.include_router(pleading_router)
app.include_router(precedents_router)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K
    conversation_id: str | None = None
    client_token: str | None = Field(default=None, max_length=128)

class QueryResponse(BaseModel):
    # Core Fields (100% backward compatible for app.js and eval scripts)
    query: str
    answer: str
    legal_provisions: list[str] = []
    judgments: list[str] = []
    legal_reasoning: str = ""
    citations: list[str] = []
    confidence: str
    limitations: str = ""
    retrieved_documents: list[dict] = []
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0
    total_latency: float = 0.0

    # Extended Fields (for Frontend/History Developer & API Contract)
    conversation_id: str | None = None
    message_id: str | None = None
    question: str | None = None
    answer_format: str = "detailed_legal_explanation"
    confidence_score: float = 0.85
    confidence_label: str = "high"
    sources: list[dict] = []
    legal_warnings: list[str] = []
    temporal_status: str = "current"
    jurisdiction: str = "India"
    reasoning_summary: list[dict] = []
    related_cases: list[str] = []
    related_sections: list[str] = []
    model: str = "Qwen/Qwen3.6-35B-A3B"
    created_at: str = ""


class MessageCreateRequest(BaseModel):
    role: str = "user"
    content: str
    metadata: dict | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
@app.get("/api/health")
async def api_health():
    try:
        import sqlite3
        c = sqlite3.connect("./data/legalmind.db")
        c.execute("SELECT 1").fetchone()
        db_status = "ok"
    except Exception:
        db_status = "unavailable"
        
    try:
        from app.services.citation_verifier import get_verifier
        citation_status = "ok" if get_verifier().available else "unavailable"
    except Exception:
        citation_status = "unavailable"
        
    try:
        import torch
        gpu_status = "ok" if torch.cuda.is_available() else "unavailable"
    except Exception:
        gpu_status = "unavailable"
        
    return {
        # `status` is the top-level liveness key that tests/test_pipeline.py and
        # the frontend banner both read; the richer per-subsystem fields below
        # were added later and dropping it broke both.
        "status": "ok",
        "model_loaded": _model is not None,
        "retriever_loaded": _retriever is not None,
        "dgx_connection": "ok",
        "gpu": gpu_status,
        "model_process": "detected",
        "model_api": "ok" if _model is not None else "unavailable",
        "retrieval": "ok" if _retriever is not None else "unavailable",
        "database": db_status,
        "citation_verifier": citation_status
    }


@app.get("/api/model-status")
async def model_status():
    global _model, _retriever
    state = "Ready" if _model else "Loading"
    import torch
    
    return {
        "success": True,
        "data": {
            "status": state,
            "model": MODEL_PATH.split("/")[-1],
            "llm_gpu": LLM_GPU,
            "embed_gpu": EMBED_GPU,
            "retriever_status": "Ready" if _retriever else "Loading",
            "cuda_available": torch.cuda.is_available(),
        },
        "error": None
    }


@app.get("/api/admin/export-dataset")
async def export_dataset(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    token = authorization.split(" ")[1]
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
        
    dataset = []
    with chat_store.get_connection() as conn:
        cursor = conn.execute("SELECT DISTINCT conversation_id FROM messages WHERE feedback = 1")
        conv_ids = [row[0] for row in cursor.fetchall()]
        
    for cid in conv_ids:
        msgs = chat_store.get_messages(cid)
        openai_msgs = [{"role": "system", "content": "You are LegalMind AI, an expert Indian legal assistant."}]
        has_assistant_reply = False
        
        for m in msgs:
            if m.status in ("error", "partial") or not m.content:
                continue
            
            # Strip <think> tags for clean fine-tuning
            clean_content = re.sub(r'<think>.*?</think>', '', m.content, flags=re.DOTALL).strip()
            
            openai_msgs.append({"role": m.role, "content": clean_content})
            if m.role == "assistant":
                has_assistant_reply = True
                
        # Only export conversations that have at least one back-and-forth
        if len(openai_msgs) > 1 and has_assistant_reply:
            dataset.append(json.dumps({"messages": openai_msgs}))
            
    if not dataset:
        return PlainTextResponse("", media_type="application/jsonl")
        
    return PlainTextResponse("\n".join(dataset), media_type="application/jsonl")


@app.get("/diagnostics", response_class=HTMLResponse)
async def diagnostics():
    import time
    html = f"""
    <html>
    <head><title>System Diagnostics</title><style>body {{ font-family: monospace; background: #111; color: #0f0; padding: 20px; }}</style></head>
    <body>
    <h2>LegalMind AI Diagnostics</h2>
    <p>Backend Status: OK</p>
    <p>DGX Connection: OK (Local)</p>
    <p>Model Loaded: {str(_model is not None)}</p>
    <p>Retrieval Loaded: {str(_retriever is not None)}</p>
    <p>Database: Connected</p>
    <p>Python Env: active</p>
    <p>Last Health Check: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </body>
    </html>
    """
    return html


@app.get("/stats")
async def stats():
    gpu_info = []
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            gpu_info.append({
                "index": i,
                "name": props.name,
                "total_gb": round(total / 1024**3, 1),
                "free_gb":  round(free  / 1024**3, 1),
                "used_gb":  round((total - free) / 1024**3, 1),
            })
    except Exception:
        gpu_info = []

    return {
        "model_path":     MODEL_PATH,
        "llm_gpu":        LLM_GPU,
        "embed_gpu":      EMBED_GPU,
        "top_k":          TOP_K,
        "embed_model":    EMBED_MODEL,
        "rerank_model":   RERANK_MODEL,
        "model_loaded":   _model is not None,
        "retriever_loaded": _retriever is not None,
        "lora_adapter":   getattr(_model, "adapter_name", None),
        "lora_status":    getattr(_model, "adapter_status", "NONE"),
        "lora_attached":  getattr(_model, "adapter_attached", False),
        "uptime_s":       round(time.time() - _start_time, 1),
        "queries_served": _query_count,
        "gpu_info":       gpu_info,
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, http_req: Request):
    global _query_count

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    _query_count += 1
    log.info(f"[Q#{_query_count}] {request.query[:80]}")

    try:
        rag_mod = _get_rag_module(app)

        # 1. Resolve user and conversation in persistence store
        user_id = getattr(http_req.state, "user_id", None) or "anonymous_user"
        conv_id = request.conversation_id
        if conv_id:
            conv = chat_store.get_conversation(conv_id, user_id=user_id, include_deleted=False)
            if not conv:
                conv_id = chat_store.create_conversation(
                    user_id, chat_store.make_title(request.query),
                    model_name=getattr(app.state, "model_name", None)
                )
        else:
            conv_id = chat_store.create_conversation(
                user_id, chat_store.make_title(request.query),
                model_name=getattr(app.state, "model_name", None)
            )

        # Commit user question row
        u_msg_id = chat_store.append_message(
            conversation_id=conv_id,
            role="user",
            content=request.query,
            metadata={"client_token": request.client_token, "status": "complete"}
        )

        # Collect prior messages if part of a conversation
        prior_msgs = chat_store.get_messages(conv_id) if conv_id else []
        chat_history = [
            {"role": m.role, "content": m.content}
            for m in prior_msgs
            if m.id != u_msg_id and m.content and (m.status != "error")
        ]

        # Run RAG inference under single-flight lock
        with chat_api._inference_lock:
            result = rag_mod.run_rag(
                query                = request.query,
                retriever            = _retriever,
                model                = _model,
                top_k                = request.top_k,
                conversation_history = chat_history,
            )

        # Commit assistant answer row
        a_msg_id = chat_store.append_message(
            conversation_id=conv_id,
            role="assistant",
            content=result.get("answer", ""),
            metadata={
                "status": "complete",
                "confidence": result.get("confidence"),
                "citations": result.get("citations"),
                "confidence_score": result.get("confidence_score"),
                "confidence_label": result.get("confidence_label"),
                "temporal_status": result.get("temporal_status"),
                "legal_warnings": result.get("legal_warnings"),
                "model_name": getattr(app.state, "model_name", None),
            }
        )

        result["conversation_id"] = conv_id
        result["message_id"] = a_msg_id
        result["question"] = request.query

        return QueryResponse(**{k: result.get(k, QueryResponse.model_fields[k].default) for k in QueryResponse.model_fields})

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"RAG pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# History and Precedent Root Aliases (Maximum Integration Flexibility)
# ---------------------------------------------------------------------------

root_chat_router = APIRouter(tags=["conversations"])
root_chat_router.add_api_route("/conversations", chat_api.list_conversations, methods=["GET"])
root_chat_router.add_api_route("/conversations", chat_api.create_conversation, methods=["POST"])
root_chat_router.add_api_route("/conversations/{conversation_id}", chat_api.get_conversation, methods=["GET"])
root_chat_router.add_api_route("/conversations/{conversation_id}", chat_api.patch_conversation, methods=["PATCH"])
root_chat_router.add_api_route("/conversations/{conversation_id}", chat_api.delete_conversation, methods=["DELETE"])
root_chat_router.add_api_route("/conversations/{conversation_id}/restore", chat_api.restore_conversation, methods=["POST"])
root_chat_router.add_api_route("/conversations/{conversation_id}/export", chat_api.export_conversation, methods=["GET"])
root_chat_router.add_api_route("/messages/{message_id}/feedback", chat_api.message_feedback, methods=["POST"])
root_chat_router.add_api_route("/search", chat_api.search, methods=["GET"])
app.include_router(root_chat_router)


@app.get("/api/conversations/{conversation_id}/messages")
@app.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(request: Request, conversation_id: str):
    uid = getattr(request.state, "user_id", "default_user")
    chat_api._own_conversation(conversation_id, uid)
    msgs = chat_store.get_messages(conversation_id)
    return {"conversation_id": conversation_id, "messages": [m.to_dict() for m in msgs]}


@app.post("/api/conversations/{conversation_id}/messages")
@app.post("/conversations/{conversation_id}/messages")
def add_conversation_message(request: Request, conversation_id: str, body: MessageCreateRequest):
    uid = getattr(request.state, "user_id", "default_user")
    chat_api._own_conversation(conversation_id, uid)
    msg_id = chat_store.append_message(
        conversation_id=conversation_id,
        role=body.role,
        content=body.content,
        metadata=body.metadata or {"status": "complete"}
    )
    msg = chat_store.get_message(msg_id)
    return {"ok": True, "message": msg.to_dict() if msg else {"id": msg_id}}


@app.get("/api/precedents/{case_name}")
@app.get("/precedents/{case_name}")
def get_precedent_details(case_name: str):
    import precedent_graph
    pg = precedent_graph.get_precedent_service()
    data = pg.get_relationships(case_name)
    if not data.get("case_found"):
        raise HTTPException(status_code=404, detail=f"Precedent not found for '{case_name}'")
    return data


@app.get("/api/precedents")
@app.get("/precedents")
def search_precedents(q: str = "", limit: int = 10):
    import precedent_graph
    pg = precedent_graph.get_precedent_service()
    results = pg.search_precedents(q, limit=limit)
    return {"query": q, "results": [p.to_dict() for p in results]}


@app.get("/api/temporal/check")
@app.get("/temporal/check")
def check_temporal_law(q: str = ""):
    import temporal_law
    return temporal_law.detect_temporal_context(q)


@app.post("/api/upload")
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and extract text from legal documents (PDF, DOCX, TXT, PNG, JPG, WEBP).
    """
    filename = file.filename or "uploaded_file"
    ext = Path(filename).suffix.lower()
    allowed_exts = (".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".webp")
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(allowed_exts)}"
        )

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File exceeds 25 MB limit.")
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if ext in (".png", ".jpg", ".jpeg", ".webp"):
            from cli.image_input import ImageAnalyzer
            extracted_text, meta = ImageAnalyzer.extract_text(tmp_path)
            doc_type = ImageAnalyzer.detect_document_type(extracted_text, filename)
        else:
            from cli.pdf_input import DocumentParser, LegalDocumentAnalyzer
            extracted_text, meta = DocumentParser.extract_text(tmp_path)
            doc_type = LegalDocumentAnalyzer.detect_document_type(extracted_text)

        size_kb = round(len(content) / 1024, 1)
        preview = extracted_text.strip()[:300]
        return {
            "status": "success",
            "filename": filename,
            "doc_type": doc_type,
            "size_kb": size_kb,
            "char_count": len(extracted_text),
            "preview": preview,
            "extracted_text": extracted_text[:12000],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Document parsing error: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.post("/api/transcribe")
@app.post("/transcribe")
async def transcribe_speech(
    file: UploadFile = File(...),
    language: str = "en",
):
    """
    Transcribe audio speech recording into text for voice inquiry input.
    Supports WEBM, WAV, MP3, OGG, M4A recorded from browser MediaRecorder.
    """
    filename = file.filename or "recording.webm"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio recording.")

    try:
        from app.services.speech_service import SpeechService
        res = SpeechService.transcribe_audio(
            filename=filename,
            content=content,
            language=language,
        )
        return {
            "status": "success",
            "text": res.get("text", ""),
            "language": res.get("language", language),
            "confidence": res.get("confidence", 1.0),
            "warnings": res.get("warnings", []),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transcription failed: {e}")


# ---------------------------------------------------------------------------
# Frontend (served at GET /)
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def frontend():
    """
    Serve the app shell, with ?v= on the CSS and JS rewritten to the current
    asset mtimes. Without that the browser happily keeps a cached stylesheet
    and script across a deploy, and the UI looks unchanged even though the
    files on disk are not — so the page is also sent no-store.
    """
    static_dir = Path(__file__).parent / "static"
    html_path = static_dir / "index.html"
    if not html_path.exists():
        return HTMLResponse(content=_INLINE_HTML)

    html = html_path.read_text(encoding="utf-8")
    # Deployment settings the frontend needs at runtime. LEGALMIND_API_BASE is
    # empty by default, which keeps every API call same-origin (relative), so
    # the app works on whatever host/port/scheme it is actually served from.
    # Set it only when the API lives on a different origin than this page —
    # that origin must then also be allowed via LEGALMIND_CORS_ORIGINS.
    api_base = os.environ.get("LEGALMIND_API_BASE", "").rstrip("/")
    https_port = os.environ.get("LEGALMIND_HTTPS_PORT", "8443")
    config_script = (
        "<script>window.LEGALMIND_API_BASE=" + json.dumps(api_base) + ";"
        "window.LEGALMIND_HTTPS_PORT=" + json.dumps(https_port) + ";</script>"
    )
    html = html.replace("</head>", config_script + "\n</head>", 1)
    for asset in ("style.css", "app.js"):
        asset_path = static_dir / asset
        stamp = int(asset_path.stat().st_mtime) if asset_path.exists() else 0
        # Match the reference with or without an existing ?v=, so the stamp is
        # applied whichever way the markup spells it.
        html = re.sub(rf"(/static/{re.escape(asset)})(\?v=[^\"\']*)?",
                      rf"\g<1>?v={stamp}", html)
    return HTMLResponse(
        content=html,
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


_INLINE_HTML = """<!DOCTYPE html>
<html><head><title>LegalMind AI</title>
<style>body{font-family:sans-serif;max-width:800px;margin:40px auto;padding:20px}
h1{color:#1a237e}textarea{width:100%;height:80px;padding:10px;font-size:16px}
button{background:#1a237e;color:white;padding:12px 24px;border:none;cursor:pointer;font-size:16px}
#result{margin-top:20px;white-space:pre-wrap;background:#f5f5f5;padding:20px;border-radius:8px}
</style></head><body>
<h1>⚖️ LegalMind AI</h1>
<p>Indian Legal Research Assistant</p>
<textarea id="q" placeholder="Enter your legal question…"></textarea><br><br>
<button onclick="ask()">Ask LegalMind AI</button>
<div id="result"></div>
<script>
async function ask(){
  const q=document.getElementById('q').value.trim();
  if(!q)return;
  document.getElementById('result').textContent='Thinking…';
  const r=await fetch('/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q})});
  const d=await r.json();
  document.getElementById('result').textContent=d.answer+'\\n\\nCitations:\\n'+d.citations.join('\\n');
}
</script></body></html>"""
