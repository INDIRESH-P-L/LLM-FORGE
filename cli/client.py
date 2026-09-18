"""HTTP client for LegalMindAI backend APIs."""

import json
import logging
import time
import uuid
from typing import Any, Dict, Generator, List, Optional, Tuple

import httpx

from cli.config import config

logger = logging.getLogger("legalmind.cli.client")


class BackendError(Exception):
    """Raised when backend API returns an error or is unreachable."""
    pass


class LegalMindClient:
    """Client for communicating with the LegalMindAI FastAPI backend."""

    def __init__(self, server_url: Optional[str] = None, timeout: Optional[float] = None):
        self.server_url = (server_url or config.server_url).rstrip("/")
        self.timeout = timeout or config.timeout
        self.connect_timeout = config.connect_timeout

    def _get_client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.server_url,
            timeout=httpx.Timeout(self.timeout, connect=self.connect_timeout),
            headers={"User-Agent": "LegalMindAI-CLI/1.0.0"},
        )

    def check_health(self) -> Dict[str, Any]:
        """Query /health endpoint to check server readiness.
        
        Returns dict with status, model_loaded, retriever_loaded, etc.
        """
        try:
            with self._get_client() as client:
                res = client.get("/health")
                if res.status_code == 200:
                    data = res.json()
                    data["backend_connected"] = True
                    return data
                return {
                    "backend_connected": False,
                    "model_loaded": False,
                    "retriever_loaded": False,
                    "error": f"HTTP {res.status_code}",
                }
        except httpx.ConnectError:
            return {
                "backend_connected": False,
                "model_loaded": False,
                "retriever_loaded": False,
                "error": "Connection refused",
            }
        except httpx.TimeoutException:
            return {
                "backend_connected": False,
                "model_loaded": False,
                "retriever_loaded": False,
                "error": "Connection timed out",
            }
        except Exception as e:
            return {
                "backend_connected": False,
                "model_loaded": False,
                "retriever_loaded": False,
                "error": str(e),
            }

    def get_stats(self) -> Dict[str, Any]:
        """Fetch system and model statistics from /stats."""
        try:
            with self._get_client() as client:
                res = client.get("/stats")
                if res.status_code == 200:
                    return res.json()
                return {}
        except Exception as e:
            if config.debug:
                logger.debug(f"Failed to fetch /stats: {e}")
            return {}

    def chat(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a question to the backend /api/chat endpoint.
        
        Preserves conversation_id and parses citations and confidence.
        """
        target_conv_id = conversation_id or config.active_conversation_id
        client_token = str(uuid.uuid4())

        # If an active document is attached in context and not yet referenced, inject document context
        full_query = query
        if config.active_document_name and config.active_document_text:
            if not target_conv_id:
                # Prepend document reference on the initial turn
                full_query = (
                    f"Context from uploaded document '{config.active_document_name}':\n"
                    f"\"\"\"\n{config.active_document_text[:4000]}\n\"\"\"\n\n"
                    f"User question: {query}"
                )

        payload: Dict[str, Any] = {
            "query": full_query,
            "client_token": client_token,
        }
        if target_conv_id:
            payload["conversation_id"] = target_conv_id
        if top_k:
            payload["top_k"] = top_k
        if mode:
            payload["mode"] = mode

        t0 = time.time()
        try:
            with self._get_client() as client:
                res = client.post("/api/chat", json=payload)
        except httpx.ConnectError:
            raise BackendError(f"[ERROR] Backend is unavailable at {self.server_url}. Is the server running?")
        except httpx.TimeoutException:
            raise BackendError(f"[ERROR] Connection timeout after {self.timeout:.0f} seconds.")
        except Exception as e:
            raise BackendError(f"[ERROR] Failed to communicate with backend: {e}")

        elapsed_ms = (time.time() - t0) * 1000

        if res.status_code != 200:
            try:
                err_detail = res.json().get("detail", res.text)
            except Exception:
                err_detail = res.text
            raise BackendError(f"[ERROR] Backend returned HTTP {res.status_code}: {err_detail}")

        try:
            data = res.json()
        except json.JSONDecodeError:
            raise BackendError("[ERROR] Backend returned invalid JSON response.")

        # Extract answer and metadata
        conv_id = data.get("conversation_id")
        if conv_id:
            config.active_conversation_id = conv_id

        assistant_msg = data.get("assistant_message") or {}
        answer = assistant_msg.get("content") or data.get("answer") or data.get("response") or ""
        metadata = assistant_msg.get("metadata") or {}

        citations = (
            data.get("citations")
            or metadata.get("citations")
            or data.get("sources")
            or []
        )
        confidence = (
            metadata.get("confidence")
            or data.get("confidence")
            or "MEDIUM"
        )
        evidence_coverage = (
            metadata.get("evidence_coverage")
            or data.get("evidence_coverage")
            or 0.0
        )
        warnings = (
            metadata.get("warnings")
            or data.get("warnings")
            or []
        )
        status = (
            metadata.get("status")
            or data.get("status")
            or "completed"
        )

        response_dict = {
            "conversation_id": conv_id,
            "message_id": assistant_msg.get("id"),
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "evidence_coverage": evidence_coverage,
            "warnings": warnings,
            "status": status,
            "latency_ms": elapsed_ms,
            "raw": data,
        }

        # Update config state
        config.last_response = response_dict
        config.last_citations = citations
        config.last_confidence = confidence
        config.last_latency_ms = elapsed_ms

        return response_dict

    def direct_query(
        self,
        query: str,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send a question directly to /query endpoint."""
        payload: Dict[str, Any] = {"query": query}
        if mode:
            payload["mode"] = mode
        if top_k:
            payload["top_k"] = top_k

        t0 = time.time()
        try:
            with self._get_client() as client:
                res = client.post("/query", json=payload)
        except httpx.ConnectError:
            raise BackendError(f"[ERROR] Backend is unavailable at {self.server_url}.")
        except httpx.TimeoutException:
            raise BackendError(f"[ERROR] Connection timeout after {self.timeout:.0f} seconds.")
        except Exception as e:
            raise BackendError(f"[ERROR] Query failed: {e}")

        elapsed_ms = (time.time() - t0) * 1000

        if res.status_code != 200:
            raise BackendError(f"[ERROR] Backend error HTTP {res.status_code}: {res.text}")

        data = res.json()
        citations = data.get("sources") or data.get("citations") or []
        confidence = data.get("confidence", "MEDIUM")
        evidence_coverage = data.get("evidence_coverage", 0.0)

        response_dict = {
            "answer": data.get("answer", ""),
            "citations": citations,
            "confidence": confidence,
            "evidence_coverage": evidence_coverage,
            "warnings": data.get("warnings", []),
            "latency_ms": elapsed_ms,
            "raw": data,
        }

        config.last_response = response_dict
        config.last_citations = citations
        config.last_confidence = confidence
        config.last_latency_ms = elapsed_ms

        return response_dict

    def list_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent conversations from /api/conversations."""
        try:
            with self._get_client() as client:
                res = client.get("/api/conversations")
                if res.status_code == 200:
                    convs = res.json()
                    if isinstance(convs, list):
                        return convs[:limit]
                return []
        except Exception as e:
            if config.debug:
                logger.debug(f"Failed to fetch conversations: {e}")
            return []

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific conversation and its messages."""
        try:
            with self._get_client() as client:
                res = client.get(f"/api/conversations/{conversation_id}")
                if res.status_code == 200:
                    return res.json()
                return None
        except Exception as e:
            if config.debug:
                logger.debug(f"Failed to fetch conversation {conversation_id}: {e}")
            return None
