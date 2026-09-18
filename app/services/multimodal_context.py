"""
app/services/multimodal_context.py
==================================
Thread-Safe In-Memory Context Manager for Multimodal Documents & Images.

Features:
- Stores active document/image extractions, paragraphs, OCR blocks, and entity graphs.
- Maps `document_id` (doc_xxxxx) and `image_id` (img_xxxxx) to structured context.
- Enables follow-up questions:
  * "What is this document?"
  * "What is the deadline?"
  * "Explain paragraph 3."
  * "Which Act is mentioned?"
  * "Is this related to civil or criminal law?"
- Paragraph indexer and local chunk retriever for targeted document context retrieval.
- Thread-safe with automatic expiration / cache bounds.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("legalmind.context")


@dataclass
class DocumentContext:
    context_id: str
    doc_type: str
    filename: str
    full_text: str
    paragraphs: List[str] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    ocr_blocks: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_paragraph(self, paragraph_num: int) -> Optional[str]:
        """Returns 1-indexed paragraph if available."""
        idx = paragraph_num - 1
        if 0 <= idx < len(self.paragraphs):
            return self.paragraphs[idx]
        return None

    def search_relevant_excerpts(self, query: str, max_chars: int = 4000) -> str:
        """Finds paragraphs most relevant to the user query."""
        if not self.paragraphs:
            return self.full_text[:max_chars]

        # Check for direct paragraph reference e.g., "paragraph 3", "para 2"
        para_match = re.search(r"\b(?:paragraph|para\.?|clause|point)\s*(\d+)\b", query, re.I)
        if para_match:
            p_num = int(para_match.group(1))
            p_txt = self.get_paragraph(p_num)
            if p_txt:
                return f"[Target Paragraph {p_num}]:\n{p_txt}"

        # Simple term matching score across paragraphs
        query_words = set(re.findall(r"\w+", query.lower()))
        scored_paras = []
        for idx, p in enumerate(self.paragraphs):
            p_words = set(re.findall(r"\w+", p.lower()))
            overlap = len(query_words.intersection(p_words))
            if overlap > 0:
                scored_paras.append((overlap, idx, p))

        if scored_paras:
            scored_paras.sort(key=lambda x: x[0], reverse=True)
            top = [p for _, _, p in scored_paras[:4]]
            return "\n\n".join(top)[:max_chars]

        # Fallback to document prefix
        return self.full_text[:max_chars]


class MultimodalContextManager:
    """Thread-safe context store for document and image interactions."""

    _instance: Optional[MultimodalContextManager] = None
    _lock = threading.Lock()

    def __init__(self, max_items: int = 100, ttl_seconds: int = 7200):
        self._contexts: Dict[str, DocumentContext] = {}
        self._max_items = max_items
        self._ttl = ttl_seconds

    @classmethod
    def get_instance(cls) -> MultimodalContextManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register_context(
        self,
        context_id: str,
        doc_type: str,
        filename: str,
        full_text: str,
        entities: Optional[Dict[str, Any]] = None,
        ocr_blocks: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentContext:
        """Registers an uploaded document or image into memory."""
        with self._lock:
            self._cleanup_stale()

            # Split into natural paragraphs
            raw_paras = [p.strip() for p in re.split(r"\n\s*\n", full_text) if len(p.strip()) > 20]
            if not raw_paras and full_text.strip():
                raw_paras = [full_text.strip()]

            ctx = DocumentContext(
                context_id=context_id,
                doc_type=doc_type,
                filename=filename,
                full_text=full_text,
                paragraphs=raw_paras,
                entities=entities or {},
                ocr_blocks=ocr_blocks or [],
                metadata=metadata or {},
            )
            self._contexts[context_id] = ctx
            log.info(f"Registered multimodal context: {context_id} ({filename}, {len(raw_paras)} paragraphs)")
            return ctx

    def get_context(self, context_id: str) -> Optional[DocumentContext]:
        """Retrieves active document context by document_id or image_id."""
        with self._lock:
            ctx = self._contexts.get(context_id)
            if ctx and (time.time() - ctx.created_at) > self._ttl:
                del self._contexts[context_id]
                return None
            return ctx

    def delete_context(self, context_id: str) -> bool:
        """Removes a document context."""
        with self._lock:
            if context_id in self._contexts:
                del self._contexts[context_id]
                log.info(f"Evicted multimodal context: {context_id}")
                return True
            return False

    def _cleanup_stale(self) -> None:
        """Prunes expired or excessive entries."""
        now = time.time()
        expired = [cid for cid, ctx in self._contexts.items() if (now - ctx.created_at) > self._ttl]
        for cid in expired:
            del self._contexts[cid]

        # If still over limit, drop oldest
        if len(self._contexts) >= self._max_items:
            sorted_items = sorted(self._contexts.items(), key=lambda x: x[1].created_at)
            to_remove = len(self._contexts) - self._max_items + 1
            for cid, _ in sorted_items[:to_remove]:
                del self._contexts[cid]
