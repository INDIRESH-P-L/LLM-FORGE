"""
app/services/document_service.py
================================
Document Processing, Analysis, and Q&A Service for LegalMindAI.

Features:
- Multi-format extraction: PDF, DOCX, TXT, and image documents.
- Legal document classification and domain-specific template breakdown:
  * Judgments (Case details, facts, issues, ratio, holding, citations)
  * Legal Notices (Issuer, demands, provisions, deadlines, advice)
  * Contracts (Parties, obligations, payment, duration, risks)
- Multi-turn document context preservation under `document_id`.
- Follow-up document question answering through the unified RAG pipeline.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.services.legal_extraction_service import LegalExtractionService
from app.services.multimodal_context import MultimodalContextManager
from app.services.ocr_service import OCRService

log = logging.getLogger("legalmind.documents")

ALLOWED_DOC_EXTS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".webp"}
MAX_DOC_BYTES = 25 * 1024 * 1024  # 25 MB


class DocumentServiceError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DocumentService:
    """Handles legal document upload, multi-format extraction, and grounded analysis."""

    @classmethod
    def validate_file(cls, filename: str, content: bytes) -> Tuple[str, str]:
        """Validates file presence, size limits, and allowable extensions."""
        if not content or len(content) == 0:
            raise DocumentServiceError("Uploaded file is empty.", status_code=400)

        if len(content) > MAX_DOC_BYTES:
            raise DocumentServiceError(
                f"File exceeds maximum allowable size of {MAX_DOC_BYTES // (1024*1024)} MB.",
                status_code=400
            )

        clean_name = Path(filename).name
        clean_name = re.sub(r"[^\w\.-]", "_", clean_name)
        ext = Path(clean_name).suffix.lower()

        if ext not in ALLOWED_DOC_EXTS:
            raise DocumentServiceError(
                f"Unsupported file format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_DOC_EXTS))}",
                status_code=400
            )

        return clean_name, ext

    @classmethod
    def extract_text_from_file(cls, path: Path, ext: str) -> Tuple[str, Dict[str, Any]]:
        """Extracts text based on document extension with robust fallback parsers."""
        meta: Dict[str, Any] = {"page_count": 1}

        if ext == ".txt":
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                return text, meta
            except Exception as e:
                raise DocumentServiceError(f"Failed to read text file: {e}")

        if ext == ".docx":
            # Try python-docx
            try:
                import docx
                doc = docx.Document(path)
                paras = [p.text for p in doc.paragraphs if p.text]
                return "\n\n".join(paras), meta
            except Exception:
                # Fallback: XML zip parsing
                try:
                    import zipfile
                    import xml.etree.ElementTree as ET
                    with zipfile.ZipFile(path) as z:
                        xml_content = z.read("word/document.xml")
                        tree = ET.fromstring(xml_content)
                        texts = [elem.text for elem in tree.iter() if elem.tag.endswith("t") and elem.text]
                        return " ".join(texts), meta
                except Exception as e:
                    raise DocumentServiceError(f"Failed to parse DOCX file: {e}")

        if ext == ".pdf":
            # 1. Try PyMuPDF
            try:
                import fitz
                doc = fitz.open(path)
                pages = [page.get_text() or "" for page in doc]
                meta["page_count"] = len(pages)
                doc.close()
                text = "\n\n".join(pages).strip()
                if text:
                    return text, meta
            except Exception:
                pass

            # 2. Try pdfplumber
            try:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    pages = [p.extract_text() or "" for p in pdf.pages]
                    meta["page_count"] = len(pages)
                    text = "\n\n".join(pages).strip()
                    if text:
                        return text, meta
            except Exception:
                pass

            # 3. Try pypdf
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                pages = [p.extract_text() or "" for p in reader.pages]
                meta["page_count"] = len(pages)
                text = "\n\n".join(pages).strip()
                if text:
                    return text, meta
            except Exception as e:
                raise DocumentServiceError(f"Failed to extract PDF text: {e}")

            raise DocumentServiceError("PDF contains no selectable text (scanned document).")

        if ext in {".png", ".jpg", ".jpeg", ".webp"}:
            ocr_res = OCRService.extract_text_from_image(path)
            meta["ocr_confidence"] = ocr_res.ocr_confidence
            meta["warnings"] = ocr_res.warnings
            return ocr_res.text, meta

        raise DocumentServiceError(f"Unsupported extraction handler for {ext}")

    @classmethod
    def process_document_upload(cls, filename: str, content: bytes) -> Dict[str, Any]:
        """Validates, extracts, and caches document in context manager."""
        clean_name, ext = cls.validate_file(filename, content)
        document_id = f"doc_{uuid.uuid4().hex[:10]}"

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            extracted_text, meta = cls.extract_text_from_file(Path(tmp_path), ext)
            if not extracted_text.strip():
                raise DocumentServiceError(f"Document '{clean_name}' contains no readable text.")

            entities = LegalExtractionService.extract_entities(extracted_text, filename=clean_name)
            doc_type = entities.document_type

            # Register in MultimodalContextManager
            ctx_mgr = MultimodalContextManager.get_instance()
            ctx_mgr.register_context(
                context_id=document_id,
                doc_type=doc_type,
                filename=clean_name,
                full_text=extracted_text,
                entities=entities.to_dict(),
                metadata={
                    "size_bytes": len(content),
                    "page_count": meta.get("page_count", 1),
                    "extension": ext,
                    "preview": extracted_text[:300],
                }
            )

            mime_map = {
                ".pdf": "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".txt": "text/plain",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
            }

            return {
                "success": True,
                "document_id": document_id,
                "filename": clean_name,
                "mime_type": mime_map.get(ext, "application/octet-stream"),
                "size_bytes": len(content),
                "status": "uploaded",
            }
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    @classmethod
    def analyze_document(cls, document_id: str) -> Dict[str, Any]:
        """Provides structured legal entity breakdown and domain templates."""
        ctx_mgr = MultimodalContextManager.get_instance()
        ctx = ctx_mgr.get_context(document_id)
        if not ctx:
            raise DocumentServiceError(f"Document ID '{document_id}' not found or expired.", status_code=404)

        entities_dict = ctx.entities
        doc_type = ctx.doc_type

        # Build domain structured template breakdown
        structured_analysis: Dict[str, Any] = {
            "document_genre": doc_type,
            "parties": entities_dict.get("parties", []),
            "court": entities_dict.get("court"),
            "case_number": entities_dict.get("case_number"),
            "date": entities_dict.get("date"),
            "statutory_acts": entities_dict.get("acts", []),
            "sections_referenced": entities_dict.get("sections", []),
            "citations": entities_dict.get("citations", []),
            "deadlines": entities_dict.get("deadlines", []),
        }

        return {
            "success": True,
            "document_id": document_id,
            "document_type": doc_type,
            "extracted_text": ctx.full_text[:6000],
            "entities": entities_dict,
            "structured_analysis": structured_analysis,
            "warnings": ctx.metadata.get("warnings", []),
            "confidence": entities_dict.get("confidence", "medium"),
        }

    @classmethod
    def ask_document_question(
        cls,
        document_id: str,
        question: str,
        rag_inference_fn: Any,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Answers a user question grounded in uploaded document excerpts and RAG legal corpus.
        """
        if not question.strip():
            raise DocumentServiceError("Question cannot be empty.", status_code=400)

        ctx_mgr = MultimodalContextManager.get_instance()
        ctx = ctx_mgr.get_context(document_id)
        if not ctx:
            raise DocumentServiceError(f"Document ID '{document_id}' not found or expired.", status_code=404)

        relevant_text = ctx.search_relevant_excerpts(question, max_chars=4500)

        grounded_prompt = (
            f"[Document Context: {ctx.filename} ({ctx.doc_type})]\n"
            f"--- RELEVANT EXCERPTS ---\n"
            f"{relevant_text}\n"
            f"--- END DOCUMENT EXCERPTS ---\n\n"
            f"User Question: {question}\n\n"
            f"Provide an objective, evidence-backed answer strictly grounded in the document above "
            f"and Indian legal statutory principles. If the document does not mention the requested point, "
            f"state so clearly."
        )

        try:
            rag_res = rag_inference_fn(grounded_prompt)
            answer = rag_res.get("answer", "")
            citations = rag_res.get("citations", [])
            confidence = rag_res.get("confidence", "medium")

            return {
                "success": True,
                "answer": answer,
                "extracted_text": relevant_text[:1200],
                "citations": citations,
                "confidence": confidence,
                "evidence_coverage": rag_res.get("evidence_coverage", 0.9),
                "warnings": rag_res.get("legal_warnings", []),
                "document_id": document_id,
            }
        except Exception as e:
            log.error(f"RAG execution failure for document {document_id}: {e}", exc_info=True)
            raise DocumentServiceError(f"Inference execution failed: {e}", status_code=500)

    @classmethod
    def get_document_info(cls, document_id: str) -> Dict[str, Any]:
        """Returns metadata about the active document."""
        ctx_mgr = MultimodalContextManager.get_instance()
        ctx = ctx_mgr.get_context(document_id)
        if not ctx:
            raise DocumentServiceError(f"Document ID '{document_id}' not found or expired.", status_code=404)

        return {
            "success": True,
            "document_id": document_id,
            "filename": ctx.filename,
            "document_type": ctx.doc_type,
            "paragraph_count": len(ctx.paragraphs),
            "char_count": len(ctx.full_text),
            "created_at": ctx.created_at,
            "metadata": ctx.metadata,
        }

    @classmethod
    def delete_document(cls, document_id: str) -> Dict[str, Any]:
        """Evicts document from memory."""
        ctx_mgr = MultimodalContextManager.get_instance()
        deleted = ctx_mgr.delete_context(document_id)
        if not deleted:
            raise DocumentServiceError(f"Document ID '{document_id}' not found.", status_code=404)

        return {
            "success": True,
            "document_id": document_id,
            "status": "deleted"
        }
