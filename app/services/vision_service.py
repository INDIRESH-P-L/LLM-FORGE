"""
app/services/vision_service.py
==============================
Vision & Image Analysis Service for LegalMindAI.

Features:
- Image validation (JPG, JPEG, PNG, WEBP)
- File size checking (25 MB cap)
- Corrupt and empty image rejection
- Temporary file handling and auto-cleanup
- OCR integration via OCRService (with confusion detection)
- Legal entity extraction via LegalExtractionService
- Q&A orchestration via existing LegalMindAI RAG inference pipeline
- Low-confidence guarding
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

log = logging.getLogger("legalmind.vision")

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 25 * 1024 * 1024  # 25 MB


class VisionServiceError(Exception):
    """Base exception for vision processing failures."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class VisionService:
    """Handles image upload validation, OCR analysis, and RAG-integrated question answering."""

    @classmethod
    def validate_image_file(cls, filename: str, content: bytes) -> Tuple[str, str, Tuple[int, int]]:
        """
        Validates image extension, byte size, emptiness, and decodability.
        Returns (clean_filename, mime_type, (width, height)).
        """
        if not content or len(content) == 0:
            raise VisionServiceError("Uploaded image file is empty.", status_code=400)

        if len(content) > MAX_IMAGE_BYTES:
            raise VisionServiceError(f"Image exceeds maximum allowable size of {MAX_IMAGE_BYTES // (1024*1024)} MB.", status_code=400)

        clean_name = Path(filename).name
        clean_name = re.sub(r"[^\w\.-]", "_", clean_name)
        ext = Path(clean_name).suffix.lower()

        if ext not in ALLOWED_IMAGE_EXTS:
            raise VisionServiceError(
                f"Unsupported image format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTS))}",
                status_code=400
            )

        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        mime_type = mime_map.get(ext, "image/jpeg")

        # Verify decodability via Pillow
        try:
            from PIL import Image
            import io
            with Image.open(io.BytesIO(content)) as img:
                img.verify()
            with Image.open(io.BytesIO(content)) as img:
                dims = (img.width, img.height)
                if img.width < 10 or img.height < 10:
                    raise VisionServiceError("Image dimensions too small to contain readable text.", status_code=400)
        except Exception as e:
            if isinstance(e, VisionServiceError):
                raise
            raise VisionServiceError(f"Corrupt or unreadable image file: {e}", status_code=400)

        return clean_name, mime_type, dims

    @classmethod
    def process_image_upload(cls, filename: str, content: bytes) -> Dict[str, Any]:
        """Validates and stores the image into context manager."""
        clean_name, mime_type, dims = cls.validate_image_file(filename, content)
        image_id = f"img_{uuid.uuid4().hex[:10]}"

        # Write to temporary file for OCR inspection
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(clean_name).suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            ocr_res = OCRService.extract_text_from_image(tmp_path)
            extracted_text = ocr_res.text
            entities = LegalExtractionService.extract_entities(extracted_text, filename=clean_name)
            doc_type = entities.document_type

            # Register in MultimodalContextManager
            ctx_mgr = MultimodalContextManager.get_instance()
            ctx_mgr.register_context(
                context_id=image_id,
                doc_type=doc_type,
                filename=clean_name,
                full_text=extracted_text,
                entities=entities.to_dict(),
                ocr_blocks=ocr_res.blocks,
                metadata={
                    "mime_type": mime_type,
                    "dimensions": f"{dims[0]}x{dims[1]}",
                    "size_bytes": len(content),
                    "ocr_confidence": ocr_res.ocr_confidence,
                    "warnings": ocr_res.warnings,
                }
            )

            return {
                "success": True,
                "image_id": image_id,
                "filename": clean_name,
                "mime_type": mime_type,
                "size_bytes": len(content),
                "status": "uploaded",
            }
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    @classmethod
    def analyze_image(cls, image_id: str) -> Dict[str, Any]:
        """Retrieves and returns structured analysis of a previously uploaded image."""
        ctx_mgr = MultimodalContextManager.get_instance()
        ctx = ctx_mgr.get_context(image_id)
        if not ctx:
            raise VisionServiceError(f"Image ID '{image_id}' not found or expired.", status_code=404)

        entities_dict = ctx.entities
        warnings = ctx.metadata.get("warnings", [])
        ocr_conf = ctx.metadata.get("ocr_confidence", 0.0)

        # Confidence protection
        confidence_label = "low"
        if ocr_conf > 0.75 and entities_dict.get("case_number"):
            confidence_label = "high"
        elif ocr_conf > 0.4:
            confidence_label = "medium"

        return {
            "success": True,
            "image_id": image_id,
            "document_type": ctx.doc_type,
            "extracted_text": ctx.full_text[:4000],
            "entities": {
                "case_number": entities_dict.get("case_number"),
                "court": entities_dict.get("court"),
                "date": entities_dict.get("date"),
                "parties": entities_dict.get("parties", []),
                "sections": entities_dict.get("sections", []),
                "acts": entities_dict.get("acts", []),
                "judges": entities_dict.get("judges", []),
            },
            "warnings": warnings,
            "confidence": confidence_label,
        }

    @classmethod
    def ask_image_question(
        cls,
        image_id: str,
        question: str,
        rag_inference_fn: Any,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Executes question answering over the image context grounded in the existing RAG pipeline.
        """
        if not question.strip():
            raise VisionServiceError("Question cannot be empty.", status_code=400)

        ctx_mgr = MultimodalContextManager.get_instance()
        ctx = ctx_mgr.get_context(image_id)
        if not ctx:
            raise VisionServiceError(f"Image ID '{image_id}' not found or expired.", status_code=404)

        ocr_conf = ctx.metadata.get("ocr_confidence", 0.0)
        warnings = list(ctx.metadata.get("warnings", []))

        # Check for empty OCR
        if not ctx.full_text.strip() or ocr_conf < 0.2:
            warnings.append("Low OCR confidence: answer is based on general Indian legal principles.")

        # Build grounded query for the existing RAG inference engine
        relevant_excerpt = ctx.search_relevant_excerpts(question, max_chars=3000)
        grounded_prompt = (
            f"[Document Context: {ctx.filename} ({ctx.doc_type})]\n"
            f"--- EXTRACTED TEXT ---\n"
            f"{relevant_excerpt}\n"
            f"--- END CONTEXT ---\n\n"
            f"Question regarding document: {question}"
        )

        try:
            rag_res = rag_inference_fn(grounded_prompt)
            answer = rag_res.get("answer", "")
            citations = rag_res.get("citations", [])
            confidence = rag_res.get("confidence", "medium")

            # Confidence guard: if OCR is low, never return high confidence
            if ocr_conf < 0.6 and confidence == "high":
                confidence = "medium"

            return {
                "success": True,
                "answer": answer,
                "extracted_text": relevant_excerpt[:1000],
                "citations": citations,
                "confidence": confidence,
                "evidence_coverage": rag_res.get("evidence_coverage", 0.8),
                "warnings": warnings,
                "document_id": image_id,
            }
        except Exception as e:
            log.error(f"RAG execution failure for image {image_id}: {e}", exc_info=True)
            raise VisionServiceError(f"Inference execution failed: {e}", status_code=500)
