"""Image input, validation, and OCR pipeline for LegalMindAI CLI."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cli.config import config
from cli.validators import validate_image_file, ValidationError


class ImageAnalyzer:
    """Validates images, executes OCR extraction, checks quality, and prepares queries."""

    @staticmethod
    def extract_text(file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Validate image, extract text via OCR, and determine quality heuristics."""
        path, mime_type, (width, height) = validate_image_file(file_path)
        metadata: Dict[str, Any] = {
            "file_name": path.name,
            "mime_type": mime_type,
            "dimensions": f"{width}x{height}",
            "file_size_kb": round(path.stat().st_size / 1024, 1),
            "ocr_uncertain": False,
            "ocr_method": "none",
        }

        extracted_text = ""
        # Attempt OCR using pytesseract
        try:
            from PIL import Image
            import pytesseract

            with Image.open(path) as img:
                # Convert to grayscale for better contrast
                gray = img.convert("L")
                extracted_text = pytesseract.image_to_string(gray)
                metadata["ocr_method"] = "pytesseract"
        except Exception as e:
            # Pytesseract or tesseract binary not available / failed
            metadata["ocr_error"] = str(e)

        # Evaluate quality heuristics
        raw_text = extracted_text.strip()
        if not raw_text:
            # Fallback text representation indicating image characteristics
            metadata["ocr_uncertain"] = True
            raw_text = (
                f"[Image Document: {path.name}, Format: {mime_type}, Dimensions: {width}x{height}]\n"
                f"Note: Direct OCR extraction yielded no text or OCR binary is not installed on this host."
            )
        else:
            # Check for high garbled/uncertain character ratio
            total_chars = len(raw_text)
            alphanumeric = sum(1 for c in raw_text if c.isalnum() or c.isspace())
            ratio = alphanumeric / max(total_chars, 1)
            if ratio < 0.75:
                metadata["ocr_uncertain"] = True

        metadata["char_count"] = len(raw_text)
        metadata["word_count"] = len(raw_text.split())
        return raw_text, metadata

    @staticmethod
    def detect_document_type(text: str, file_name: str) -> str:
        """Classify image document type based on extracted text and filename."""
        text_lower = text.lower()
        name_lower = file_name.lower()

        if "notice" in name_lower or any(k in text_lower for k in ["legal notice", "statutory notice", "demand notice"]):
            return "Legal Notice"
        elif "fir" in name_lower or any(k in text_lower for k in ["first information report", "police station", "cognizable"]):
            return "First Information Report (FIR)"
        elif any(k in text_lower for k in ["in the supreme court", "in the high court", "order", "judgment", "decree"]):
            return "Court Order / Judgment"
        elif any(k in text_lower for k in ["act, 19", "act, 20", "gazette of india", "ordinance"]):
            return "Statute / Legislative Act"
        elif any(k in text_lower for k in ["affidavit", "solemnly affirm", "deponent"]):
            return "Affidavit"
        return "Legal Document (Image)"

    @staticmethod
    def build_image_prompt(
        file_name: str,
        doc_type: str,
        extracted_text: str,
        user_question: Optional[str] = None,
        is_uncertain: bool = False,
    ) -> str:
        """Create prompt to submit to LegalMindAI backend."""
        uncertainty_note = ""
        if is_uncertain:
            uncertainty_note = (
                "\nNote: The text was extracted via OCR and may have OCR artifacts. "
                "Base your legal analysis strictly on recognized principles and verify statutory citations.\n"
            )

        if user_question:
            return (
                f"You are LegalMindAI, an Indian legal research assistant.\n"
                f"The user has uploaded a document image '{file_name}' (identified as {doc_type}).\n"
                f"{uncertainty_note}\n"
                f"--- EXTRACTED TEXT FROM IMAGE ---\n"
                f"{extracted_text[:5000]}\n"
                f"--- END EXTRACTED TEXT ---\n\n"
                f"User Question: {user_question}"
            )

        return (
            f"You are LegalMindAI, an Indian legal research assistant.\n"
            f"Analyze the following {doc_type} image '{file_name}'.\n"
            f"{uncertainty_note}\n"
            f"--- EXTRACTED TEXT FROM IMAGE ---\n"
            f"{extracted_text[:5000]}\n"
            f"--- END EXTRACTED TEXT ---\n\n"
            f"Provide a structured legal breakdown including:\n"
            f"1. Document Classification & Context\n"
            f"2. Parties & Entities Identified\n"
            f"3. Statutory Sections & Acts Mentioned\n"
            f"4. Key Facts, Demands, or Directives\n"
            f"5. Legal Significance & Recommended Actions"
        )
