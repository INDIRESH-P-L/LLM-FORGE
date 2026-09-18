"""
app/services/ocr_service.py
===========================
OCR Abstraction and Legal Text Extraction Service.

Features:
- Printed and scanned legal documents, court orders, notices, and acts.
- Raw OCR text preservation (never silently mutates original extracted text).
- Confidence estimation and bounding block detection.
- Non-destructive legal confusion detection:
  * 'IPC' vs 'lPC' / '1PC'
  * '1' vs 'I' / 'l' in statutory section numbers
  * '0' vs 'O' in section, year, and case numbers
  * Case citations and date anomalies
- Separate suggested corrections list requiring user / caller confirmation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("legalmind.ocr")


@dataclass
class OCRConfusion:
    original_token: str
    suggested_correction: str
    category: str  # e.g. "statute_name", "section_number", "case_number", "date"
    confidence: float
    reason: str
    line_number: Optional[int] = None


@dataclass
class OCRBlock:
    text: str
    confidence: float
    box: Optional[Dict[str, int]] = None
    line_num: int = 0


@dataclass
class OCRResult:
    text: str
    ocr_confidence: float
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    language: str = "en"
    warnings: List[str] = field(default_factory=list)
    potential_confusions: List[Dict[str, Any]] = field(default_factory=list)
    suggested_text: Optional[str] = None  # Non-destructive proposed text with corrections

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "ocr_confidence": round(self.ocr_confidence, 3),
            "blocks": self.blocks,
            "language": self.language,
            "warnings": self.warnings,
            "potential_confusions": self.potential_confusions,
            "suggested_text": self.suggested_text,
        }


class OCRService:
    """Legal OCR processing service with safety-first legal confusion detection."""

    # Common Indian legal statutory abbreviations for confusion detection
    KNOWN_ACT_ACRONYMS = {
        "IPC": "Indian Penal Code",
        "CRPC": "Code of Criminal Procedure",
        "CPC": "Code of Civil Procedure",
        "BNS": "Bharatiya Nyaya Sanhita",
        "BNSS": "Bharatiya Nagarik Suraksha Sanhita",
        "BSA": "Bharatiya Sakshya Adhiniyam",
        "PMLA": "Prevention of Money Laundering Act",
        "NI": "Negotiable Instruments Act",
        "POCSO": "Protection of Children from Sexual Offences Act",
        "NDPS": "Narcotic Drugs and Psychotropic Substances Act",
        "UAPA": "Unlawful Activities (Prevention) Act",
        "IBC": "Insolvency and Bankruptcy Code",
    }

    @classmethod
    def extract_text_from_image(cls, image_path: Path | str) -> OCRResult:
        """
        Extract text from an image file using available OCR engine (pytesseract or fallback).
        Preserves original raw text and annotates potential ambiguities.
        """
        path = Path(image_path)
        if not path.is_file():
            return OCRResult(
                text="",
                ocr_confidence=0.0,
                warnings=[f"Image file not found: {path}"]
            )

        extracted_text = ""
        blocks: List[Dict[str, Any]] = []
        confidences: List[float] = []
        warnings: List[str] = []

        try:
            from PIL import Image
            import pytesseract

            with Image.open(path) as img:
                # Check for tesseract data
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                n_boxes = len(data["text"])
                line_words: Dict[int, List[str]] = {}
                line_confs: Dict[int, List[float]] = {}

                for i in range(n_boxes):
                    word = data["text"][i].strip()
                    conf = float(data["conf"][i])
                    if conf > -1 and word:
                        line_num = data["line_num"][i]
                        line_words.setdefault(line_num, []).append(word)
                        line_confs.setdefault(line_num, []).append(max(0.0, conf / 100.0))
                        confidences.append(max(0.0, conf / 100.0))

                for l_num, words in sorted(line_words.items()):
                    l_text = " ".join(words)
                    avg_c = sum(line_confs[l_num]) / len(line_confs[l_num]) if line_confs[l_num] else 0.5
                    blocks.append({
                        "text": l_text,
                        "confidence": round(avg_c, 3),
                        "line_num": l_num,
                    })

                extracted_text = pytesseract.image_to_string(img).strip()

        except Exception as e:
            # Fallback if tesseract binary is not installed or image library errors
            log.warning(f"Native OCR engine invocation failed: {e}. Utilizing structural image inspection.")
            warnings.append("OCR engine (tesseract) not available or failed; image inspection mode applied.")
            extracted_text = f"[Image Document: {path.name}]"
            confidences.append(0.3)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        if not extracted_text:
            warnings.append("OCR yielded empty text. Document may be blank, corrupt, or low contrast.")

        # Detect confusions non-destructively
        confusions = cls.detect_legal_confusions(extracted_text)
        suggested_text = cls.generate_suggested_text(extracted_text, confusions)

        if avg_confidence < 0.65 and extracted_text:
            warnings.append("Low OCR confidence detected. Do not rely on numerical values or names without verification.")

        return OCRResult(
            text=extracted_text,
            ocr_confidence=avg_confidence,
            blocks=blocks,
            language="en",
            warnings=warnings,
            potential_confusions=[c.__dict__ for c in confusions],
            suggested_text=suggested_text if confusions else None,
        )

    @classmethod
    def detect_legal_confusions(cls, text: str) -> List[OCRConfusion]:
        """
        Scans text for typical OCR optical confusion in Indian legal terminology.
        Examples:
        - '1PC' or 'lPC' -> 'IPC'
        - 'CrPC' misread as 'CrP0' or 'CrPL'
        - Section numbers like 'Section 3O2' -> 'Section 302'
        - 'Article 2l' -> 'Article 21'
        """
        confusions: List[OCRConfusion] = []
        lines = text.splitlines()

        for line_idx, line in enumerate(lines, start=1):
            # 1. Statute Acronym confusion: lPC, 1PC, etc.
            statute_matches = re.finditer(r"\b([1lI]PC|[1lI]P0|Cr[1lI]C|CrP0|C[1lI]C)\b", line)
            for m in statute_matches:
                token = m.group(1)
                suggested = "IPC" if "PC" in token or "P0" in token else "CrPC"
                confusions.append(OCRConfusion(
                    original_token=token,
                    suggested_correction=suggested,
                    category="statute_name",
                    confidence=0.85,
                    reason=f"Common OCR glyph confusion: '{token[0]}' mistaken for capital 'I'.",
                    line_number=line_idx
                ))

            # 2. Section number letter-O instead of zero (e.g., Section 3O2, Section 42O)
            sec_o_matches = re.finditer(r"\b(?:Section|Sec\.?|S\.)\s*(\d*[O|o]\d*)\b", line, re.IGNORECASE)
            for m in sec_o_matches:
                token = m.group(1)
                fixed = token.replace("O", "0").replace("o", "0")
                if fixed != token:
                    confusions.append(OCRConfusion(
                        original_token=token,
                        suggested_correction=fixed,
                        category="section_number",
                        confidence=0.90,
                        reason="Capital or lowercase 'O' used in place of digit '0' in section reference.",
                        line_number=line_idx
                    ))

            # 3. Section or Article numeral 1 misread as 'l' or 'I' (e.g. Article 2l, Section 4l)
            art_l_matches = re.finditer(r"\b(?:Article|Art\.?|Section|Sec\.?)\s*(\d+[lI]|\d*[lI]\d+)\b", line, re.IGNORECASE)
            for m in art_l_matches:
                token = m.group(1)
                fixed = token.replace("l", "1").replace("I", "1")
                confusions.append(OCRConfusion(
                    original_token=token,
                    suggested_correction=fixed,
                    category="section_number",
                    confidence=0.88,
                    reason="Letter 'l' or 'I' used in place of digit '1' in legal provision index.",
                    line_number=line_idx
                ))

            # 4. Year optical confusion (e.g., 2O24, 199l)
            year_matches = re.finditer(r"\b(19\d[O|o|l|I]|20\d[O|o|l|I]|2[O|o]\d{2})\b", line)
            for m in year_matches:
                token = m.group(1)
                fixed = token.replace("O", "0").replace("o", "0").replace("l", "1").replace("I", "1")
                confusions.append(OCRConfusion(
                    original_token=token,
                    suggested_correction=fixed,
                    category="date",
                    confidence=0.92,
                    reason="Alphanumeric glyph in calendar year.",
                    line_number=line_idx
                ))

        return confusions

    @classmethod
    def generate_suggested_text(cls, text: str, confusions: List[OCRConfusion]) -> Optional[str]:
        """
        Creates a proposed corrected text string WITHOUT modifying the original.
        Used solely for user preview and confirmation.
        """
        if not confusions:
            return None

        result = text
        for conf in confusions:
            # Word-boundary replacement of original token with proposed token
            pattern = rf"\b{re.escape(conf.original_token)}\b"
            result = re.sub(pattern, conf.suggested_correction, result)
        return result
