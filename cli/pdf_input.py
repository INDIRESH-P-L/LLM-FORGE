"""Document & PDF analysis module for LegalMindAI CLI."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cli.config import config
from cli.validators import validate_document_file, ValidationError


class DocumentParser:
    """Extracts text and metadata from PDF, DOCX, and TXT files."""

    @staticmethod
    def extract_text(file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract plain text and basic metadata from a document file."""
        path, ext, page_count = validate_document_file(file_path)
        metadata: Dict[str, Any] = {
            "file_name": path.name,
            "extension": ext,
            "file_size_kb": round(path.stat().st_size / 1024, 1),
            "page_count": page_count,
        }

        text = ""
        if ext == ".pdf":
            text = DocumentParser._extract_pdf(path)
        elif ext == ".docx":
            text = DocumentParser._extract_docx(path)
        elif ext == ".txt":
            text = DocumentParser._extract_txt(path)
        elif ext in config.supported_image_exts:
            from cli.image_input import ImageAnalyzer
            text, img_meta = ImageAnalyzer.extract_text(str(path))
            metadata.update(img_meta)

        if not text.strip():
            raise ValidationError(f"[ERROR] Could not extract any readable text from {path.name}")

        metadata["char_count"] = len(text)
        metadata["word_count"] = len(text.split())
        return text, metadata

    @staticmethod
    def _extract_pdf(path: Path) -> str:
        """Extract text from PDF using PyMuPDF (fitz) with pdfplumber fallback."""
        try:
            import fitz
            doc = fitz.open(path)
            pages = []
            for page in doc:
                pages.append(page.get_text() or "")
            doc.close()
            return "\n\n".join(pages)
        except Exception:
            pass

        # Fallback to pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
                return "\n\n".join(pages)
        except Exception:
            pass

        # Fallback to pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n\n".join(pages)
        except Exception as e:
            raise ValidationError(f"[ERROR] Failed to extract text from PDF {path.name}: {e}")

    @staticmethod
    def _extract_docx(path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            import docx
            doc = docx.Document(path)
            return "\n".join([p.text for p in doc.paragraphs if p.text])
        except Exception:
            # Fallback: unzip and extract word/document.xml
            try:
                import zipfile
                import xml.etree.ElementTree as ET
                with zipfile.ZipFile(path) as z:
                    xml_content = z.read("word/document.xml")
                    tree = ET.fromstring(xml_content)
                    texts = []
                    for elem in tree.iter():
                        if elem.tag.endswith("t") and elem.text:
                            texts.append(elem.text)
                    return " ".join(texts)
            except Exception as e:
                raise ValidationError(f"[ERROR] Failed to extract text from DOCX {path.name}: {e}")

    @staticmethod
    def _extract_txt(path: Path) -> str:
        """Read standard text file."""
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise ValidationError(f"[ERROR] Failed to read text file {path.name}: {e}")


class LegalDocumentAnalyzer:
    """Analyzes extracted text for Indian legal entities, structure, and judgment ratios."""

    @staticmethod
    def detect_document_type(text: str) -> str:
        """Heuristic detection of legal document genre."""
        lower = text.lower()
        if any(term in lower for term in ["supreme court of india", "in the high court", "writ petition", "appellant", "respondent"]):
            return "Court Judgment / Order"
        elif any(term in lower for term in ["legal notice", "statutory notice", "advocate notice", "under section 138"]):
            return "Legal Notice"
        elif any(term in lower for term in ["first information report", "police station", "fir no"]):
            return "FIR (First Information Report)"
        elif any(term in lower for term in ["act, 19", "act, 20", "be it enacted", "short title and commencement"]):
            return "Statute / Legislative Act"
        elif any(term in lower for term in ["agreement", "contract", "memorandum of understanding", "whereas the parties"]):
            return "Agreement / Legal Contract"
        return "Legal Document"

    @staticmethod
    def extract_legal_entities(text: str) -> Dict[str, Any]:
        """Extract key legal metadata such as courts, case numbers, acts, and dates."""
        entities: Dict[str, Any] = {
            "court": None,
            "case_number": None,
            "dates": [],
            "acts_referenced": [],
            "sections_referenced": [],
        }

        # Court detection
        if re.search(r"supreme\s+court\s+of\s+india", text, re.I):
            entities["court"] = "Supreme Court of India"
        else:
            hc_match = re.search(r"high\s+court\s+of\s+([A-Za-z\s]+)", text, re.I)
            if hc_match:
                entities["court"] = f"High Court of {hc_match.group(1).strip()}"

        # Case number pattern
        case_match = re.search(
            r"(writ\s+petition|civil\s+appeal|criminal\s+appeal|special\s+leave\s+petition|slp)\s*(?:no\.?|\(?c\)?\s*no\.?)?\s*[\d/]+(?:\s*of\s*\d{4})?",
            text, re.I
        )
        if case_match:
            entities["case_number"] = case_match.group(0).strip()

        # Dates (DD/MM/YYYY, Month DD, YYYY)
        dates = re.findall(r"\b\d{1,2}[-/.](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2})[-/.]\d{2,4}\b", text, re.I)
        entities["dates"] = list(dict.fromkeys(dates))[:5]

        # Acts referenced
        acts = re.findall(r"(?:Constitution\s+of\s+India|[A-Z][A-Za-z\s]+Act,\s*\d{4}|Indian\s+Penal\s+Code|Code\s+of\s+Criminal\s+Procedure|Bharatiya\s+Nyaya\s+Sanhita)", text)
        entities["acts_referenced"] = list(dict.fromkeys(acts))[:6]

        # Sections referenced
        sections = re.findall(r"(?:Section|Sec\.|Article|Art\.)\s*\d+[A-Za-z]*(?:\(\d+\))?", text, re.I)
        entities["sections_referenced"] = list(dict.fromkeys(sections))[:10]

        return entities

    @staticmethod
    def build_analysis_prompt(file_name: str, doc_type: str, text: str, user_question: Optional[str] = None) -> str:
        """Construct the prompt sent to the backend to generate a structured analysis."""
        sample_text = text[:8000]  # Cap to prevent model prompt overflow
        
        if user_question:
            return (
                f"You are LegalMindAI, an expert Indian legal assistant. "
                f"Analyze the following {doc_type} titled '{file_name}'.\n\n"
                f"--- DOCUMENT EXCERPT ---\n"
                f"{sample_text}\n"
                f"--- END EXCERPT ---\n\n"
                f"Answer the user's specific inquiry thoroughly with verified legal principles and references:\n"
                f"Question: {user_question}"
            )

        # Default structured breakdown for judgments / documents
        if "judgment" in doc_type.lower():
            return (
                f"You are LegalMindAI, an expert Indian legal assistant. "
                f"Provide a comprehensive, authoritative legal analysis of the following judgment ({file_name}).\n\n"
                f"--- DOCUMENT TEXT ---\n"
                f"{sample_text}\n"
                f"--- END DOCUMENT TEXT ---\n\n"
                f"Organize your answer strictly into the following 13 structured sections:\n"
                f"1. Case Name & Parties\n"
                f"2. Court & Bench\n"
                f"3. Date / Year\n"
                f"4. Material Facts\n"
                f"5. Legal Issues\n"
                f"6. Arguments of the Parties\n"
                f"7. Relevant Provisions (Constitution, Statutes, Sections)\n"
                f"8. Court Decision / Order\n"
                f"9. Ratio Decidendi & Legal Principles\n"
                f"10. Legal Significance & Precedential Value\n"
                f"11. Related / Referred Judgments\n"
                f"12. Verified Sources\n"
                f"13. Limitations & Disclaimers"
            )

        return (
            f"You are LegalMindAI, an expert Indian legal assistant. "
            f"Analyze the following {doc_type} titled '{file_name}'.\n\n"
            f"--- DOCUMENT TEXT ---\n"
            f"{sample_text}\n"
            f"--- END DOCUMENT TEXT ---\n\n"
            f"Provide a comprehensive summary including:\n"
            f"- Document Overview & Purpose\n"
            f"- Key Parties / Authorities\n"
            f"- Core Legal Provisions & Sections Cited\n"
            f"- Important Deadlines, Obligations, or Directives\n"
            f"- Legal Implications & Recommended Next Steps"
        )
