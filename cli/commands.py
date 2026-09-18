"""Command implementations for LegalMindAI CLI."""

import time
from typing import Any, Dict, List, Optional

from cli.client import BackendError, LegalMindClient
from cli.config import config
from cli.image_input import ImageAnalyzer
from cli.pdf_input import DocumentParser, LegalDocumentAnalyzer
from cli.renderer import (
    console,
    render_answer_box,
    render_document_analysis,
    render_error,
    render_info,
    render_sources_list,
    render_status,
    render_warning,
)
from cli.speech_output import SpeechOutputHandler
from cli.validators import ValidationError, validate_document_file, validate_image_file
from cli.voice_input import VoiceInputHandler


def ask_question(
    client: LegalMindClient,
    query: str,
    mode: Optional[str] = None,
    json_output: bool = False,
    stream: bool = True,
) -> Optional[Dict[str, Any]]:
    """Execute a legal inquiry and render the formatted answer."""
    clean_query = query.strip()
    if not clean_query:
        render_error("[ERROR] Question cannot be empty.")
        return None

    prev_json = config.json_mode
    if json_output:
        config.json_mode = True

    try:
        # Responsive progress status indicator
        with console.status("[bold cyan]Consulting LegalMindAI Indian Legal Corpus...[/bold cyan]", spinner="dots"):
            response = client.chat(clean_query, mode=mode)
    except BackendError as e:
        render_error(str(e))
        return None
    except KeyboardInterrupt:
        console.print()
        render_warning("[WARNING] Answer generation was interrupted.")
        render_info("[INFO] The displayed answer may be incomplete.")
        return None
    except Exception as e:
        render_error(f"[ERROR] Unexpected error during query: {e}")
        return None
    finally:
        config.json_mode = prev_json

    render_answer_box(
        answer=response.get("answer", ""),
        citations=response.get("citations", []),
        confidence=response.get("confidence", "MEDIUM"),
        evidence_coverage=response.get("evidence_coverage", 0.0),
        warnings=response.get("warnings", []),
        latency_ms=response.get("latency_ms", 0.0),
        status=response.get("status", "completed"),
        raw_response=response.get("raw"),
    )
    return response


def analyze_image_command(
    client: LegalMindClient,
    file_path: str,
    question: Optional[str] = None,
    show_ocr: bool = False,
) -> Optional[Dict[str, Any]]:
    """Validate image, extract OCR text, and analyze legal notice/order."""
    try:
        raw_text, meta = ImageAnalyzer.extract_text(file_path)
    except ValidationError as e:
        render_error(str(e))
        return None
    except Exception as e:
        render_error(f"[ERROR] Image validation failed: {e}")
        return None

    if show_ocr or config.debug:
        console.print()
        console.print(f"[bold cyan]Extracted OCR Content ({meta['file_name']}):[/bold cyan]")
        console.print("─" * 60)
        console.print(raw_text)
        console.print("─" * 60)
        console.print()

    if meta.get("ocr_uncertain"):
        render_warning("[WARNING] Some text may have been incorrectly recognized.")
        render_info("[INFO] Please verify the extracted text before relying on it.")

    doc_type = ImageAnalyzer.detect_document_type(raw_text, meta["file_name"])
    prompt = ImageAnalyzer.build_image_prompt(
        file_name=meta["file_name"],
        doc_type=doc_type,
        extracted_text=raw_text,
        user_question=question,
        is_uncertain=meta.get("ocr_uncertain", False),
    )

    # Attach to active session context
    config.active_document_name = meta["file_name"]
    config.active_document_type = doc_type
    config.active_document_text = raw_text

    try:
        with console.status("[bold cyan]Analyzing legal image document...[/bold cyan]", spinner="dots"):
            response = client.chat(prompt)
    except BackendError as e:
        render_error(str(e))
        return None
    except Exception as e:
        render_error(f"[ERROR] Backend analysis failed: {e}")
        return None

    render_document_analysis(
        file_name=meta["file_name"],
        doc_type=doc_type,
        extracted_text=raw_text,
        answer=response.get("answer", ""),
        confidence=response.get("confidence", "MEDIUM"),
        sources=response.get("citations", []),
        warnings=response.get("warnings", []),
        metadata={
            "Dimensions": meta.get("dimensions"),
            "MIME Type": meta.get("mime_type"),
            "OCR Method": meta.get("ocr_method"),
        },
    )
    return response


def analyze_document_command(
    client: LegalMindClient,
    file_path: str,
    question: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Validate and analyze a PDF, DOCX, or text document."""
    try:
        raw_text, meta = DocumentParser.extract_text(file_path)
    except ValidationError as e:
        render_error(str(e))
        return None
    except Exception as e:
        render_error(f"[ERROR] Document parsing failed: {e}")
        return None

    doc_type = LegalDocumentAnalyzer.detect_document_type(raw_text)
    entities = LegalDocumentAnalyzer.extract_legal_entities(raw_text)
    prompt = LegalDocumentAnalyzer.build_analysis_prompt(
        file_name=meta["file_name"],
        doc_type=doc_type,
        text=raw_text,
        user_question=question,
    )

    # Update active document context for follow-up questions
    config.active_document_name = meta["file_name"]
    config.active_document_type = doc_type
    config.active_document_text = raw_text

    try:
        with console.status("[bold cyan]Analyzing legal document & extracting statutory provisions...[/bold cyan]", spinner="dots"):
            response = client.chat(prompt)
    except BackendError as e:
        render_error(str(e))
        return None
    except Exception as e:
        render_error(f"[ERROR] Analysis failed: {e}")
        return None

    # Merge extracted entities into metadata
    meta_display = {
        "Pages": str(meta.get("page_count") or "N/A"),
        "Court": entities.get("court"),
        "Case Number": entities.get("case_number"),
        "Acts Cited": ", ".join(entities.get("acts_referenced", [])[:3]),
    }

    render_document_analysis(
        file_name=meta["file_name"],
        doc_type=doc_type,
        extracted_text=raw_text,
        answer=response.get("answer", ""),
        confidence=response.get("confidence", "HIGH" if "judgment" in doc_type.lower() else "MEDIUM"),
        sources=response.get("citations", []),
        warnings=response.get("warnings", []),
        metadata=meta_display,
    )
    return response


def show_sources_command(client: LegalMindClient) -> None:
    """Show sources from the latest answer."""
    render_sources_list(config.last_citations)


def show_status_command(client: LegalMindClient) -> None:
    """Check backend health and display system status."""
    health = client.check_health()
    stats = client.get_stats()
    render_status(
        client_health=health,
        stats=stats,
        voice_available=VoiceInputHandler.is_available(),
        ocr_available=True,
        tts_available=SpeechOutputHandler.is_available(),
    )


def voice_command(client: LegalMindClient, language: str = "en") -> None:
    """Execute voice typing flow."""
    transcript = VoiceInputHandler.capture_and_transcribe(language=language)
    if transcript:
        ask_question(client, transcript)


def speak_command(summary: bool = False) -> None:
    """Read the latest answer aloud."""
    if not config.last_response or not config.last_response.get("answer"):
        render_warning("[WARNING] No answer available to read aloud.")
        return
    SpeechOutputHandler.speak_text(config.last_response["answer"], summary_only=summary)
