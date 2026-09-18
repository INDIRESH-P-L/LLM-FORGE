"""
app/services/__init__.py
========================
Modular multimodal services package for LegalMindAI.
"""

from app.services.ocr_service import OCRService, OCRResult
from app.services.legal_extraction_service import LegalExtractionService, LegalExtractionResult
from app.services.multimodal_context import MultimodalContextManager, DocumentContext
from app.services.vision_service import VisionService, VisionServiceError
from app.services.document_service import DocumentService, DocumentServiceError
from app.services.speech_service import SpeechService, SpeechServiceError
from app.services.tts_service import TTSService, TTSServiceError

__all__ = [
    "OCRService",
    "OCRResult",
    "LegalExtractionService",
    "LegalExtractionResult",
    "MultimodalContextManager",
    "DocumentContext",
    "VisionService",
    "VisionServiceError",
    "DocumentService",
    "DocumentServiceError",
    "SpeechService",
    "SpeechServiceError",
    "TTSService",
    "TTSServiceError",
]
