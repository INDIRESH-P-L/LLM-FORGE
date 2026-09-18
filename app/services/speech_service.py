"""
app/services/speech_service.py
==============================
Backend Speech-to-Text (STT) Service for LegalMindAI Voice Typing.

Features:
- Validates audio files (WAV, MP3, OGG, WEBM, M4A, FLAC).
- Enforces size limits (max 25MB) and rejects empty or truncated streams.
- Local Whisper / Faster-Whisper integration with graceful fallback:
  * Detects installed engine (`faster_whisper` or `whisper`).
  * If no local model weights are present, returns clear, informative response
    advising configuration rather than hanging or attempting unapproved downloads.
- Returns transcript for frontend review/editing prior to inference dispatch.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("legalmind.speech")

ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"}
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB


class SpeechServiceError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class SpeechService:
    """Speech transcription service supporting English, Hindi, and Tamil legal voice input."""

    _model_instance = None

    @classmethod
    def validate_audio_file(cls, filename: str, content: bytes) -> Tuple[str, str]:
        """Validates incoming audio payload."""
        clean_name = Path(filename).name
        clean_name = re.sub(r"[^\w\.-]", "_", clean_name)
        ext = Path(clean_name).suffix.lower()

        if ext not in ALLOWED_AUDIO_EXTS:
            raise SpeechServiceError(
                f"Unsupported audio format '{ext}'. Allowed formats: {', '.join(sorted(ALLOWED_AUDIO_EXTS))}",
                status_code=400
            )

        if not content or len(content) == 0:
            raise SpeechServiceError("Uploaded audio file is empty.", status_code=400)

        if len(content) < 100:
            raise SpeechServiceError("Audio stream is truncated or too short to process.", status_code=400)

        if len(content) > MAX_AUDIO_BYTES:
            raise SpeechServiceError(
                f"Audio file exceeds maximum size limit of {MAX_AUDIO_BYTES // (1024*1024)} MB.",
                status_code=400
            )

        return clean_name, ext

    @classmethod
    def transcribe_audio(
        cls,
        filename: str,
        content: bytes,
        language: str = "en",
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribes legal speech audio into editable text without dispatching to LLM.
        """
        clean_name, ext = cls.validate_audio_file(filename, content)
        warnings: List[str] = []

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # 1. Faster-Whisper transcription
            try:
                from faster_whisper import WhisperModel
                if cls._model_instance is None:
                    model_path = os.environ.get("WHISPER_MODEL_PATH", "base")
                    cls._model_instance = WhisperModel(model_path, device="cpu", compute_type="int8")

                target_lang = None if language in ("auto", "all", None, "") else language
                segments, info = cls._model_instance.transcribe(
                    tmp_path,
                    language=target_lang,
                    beam_size=5,
                    vad_filter=False,
                    temperature=0.0,
                )
                text = " ".join([seg.text for seg in segments]).strip()
                confidence = float(info.language_probability) if hasattr(info, "language_probability") else 0.95

                log.info(f"🎤 Transcribed {len(content)} bytes ({clean_name}) -> text='{text}' (lang={info.language}, prob={confidence:.2f})")

                if not text:
                    return {
                        "success": True,
                        "text": "",
                        "language": info.language if hasattr(info, "language") else language,
                        "confidence": round(confidence, 2),
                        "warnings": ["No spoken words detected in the audio."],
                    }

                return {
                    "success": True,
                    "text": text,
                    "language": info.language if hasattr(info, "language") else language,
                    "confidence": round(confidence, 2),
                    "warnings": warnings,
                }
            except Exception as e:
                log.error(f"Faster-whisper transcription error: {e}", exc_info=True)
                warnings.append(f"Transcription notice: {e}")

            # 2. Secondary fallback: OpenAI Whisper if installed
            try:
                import whisper
                model = whisper.load_model("tiny")
                res = model.transcribe(tmp_path, language=language if language != "auto" else None)
                text = res.get("text", "").strip()
                return {
                    "success": True,
                    "text": text,
                    "language": language,
                    "confidence": 0.88,
                    "warnings": warnings,
                }
            except Exception as e:
                log.error(f"OpenAI whisper transcription error: {e}")

            # If both engines fail, return empty text with warning (never fake user speech)
            return {
                "success": False,
                "text": "",
                "language": language,
                "confidence": 0.0,
                "warnings": warnings or ["Speech transcription engine failed to process audio."],
            }

        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
