"""
app/services/tts_service.py
===========================
Text-to-Speech (TTS) Service for LegalMindAI.

Features:
- Validates input text (non-empty, maximum character limits).
- Audio stream generation via gTTS with Indian-English cadence ("co.in").
- In-memory LRU-style cache for instant audio replay without repeated synthesis.
- Natural markdown-to-speech cleaning (strips symbols, headers, citations).
- Preserves full textual citations and disclaimers in the main application.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

log = logging.getLogger("legalmind.tts")

MAX_TTS_CHARS = 4000
_AUDIO_CACHE: Dict[str, bytes] = {}
_MAX_CACHE_ENTRIES = 50


class TTSServiceError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class TTSService:
    """Provides speech synthesis and audio streaming for legal answers."""

    @classmethod
    def clean_for_speech(cls, text: str) -> str:
        """Cleans markdown symbols and technical syntax into smooth spoken prose."""
        if not text:
            return ""

        # Remove code blocks
        t = re.sub(r"```[\s\S]*?```", " Code omitted for audio. ", text)
        t = re.sub(r"`([^`]+)`", r"\1", t)

        # Remove markdown headers (### Header -> Header)
        t = re.sub(r"^[#]+\s*", "", t, flags=re.MULTILINE)

        # Remove markdown links [Label](url) -> Label
        t = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", t)

        # Remove bold / italic markers
        t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
        t = re.sub(r"\*([^*]+)\*", r"\1", t)
        t = re.sub(r"__([^_]+)__", r"\1", t)

        # Remove bullet points and numbered list markers
        t = re.sub(r"^[\*\-•]\s+", "", t, flags=re.MULTILINE)
        t = re.sub(r"^\d+\.\s+", "", t, flags=re.MULTILINE)

        # Clean multiple spaces and blank lines
        t = re.sub(r"\n{2,}", ". ", t)
        t = re.sub(r"\n", " ", t)
        t = re.sub(r"\s{2,}", " ", t)

        return t.strip()

    @classmethod
    def generate_audio_bytes(
        cls,
        text: str,
        language: str = "en",
    ) -> bytes:
        """
        Generates MP3 audio bytes for playback via HTML5 Audio elements.
        Uses in-memory cache to ensure instant playback on replay.
        """
        spoken_text = cls.clean_for_speech(text)
        if not spoken_text:
            raise TTSServiceError("Text for speech synthesis cannot be empty.", status_code=400)

        if len(spoken_text) > MAX_TTS_CHARS:
            spoken_text = spoken_text[:MAX_TTS_CHARS] + "... (Audio truncated for length)."

        cache_key = hashlib.md5(f"{language}:{spoken_text}".encode("utf-8")).hexdigest()
        if cache_key in _AUDIO_CACHE:
            log.info(f"Serving cached TTS audio ({len(_AUDIO_CACHE[cache_key])} bytes)")
            return _AUDIO_CACHE[cache_key]

        # Determine language and accent
        lang_code = "en"
        tld = "co.in"
        if language in ("hi", "hindi"):
            lang_code = "hi"
            tld = "co.in"
        elif language in ("ta", "tamil"):
            lang_code = "ta"
            tld = "co.in"

        audio_bytes = None

        # 1. Primary: gTTS
        try:
            from gtts import gTTS
            tts = gTTS(text=spoken_text, lang=lang_code, tld=tld, slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            audio_bytes = buf.getvalue()
        except Exception as e:
            log.warning(f"gTTS audio synthesis notice: {e}")

        # 2. Secondary: pyttsx3 fallback if available
        if not audio_bytes:
            try:
                import pyttsx3
                import tempfile
                import os
                engine = pyttsx3.init()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    wav_path = tf.name
                engine.save_to_file(spoken_text, wav_path)
                engine.runAndWait()
                with open(wav_path, "rb") as rf:
                    audio_bytes = rf.read()
                os.remove(wav_path)
            except Exception as e2:
                log.warning(f"pyttsx3 audio synthesis notice: {e2}")

        if not audio_bytes:
            raise TTSServiceError("Could not generate audio stream with current TTS engines.", status_code=500)

        # Store in cache
        if len(_AUDIO_CACHE) >= _MAX_CACHE_ENTRIES:
            # Drop oldest key
            first_key = next(iter(_AUDIO_CACHE))
            del _AUDIO_CACHE[first_key]

        _AUDIO_CACHE[cache_key] = audio_bytes
        return audio_bytes

    @classmethod
    def synthesize_speech(
        cls,
        text: str,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Synthesizes audio from text or returns a structured readiness object.
        """
        clean_text = text.strip()
        if not clean_text:
            raise TTSServiceError("Text for speech synthesis cannot be empty.", status_code=400)

        if len(clean_text) > MAX_TTS_CHARS:
            clean_text = clean_text[:MAX_TTS_CHARS]

        audio_id = f"audio_{uuid.uuid4().hex[:10]}"
        warnings: List[str] = []

        word_count = len(clean_text.split())
        est_duration = max(1.0, round((word_count / 150.0) * 60.0, 1))

        # Check for synthesis availability
        engine_available = False
        try:
            from gtts import gTTS
            engine_available = True
        except Exception:
            try:
                import pyttsx3
                engine_available = True
            except Exception:
                engine_available = False

        if not engine_available:
            warnings.append("Local TTS engine is not active. Using client-side Web Speech synthesis.")

        return {
            "success": True,
            "audio_id": audio_id,
            "language": language,
            "duration_seconds": est_duration,
            "word_count": word_count,
            "warnings": warnings,
            "status": "ready" if engine_available else "client_synthesis_recommended",
        }
