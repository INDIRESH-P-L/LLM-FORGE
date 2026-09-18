"""
app/api/speech.py
=================
FastAPI router for voice typing & speech-to-text endpoints.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Response
from pydantic import BaseModel

from app.services.speech_service import SpeechService, SpeechServiceError
from app.services.tts_service import TTSService, TTSServiceError

log = logging.getLogger("legalmind.api.speech")

router = APIRouter(tags=["speech"])


class SpeechResponse(BaseModel):
    success: bool
    text: str
    language: str = "en"
    confidence: float
    warnings: list[str] = []


class SpeakRequest(BaseModel):
    text: str
    language: str = "en"


@router.post("/speech/speak")
@router.post("/api/speech/speak")
async def speech_speak_endpoint(req: SpeakRequest):
    """
    Synthesizes and streams spoken audio (MP3) for Moot Court judicial dialogue
    and voice interactions.
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        audio_bytes = TTSService.generate_audio_bytes(text=req.text, language=req.language)
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=speech.mp3",
                "Cache-Control": "public, max-age=3600",
            },
        )
    except TTSServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        log.error(f"Speech synthesis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Speech synthesis error: {e}")


@router.post("/speech-to-text", response_model=SpeechResponse)
@router.post("/api/speech-to-text", response_model=SpeechResponse)
async def speech_to_text_endpoint(
    file: UploadFile = File(...),
    language: str = Form("en"),
    conversation_id: Optional[str] = Form(None),
):
    """
    Transcribes uploaded audio into editable legal query text.
    Does not automatically submit the query to the language model.
    """
    filename = file.filename or "audio_input.wav"
    try:
        content = await file.read()
        res = SpeechService.transcribe_audio(
            filename=filename,
            content=content,
            language=language,
            conversation_id=conversation_id,
        )
        return SpeechResponse(**res)
    except SpeechServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        log.error(f"Speech-to-text failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Speech transcription error: {e}")

