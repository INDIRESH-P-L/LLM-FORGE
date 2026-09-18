"""
app/api/audio.py
================
FastAPI router for Text-to-Speech (TTS) audio and streaming endpoints.
"""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel, Field

from app.services.tts_service import TTSService, TTSServiceError

log = logging.getLogger("legalmind.api.audio")

router = APIRouter(tags=["audio"])


class TTSRequest(BaseModel):
    text: str
    language: str = "en"


class TTSResponse(BaseModel):
    success: bool
    audio_id: str
    language: str = "en"
    duration_seconds: float
    word_count: int
    warnings: List[str] = []
    status: str = "ready"


@router.post("/text-to-speech", response_model=TTSResponse)
@router.post("/api/text-to-speech", response_model=TTSResponse)
async def text_to_speech_endpoint(req: TTSRequest):
    """
    Synthesizes speech metadata for legal analysis answers.
    """
    try:
        res = TTSService.synthesize_speech(text=req.text, language=req.language)
        return TTSResponse(**res)
    except TTSServiceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        log.error(f"TTS synthesis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text-to-speech error: {e}")


@router.get("/audio/stream")
@router.get("/api/audio/stream")
async def audio_stream_get_endpoint(
    text: str = Query(..., description="Text to synthesize into speech audio"),
    language: str = Query("en", description="Spoken language code"),
):
    """
    Streams MP3 audio for HTML5 audio playback across all browsers.
    Bypasses Linux Chromium speech-dispatcher failures.
    """
    try:
        audio_bytes = TTSService.generate_audio_bytes(text=text, language=language)
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
        log.error(f"Audio streaming error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio stream error: {e}")


@router.post("/audio/stream")
@router.post("/api/audio/stream")
async def audio_stream_post_endpoint(req: TTSRequest):
    """
    Streams MP3 audio via POST for longer legal answers.
    """
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
        log.error(f"Audio streaming error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio stream error: {e}")
