"""
app/api/pleading.py
===================
LegalMind AI — Voice Pleading Formatter API
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.pleading_service import PleadingFormatterService

router = APIRouter(prefix="/api/pleading", tags=["Voice Pleading Formatter"])

class PleadingFormatRequest(BaseModel):
    raw_text: str = Field(..., description="Raw unstructured text from dictation")
    pleading_type: str = Field(default="bail", description="e.g. bail, writ, civil_suit")

@router.post("/format")
def format_pleading(req: PleadingFormatRequest):
    if not req.raw_text.strip():
        raise HTTPException(status_code=400, detail="Transcription text cannot be empty.")
        
    try:
        result = PleadingFormatterService.format_pleading(
            raw_text=req.raw_text,
            pleading_type=req.pleading_type
        )
        return {"status": "success", "pleading_draft": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
