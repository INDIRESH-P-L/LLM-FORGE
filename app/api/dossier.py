"""
app/api/dossier.py
==================
FastAPI router for Advocate's Case Dossier & Court Brief Generator endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from app.services.dossier_service import DossierService

log = logging.getLogger("legalmind.api.dossier")

router = APIRouter(tags=["dossier"])


class DossierRequest(BaseModel):
    case_title: str = Field("LEGAL MEMORANDUM FOR ADVOCATE", description="Title of the case or brief")
    query: str = Field(..., description="Legal query or proposition")
    answer: str = Field(..., description="Legal answer or analysis text")
    court: str = Field("IN THE SUPREME COURT OF INDIA", description="Target Court Name")
    citations: List[str] = Field(default_factory=list, description="List of case citations")


@router.post("/dossier/generate")
@router.post("/api/dossier/generate")
async def generate_dossier_endpoint(req: DossierRequest):
    """
    Structures the legal research into a formal advocate's bench memorandum.
    """
    try:
        dossier = DossierService.generate_dossier(
            case_title=req.case_title,
            query=req.query,
            answer=req.answer,
            court=req.court,
            citations=req.citations,
        )
        return {"success": True, "dossier": dossier}
    except Exception as e:
        log.error(f"Dossier generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Dossier generation failed: {e}")


@router.post("/dossier/export-pdf")
@router.post("/api/dossier/export-pdf")
async def export_dossier_pdf_endpoint(req: DossierRequest):
    """
    Generates and returns a downloadable court-ready PDF file of the bench dossier.
    """
    try:
        dossier = DossierService.generate_dossier(
            case_title=req.case_title,
            query=req.query,
            answer=req.answer,
            court=req.court,
            citations=req.citations,
        )
        pdf_bytes = DossierService.generate_pdf(dossier)
        filename = f"Court_Brief_{dossier.get('dossier_id', 'case')}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-cache",
            },
        )
    except Exception as e:
        log.error(f"Dossier PDF export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Dossier PDF generation failed: {e}")
