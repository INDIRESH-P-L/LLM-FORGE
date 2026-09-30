"""
app/api/dossier.py
==================
FastAPI router for Advocate's Case Dossier & Court Brief Generator endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response
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
    brief_text: Optional[str] = Field(None, description="Pre-synthesized brief text if available")


DUMMY_VALUES = {
    "", "cause title", "legal question", "factual matrix",
    "legal memorandum for advocate", "n/a", "none", "null", "undefined", "brief title"
}
LEGACY_DUMMY_CITATIONS = {"(2024) INSC 595", "(2022) 10 SCC 51", "INSC 595", "10 SCC 51"}


def _validate_and_sanitize(req: DossierRequest) -> tuple[str, str, str, str, List[str]]:
    title = (req.case_title or "").strip()
    query = (req.query or "").strip()
    answer = (req.answer or "").strip()
    court = (req.court or "").strip() or "IN THE SUPREME COURT OF INDIA"

    if not title or title.lower() in DUMMY_VALUES:
        raise HTTPException(
            status_code=400,
            detail="Cannot compile dossier: Please enter a genuine Case Name / Cause Title.",
        )
    if not query or query.lower() in DUMMY_VALUES:
        raise HTTPException(
            status_code=400,
            detail="Cannot compile dossier: Please enter a genuine Core Legal Proposition / Question of Law.",
        )
    if not answer or answer.lower() in DUMMY_VALUES:
        raise HTTPException(
            status_code=400,
            detail="Cannot compile dossier: Please enter the genuine Factual Matrix / Submissions.",
        )

    # Discard legacy dummy citations that were injected in older builds
    clean_citations = [
        c.strip() for c in (req.citations or [])
        if c and c.strip() and c.strip() not in LEGACY_DUMMY_CITATIONS
    ]
    return title, query, answer, court, clean_citations


@router.post("/dossier/generate")
@router.post("/api/dossier/generate")
async def generate_dossier_endpoint(req: DossierRequest, request: Request):
    """
    Structures the legal research into a formal advocate's bench memorandum.
    Uses LLM analysis to synthesize a brief report rather than echoing raw input.
    """
    title, query, answer, court, clean_cites = _validate_and_sanitize(req)
    model = getattr(request.app.state, "model", None)
    try:
        dossier = DossierService.generate_dossier(
            case_title=title,
            query=query,
            answer=answer,
            court=court,
            citations=clean_cites,
            model=model,
            brief_text=req.brief_text,
        )
        return {"success": True, "dossier": dossier}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        log.error(f"Dossier generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Dossier generation failed: {e}")


@router.post("/dossier/export-pdf")
@router.post("/api/dossier/export-pdf")
async def export_dossier_pdf_endpoint(req: DossierRequest, request: Request):
    """
    Generates and returns a downloadable court-ready PDF file of the bench dossier.
    """
    title, query, answer, court, clean_cites = _validate_and_sanitize(req)
    model = getattr(request.app.state, "model", None)
    try:
        dossier = DossierService.generate_dossier(
            case_title=title,
            query=query,
            answer=answer,
            court=court,
            citations=clean_cites,
            model=model,
            brief_text=req.brief_text,
        )
        pdf_bytes = DossierService.generate_pdf(dossier)
        filename = f"Court_Brief_{dossier.get('dossier_id', 'case')}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-cache, no-store, must-revalidate",
            },
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        log.error(f"Dossier PDF export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Dossier PDF generation failed: {e}")

