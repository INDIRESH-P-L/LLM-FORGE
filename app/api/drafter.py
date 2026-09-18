"""
app/api/drafter.py
==================
FastAPI router for AI Legal Drafting & Contract Redlining endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.drafter_service import DrafterService

log = logging.getLogger("legalmind.api.drafter")

router = APIRouter(tags=["drafter"])


class DraftRequest(BaseModel):
    document_type: str = Field(..., description="Document type: legal_notice_138_ni, bail_application, commercial_nda, etc.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Custom parameters for the drafting template")


class ReviewRequest(BaseModel):
    contract_text: str = Field(..., description="Text of the contract or agreement to analyze")


@router.post("/drafter/generate")
@router.post("/api/drafter/generate")
async def generate_draft_endpoint(req: DraftRequest):
    """
    Generates tailored Indian legal notices, petitions, or commercial agreements.
    """
    try:
        res = DrafterService.draft_document(doc_type=req.document_type, params=req.parameters)
        return {"success": True, **res}
    except Exception as e:
        log.error(f"Drafting error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Legal drafting failed: {e}")


@router.post("/drafter/review")
@router.post("/api/drafter/review")
async def review_contract_endpoint(req: ReviewRequest):
    """
    Scans contracts for 8 major Indian statutory risk categories and generates redline recommendations.
    """
    try:
        res = DrafterService.review_contract(contract_text=req.contract_text)
        return res
    except Exception as e:
        log.error(f"Contract review error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Contract review failed: {e}")
