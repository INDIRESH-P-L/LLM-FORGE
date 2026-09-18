"""
app/api/moot.py
===============
FastAPI router for Moot Court & Judicial Adversary Simulator ("Judge Mode").
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.moot_service import MootCourtService

log = logging.getLogger("legalmind.api.moot")

router = APIRouter(tags=["moot"])


class MootInterjectRequest(BaseModel):
    argument: str = Field(..., description="Counsel's oral submission or legal proposition")
    bench_type: str = Field("constitutional", description="Bench type: constitutional, criminal, commercial, regulatory, tax_insolvency")
    temperament: str = Field("inquisitive", description="Judicial temperament: inquisitive, textualist, adversarial")
    round_num: int = Field(1, ge=1, le=20)
    counsel_side: Optional[str] = Field("petitioner", description="Counsel representation side: petitioner, appellant, respondent, state")
    case_topic: Optional[str] = Field(default=None, description="Optional moot problem scenario or case title")
    factual_matrix: Optional[str] = Field(default=None, description="Optional factual background")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Transcript of prior rounds in this session")


class MootExportRequest(BaseModel):
    bench_type: str = "constitutional"
    temperament: str = "inquisitive"
    rounds: List[Dict[str, Any]]
    scorecard: Dict[str, Any]
    case_title: Optional[str] = "Simulated Appellate Hearing"
    counsel_side: Optional[str] = "petitioner"


@router.get("/moot/problems")
@router.get("/api/moot/problems")
async def get_moot_problems():
    """
    Returns the curated catalog of realistic appellate moot court problems with factual matrixes.
    """
    return {
        "success": True,
        "problems": MootCourtService.MOOT_PROBLEMS,
    }


@router.get("/moot/corams")
@router.get("/api/moot/corams")
async def get_moot_corams():
    """
    Returns the metadata and statutory compendiums for the 5 specialized Appellate Corams.
    """
    return {
        "success": True,
        "corams": MootCourtService.JUDGE_PERSONAS,
    }


@router.post("/moot/interject")
@router.post("/api/moot/interject")
async def moot_interject_endpoint(req: MootInterjectRequest):
    """
    Submits counsel's argument to the simulated judicial bench and returns probing questions with scorecard.
    """
    try:
        res = MootCourtService.interject(
            argument=req.argument,
            bench_type=req.bench_type,
            temperament=req.temperament,
            round_num=req.round_num,
            counsel_side=req.counsel_side,
            case_topic=req.case_topic,
            factual_matrix=req.factual_matrix,
            prior_history=req.history,
        )
        return res
    except Exception as e:
        log.error(f"Moot simulator error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Moot court simulation error: {e}")


@router.post("/moot/export")
@router.post("/api/moot/export")
async def moot_export_endpoint(req: MootExportRequest):
    """
    Generates an official Supreme Court Order Sheet / Minutes of Proceeding.
    """
    try:
        record = MootCourtService.generate_session_record(
            bench_type=req.bench_type,
            temperament=req.temperament,
            rounds=req.rounds,
            scorecard=req.scorecard,
            case_title=req.case_title,
            counsel_side=req.counsel_side,
        )
        return {"success": True, "record": record}
    except Exception as e:
        log.error(f"Moot export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Moot export error: {e}")
