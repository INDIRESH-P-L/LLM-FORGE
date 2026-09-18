"""
app/api/bail.py
===============
LegalMind AI — Bail Feasibility Matrix API
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from app.services.bail_service import BailFeasibilityService

router = APIRouter(prefix="/api/bail", tags=["Bail Matrix"])

class BailAssessmentRequest(BaseModel):
    offense_category: str = Field(default="Economic Offence")
    is_special_act: bool = Field(default=False)
    custody_days: int = Field(default=0)
    chargesheet_filed: bool = Field(default=False)
    # Optional: the UI does not collect these. Previously they defaulted to
    # "Clean" and 7 years, which silently added a clean-antecedents bonus to
    # every request and treated every offence (even murder) as a <=7-year one.
    antecedents: Optional[str] = Field(default=None)
    co_accused_status: Optional[str] = Field(default=None)
    max_punishment_years: Optional[int] = Field(default=None, description="Overrides the offence category's default maximum")

@router.post("/assess")
def assess_bail(req: BailAssessmentRequest):
    try:
        result = BailFeasibilityService.assess_bail(
            offense_category=req.offense_category,
            is_special_act=req.is_special_act,
            custody_days=req.custody_days,
            chargesheet_filed=req.chargesheet_filed,
            antecedents=req.antecedents,
            co_accused_status=req.co_accused_status,
            max_punishment_years=req.max_punishment_years
        )
        return {"status": "success", "assessment": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
