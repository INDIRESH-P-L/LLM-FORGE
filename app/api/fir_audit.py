"""
app/api/fir_audit.py
====================
LegalMind AI — FIR Auditor API
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.fir_audit_service import FIRAuditorService

router = APIRouter(prefix="/api/fir", tags=["FIR Auditor"])

class FIRAuditRequest(BaseModel):
    fir_text: str = Field(..., description="Raw text of the FIR to audit")
    is_pmla: bool = Field(default=False)
    arrest_made: bool = Field(default=False)

@router.post("/audit")
def audit_fir(req: FIRAuditRequest, request: Request):
    if not req.fir_text.strip():
        raise HTTPException(status_code=400, detail="Error: FIR text cannot be empty.")

    try:
        # Pass the live retriever so the applicable provisions and judgments
        # can be pulled from the corpus rather than asserted from memory.
        result = FIRAuditorService.audit_fir(
            fir_text=req.fir_text,
            is_pmla=req.is_pmla,
            arrest_made=req.arrest_made,
            retriever=getattr(request.app.state, "retriever", None),
        )
        return {"status": "success", "audit_report": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
