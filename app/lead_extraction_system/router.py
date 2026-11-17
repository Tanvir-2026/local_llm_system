"""Endpoints: POST /leads/extract (conversation + existing lead state -> JSON)"""
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_manager
from app.model_fallback_system.fallback_manager import AllProvidersFailed, FallbackManager
from app.lead_extraction_system.extractor import LeadExtractor
from app.lead_extraction_system.schemas import ExtractRequest, ExtractResponse

router = APIRouter(prefix="/leads", tags=["Problem 3 - Lead Extraction"])


@router.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest, manager: FallbackManager = Depends(get_manager)):
    try:
        return await LeadExtractor(manager).extract(req)
    except AllProvidersFailed as e:
        raise HTTPException(
            status_code=503,
            detail={"error": "All providers failed", "attempts": e.attempts},
        )
