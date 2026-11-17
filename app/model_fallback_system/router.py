"""Endpoints: POST /fallback/chat, GET /fallback/usage"""
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_manager, get_tracker
from app.model_fallback_system.fallback_manager import AllProvidersFailed, FallbackManager
from app.model_fallback_system.schemas import ChatRequest, ChatResponse
from app.model_fallback_system.usage_tracker import UsageTracker

router = APIRouter(prefix="/fallback", tags=["Problem 2 - Fallback"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, manager: FallbackManager = Depends(get_manager)):
    try:
        r = await manager.generate(req.prompt, system=req.system)
    except AllProvidersFailed as e:
        raise HTTPException(
            status_code=503,
            detail={"error": "All providers failed", "attempts": e.attempts},
        )
    return ChatResponse(
        response=r.text, provider=r.provider, model=r.model,
        latency_ms=r.latency_ms, input_tokens=r.input_tokens,
        output_tokens=r.output_tokens, fallback_used=r.fallback_used,
        request_id=r.request_id, attempts=r.attempts,
    )


@router.get("/usage")
async def usage(tracker: UsageTracker = Depends(get_tracker)):
    return tracker.summary()
