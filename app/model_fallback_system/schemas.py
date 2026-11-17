"""Request/response models for Problem 2."""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    system: str | None = None


class ChatResponse(BaseModel):
    response: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    fallback_used: bool
    request_id: str
    attempts: list[dict]
