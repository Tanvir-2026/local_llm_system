"""Lead model + request/response models for Problem 3."""
from typing import Literal

from pydantic import BaseModel, Field

LEAD_FIELDS = ["name", "email", "phone", "requirements"]


class Lead(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    requirements: str | None = None


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ExtractRequest(BaseModel):
    conversation: list[Message] = Field(min_length=1)
    existing_lead: Lead | None = None  # pass back the previous response's `lead`


class ExtractResponse(BaseModel):
    lead: Lead
    already_collected: list[str]
    missing_fields: list[str]
    is_complete: bool
    next_question: str | None
    provider_used: str
    model_used: str
    fallback_used: bool
    latency_ms: float
