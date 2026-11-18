"""Request/response models for Problem 4."""
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int | None
    chunks: int


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    chunks: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    doc_id: str | None = None  # limit the search to one document
    top_k: int | None = Field(default=None, ge=1, le=10)


class Source(BaseModel):
    rank: int
    filename: str
    page: int | None
    score: float
    text: str


class AskResponse(BaseModel):
    answered: bool
    answer: str
    reason: str | None = None  # "no_relevant_passages" | "model_declined" when refused
    best_score: float
    sources: list[Source]
    provider: str | None = None
    model: str | None = None
    fallback_used: bool = False
    latency_ms: float | None = None