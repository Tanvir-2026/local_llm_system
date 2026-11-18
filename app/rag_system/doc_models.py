"""Plain dataclasses shared across Problem 4 modules (stdlib only)."""
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    page: int | None  # None for TXT files
    index: int


@dataclass
class Hit:
    text: str
    doc_id: str
    filename: str
    page: int | None
    chunk_index: int
    score: float  # cosine similarity, 0..1 (higher = more relevant)