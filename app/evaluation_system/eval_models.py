"""Result dataclass for one question evaluated on one model (stdlib only)."""
from dataclasses import dataclass


@dataclass
class QuestionResult:
    question_id: int
    category: str
    question: str
    answer: str = ""
    status: str = "success"  # success | timeout | connection_error | api_error | ...
    error: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    keyword_score: float = 0.0  # fraction of expected keyword groups found (0..1)
    keyword_correct: bool = False  # all expected keyword groups found
    judge_relevance: int | None = None  # 1..5 from the judge LLM
    judge_correctness: int | None = None  # 1..5 from the judge LLM
    judge_reason: str = ""