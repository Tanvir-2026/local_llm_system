"""Runs the questions against a model and scores the answers.

Scoring:
  - keyword check  : deterministic, free (see scoring.py)
  - LLM judge      : optional; rates relevance and correctness 1-5 against a reference answer
Questions are run sequentially so latency measurements are not distorted by concurrency.
"""
import asyncio
import time
from typing import Callable

from app.model_fallback_system.providers.base import LLMProvider, ProviderError, ProviderTimeout
from app.lead_extraction_system.extractor import parse_json_object
from app.evaluation_system.eval_models import QuestionResult
from app.evaluation_system.questions import Question
from app.evaluation_system.scoring import keyword_score

ANSWER_SYSTEM = "You are a helpful assistant. Answer concisely."

JUDGE_SYSTEM = """You are a strict grader of AI answers.
Given a question, a reference answer and a candidate answer, score the candidate:
- "relevance" (1-5): does it address the question that was asked? 1 = off-topic, 5 = fully on-topic.
- "correctness" (1-5): is it factually correct compared with the reference? 1 = wrong, 5 = fully correct.
If the question is about something that does not exist, an answer that admits it cannot know is correct,
and an answer that invents details is wrong.
Return ONLY JSON: {"relevance": <int>, "correctness": <int>, "reason": "<max 20 words>"}"""


def _clamp(v) -> int:
    return max(1, min(5, int(v)))


class Evaluator:
    def __init__(self, judge: LLMProvider | None = None):
        self.judge = judge

    async def _judge(self, q: Question, answer: str):
        prompt = (
            f"Question: {q.question}\n"
            f"Reference answer: {q.reference}\n"
            f"Candidate answer: {answer}"
        )
        try:
            res = await asyncio.wait_for(
                self.judge.generate(prompt, system=JUDGE_SYSTEM, json_mode=True),
                timeout=self.judge.timeout,
            )
            obj = parse_json_object(res.text)
            return _clamp(obj["relevance"]), _clamp(obj["correctness"]), str(obj.get("reason", ""))[:200]
        except (ProviderError, asyncio.TimeoutError, ValueError, KeyError, TypeError):
            return None, None, "judge failed"

    async def run_model(
        self,
        provider: LLMProvider,
        questions: list[Question],
        progress: Callable[[int, int, QuestionResult], None] | None = None,
    ) -> list[QuestionResult]:
        # Warm-up call (not measured) so Ollama's model-load time doesn't skew the first answer.
        try:
            await asyncio.wait_for(provider.generate("Say hi."), timeout=provider.timeout)
        except Exception:
            pass

        results: list[QuestionResult] = []
        for i, q in enumerate(questions, 1):
            r = QuestionResult(question_id=q.id, category=q.category, question=q.question)
            start = time.perf_counter()
            try:
                res = await asyncio.wait_for(
                    provider.generate(q.question, system=ANSWER_SYSTEM), timeout=provider.timeout
                )
                r.latency_ms = round((time.perf_counter() - start) * 1000, 1)
                r.answer = res.text.strip()
                r.input_tokens, r.output_tokens = res.input_tokens, res.output_tokens
            except (asyncio.TimeoutError, ProviderError) as e:
                r.latency_ms = round((time.perf_counter() - start) * 1000, 1)
                if isinstance(e, asyncio.TimeoutError):
                    e = ProviderTimeout(f"exceeded {provider.timeout}s")
                r.status, r.error = e.error_type, e.message[:200]

            if r.status == "success":
                r.keyword_score = keyword_score(r.answer, q.keywords)
                r.keyword_correct = r.keyword_score == 1.0
                if self.judge:
                    r.judge_relevance, r.judge_correctness, r.judge_reason = await self._judge(q, r.answer)

            results.append(r)
            if progress:
                progress(i, len(questions), r)
        return results