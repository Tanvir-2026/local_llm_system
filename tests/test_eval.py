"""Problem 5: questions, keyword scoring, evaluator loop, report rendering."""
import pytest

from app.model_fallback_system.providers.base import LLMProvider, ProviderConnectionError, ProviderResult
from app.evaluation_system.eval_models import QuestionResult
from app.evaluation_system.evaluator import Evaluator
from app.evaluation_system.questions import QUESTIONS
from app.evaluation_system.report import render_markdown, summarize
from app.evaluation_system.scoring import keyword_score


def test_twenty_unique_questions():
    assert len(QUESTIONS) == 20
    assert len({q.id for q in QUESTIONS}) == 20
    assert all(q.keywords and q.reference for q in QUESTIONS)


def test_keyword_scoring():
    kw = (("red",), ("green",), ("blue",))
    assert keyword_score("Red, green and blue", kw) == 1.0
    assert abs(keyword_score("Red and green", kw) - 2 / 3) < 1e-9
    assert keyword_score("The element is Au.", (("au",),)) == 1.0
    assert keyword_score("because of laundry", (("au",),)) == 0.0  # whole-word only
    assert keyword_score("It costs $0.05.", (("5 cents", "$0.05"),)) == 1.0
    assert keyword_score("It is 360", (("36",),)) == 0.0


class FakeLLM(LLMProvider):
    def __init__(self, name, answers=None, fail=False, text=None):
        self.name, self.model, self.timeout = name, name + "-m", 1.0
        self.answers, self.fail, self.text = answers or {}, fail, text

    async def generate(self, prompt, system=None, json_mode=False):
        if self.fail:
            raise ProviderConnectionError("down")
        if self.text is not None:
            return ProviderResult(self.text, self.name, self.model, 3, 4)
        return ProviderResult(self.answers.get(prompt, "I do not know."), self.name, self.model, 3, 4)


@pytest.mark.asyncio
async def test_run_model_scores_and_handles_failures():
    answers = {QUESTIONS[0].question: "Canberra.", QUESTIONS[1].question: "Marlowe."}
    ev = Evaluator(judge=None)
    res = await ev.run_model(FakeLLM("a", answers), QUESTIONS[:2])
    assert res[0].keyword_correct and not res[1].keyword_correct
    assert res[0].latency_ms >= 0 and res[0].output_tokens == 4

    bad = await ev.run_model(FakeLLM("b", fail=True), QUESTIONS[:2])
    assert all(r.status == "connection_error" and not r.keyword_correct for r in bad)


@pytest.mark.asyncio
async def test_judge_scores_are_parsed_and_clamped():
    judge = FakeLLM("j", text='{"relevance": 9, "correctness": 4, "reason": "ok"}')
    res = await Evaluator(judge).run_model(FakeLLM("a", {QUESTIONS[0].question: "Canberra"}), QUESTIONS[:1])
    assert (res[0].judge_relevance, res[0].judge_correctness) == (5, 4)

    broken = FakeLLM("j", text="not json at all")
    res = await Evaluator(broken).run_model(FakeLLM("a", {}), QUESTIONS[:1])
    assert res[0].judge_correctness is None


def _r(i, ok=True, ms=100.0, kw=True, jc=None):
    return QuestionResult(i, "factual", f"Q{i}?", "ans" if ok else "", "success" if ok else "timeout",
                          None, ms, 5, 10, 1.0 if kw else 0.0, kw, jc, jc)


def test_summary_and_report():
    results = {
        "ollama:a": [_r(1, ms=100, jc=5), _r(2, ms=300, kw=False, jc=2)],
        "ollama:b": [_r(1, ms=50, kw=False, jc=3), _r(2, ok=False, ms=1000, kw=False)],
    }
    sums = {n: summarize(r) for n, r in results.items()}
    assert sums["ollama:a"]["keyword_accuracy_pct"] == 50.0
    assert sums["ollama:a"]["avg_judge_correctness"] == 3.5
    assert sums["ollama:b"]["failed"] == 1 and sums["ollama:b"]["median_latency_ms"] == 50.0

    md = render_markdown({"timestamp": "now", "num_questions": 2, "judge": "claude:x"}, sums, results)
    assert "# LLM Evaluation Report" in md and "**Most accurate:** ollama:a" in md
    assert "**Fastest (median latency):** ollama:b" in md
    assert "ERR (timeout)" in md