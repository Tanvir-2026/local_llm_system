"""CLI: compare two LLMs on the 20 questions and write a report.

Examples:
  uv run python -m app.problem5_eval.run_eval
  uv run python -m app.problem5_eval.run_eval --model-a ollama:qwen2.5:3b --model-b claude
  uv run python -m app.problem5_eval.run_eval --judge none --limit 5

Provider spec format:  ollama[:model]  or  claude[:model]
"""
import argparse
import asyncio
from datetime import datetime

from app.config import get_settings
from app.model_fallback_system.providers.base import LLMProvider
from app.model_fallback_system.providers.claude_provider import ClaudeProvider
from app.model_fallback_system.providers.ollama_provider import OllamaProvider
from app.evaluation_system.evaluator import Evaluator
from app.evaluation_system.questions import QUESTIONS
from app.evaluation_system.report import save_report, summarize


def build_provider(spec: str, s) -> tuple[str, LLMProvider]:
    kind, _, model = spec.partition(":")  # split on first ':' only (models like qwen2.5:3b)
    if kind == "ollama":
        p = OllamaProvider(s.ollama_base_url, model or s.ollama_model, s.ollama_timeout_seconds)
    elif kind == "claude":
        if not s.anthropic_api_key:
            raise SystemExit("ANTHROPIC_API_KEY is not set in .env (needed for a claude model/judge).")
        p = ClaudeProvider(s.anthropic_api_key, model or s.claude_model, s.claude_timeout_seconds)
    else:
        raise SystemExit(f"Unknown provider '{spec}'. Use ollama[:model] or claude[:model].")
    return f"{p.name}:{p.model}", p


def _progress(label: str):
    def cb(i, total, r):
        mark = "ok " if r.status == "success" and r.keyword_correct else ("err" if r.status != "success" else "...")
        print(f"  [{label} {i:>2}/{total}] {mark} {r.latency_ms:>8.0f} ms  {r.question[:55]}")
    return cb


async def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-a", default="ollama:qwen2.5:3b")
    ap.add_argument("--model-b", default="ollama:llama3.2:3b")
    ap.add_argument("--judge", default=None, help="claude[:model] | ollama[:model] | none (default: claude if key set, else none)")
    ap.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    name_a, prov_a = build_provider(args.model_a, s)
    name_b, prov_b = build_provider(args.model_b, s)
    if name_a == name_b:
        raise SystemExit("Pick two different models to compare.")

    judge_spec = args.judge or ("claude" if s.anthropic_api_key else "none")
    judge_name, judge = (None, None) if judge_spec == "none" else build_provider(judge_spec, s)

    questions = QUESTIONS[: args.limit] if args.limit else QUESTIONS
    evaluator = Evaluator(judge)
    print(f"Comparing {name_a} vs {name_b} on {len(questions)} questions (judge: {judge_name or 'none'})")

    results = {}
    for name, prov in ((name_a, prov_a), (name_b, prov_b)):
        print(f"\nRunning {name} ...")
        results[name] = await evaluator.run_model(prov, questions, _progress(name.split(":")[-1][:8]))

    summaries = {n: summarize(r) for n, r in results.items()}
    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "num_questions": len(questions),
        "judge": judge_name,
    }
    md_path, json_path = save_report(args.out, meta, summaries, results)
    print(f"\nReport written:\n  {md_path}\n  {json_path}")


if __name__ == "__main__":
    asyncio.run(main())