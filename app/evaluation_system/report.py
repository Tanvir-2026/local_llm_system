"""Summaries and the Markdown/JSON evaluation report (stdlib only)."""
import json
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.evaluation_system.eval_models import QuestionResult


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, round(p / 100 * len(s) + 0.5) - 1))  # nearest-rank
    return s[k]


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def summarize(results: list[QuestionResult]) -> dict:
    ok = [r for r in results if r.status == "success"]
    lat = [r.latency_ms for r in ok]
    total_sec = sum(lat) / 1000
    judged = [r for r in ok if r.judge_correctness is not None]
    return {
        "questions": len(results),
        "succeeded": len(ok),
        "failed": len(results) - len(ok),
        "avg_latency_ms": round(statistics.mean(lat), 1) if lat else None,
        "median_latency_ms": round(statistics.median(lat), 1) if lat else None,
        "p95_latency_ms": round(_percentile(lat, 95), 1) if lat else None,
        "avg_output_tokens": _avg([r.output_tokens for r in ok]),
        "tokens_per_sec": round(sum(r.output_tokens for r in ok) / total_sec, 1) if total_sec else None,
        # failures count as wrong, so the denominator is all questions
        "keyword_accuracy_pct": round(100 * sum(r.keyword_correct for r in results) / len(results), 1) if results else 0.0,
        "judged": len(judged),
        "avg_judge_relevance": _avg([r.judge_relevance for r in judged]),
        "avg_judge_correctness": _avg([r.judge_correctness for r in judged]),
    }


def _verdict(summaries: dict[str, dict]) -> list[str]:
    (a, sa), (b, sb) = list(summaries.items())
    lines = []

    if sa["judged"] and sb["judged"]:
        ka, kb = (sa["avg_judge_correctness"], sa["keyword_accuracy_pct"]), (sb["avg_judge_correctness"], sb["keyword_accuracy_pct"])
        basis = "judge correctness, then keyword accuracy"
    else:
        ka, kb = (sa["keyword_accuracy_pct"],), (sb["keyword_accuracy_pct"],)
        basis = "keyword accuracy"
    best = "tie" if ka == kb else (a if ka > kb else b)
    lines.append(f"- **Most accurate:** {best} (by {basis})")

    la, lb = sa["median_latency_ms"], sb["median_latency_ms"]
    if la is not None and lb is not None:
        fastest = "tie" if la == lb else (a if la < lb else b)
        lines.append(f"- **Fastest (median latency):** {fastest}")
    return lines


def _cell(r: QuestionResult | None) -> str:
    if r is None:
        return "-"
    if r.status != "success":
        return f"ERR ({r.status}) | - | {r.latency_ms:.0f}"
    kw = "✅" if r.keyword_correct else ("🟡" if r.keyword_score > 0 else "❌")
    judge = f"{r.judge_relevance}/{r.judge_correctness}" if r.judge_correctness is not None else "n/a"
    return f"{kw} | {judge} | {r.latency_ms:.0f}"


def _esc(text: str, limit: int | None = None) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if limit is None or len(text) <= limit else text[: limit - 1] + "…"


def render_markdown(meta: dict, summaries: dict[str, dict], results: dict[str, list[QuestionResult]]) -> str:
    names = list(summaries)
    out = [
        "# LLM Evaluation Report",
        "",
        f"- **Date:** {meta['timestamp']}",
        f"- **Models compared:** {names[0]}  vs  {names[1]}",
        f"- **Questions:** {meta['num_questions']}",
        f"- **Judge:** {meta['judge'] or 'none (keyword checks only)'}",
        "",
        "## Summary",
        "",
        "| Metric | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    rows = [
        ("Succeeded / failed", lambda s: f"{s['succeeded']} / {s['failed']}"),
        ("Avg latency (ms)", lambda s: s["avg_latency_ms"]),
        ("Median latency (ms)", lambda s: s["median_latency_ms"]),
        ("P95 latency (ms)", lambda s: s["p95_latency_ms"]),
        ("Avg output tokens", lambda s: s["avg_output_tokens"]),
        ("Output tokens / sec", lambda s: s["tokens_per_sec"]),
        ("Keyword accuracy (%)", lambda s: s["keyword_accuracy_pct"]),
        ("Judge relevance (avg, 1-5)", lambda s: s["avg_judge_relevance"] if s["judged"] else "n/a"),
        ("Judge correctness (avg, 1-5)", lambda s: s["avg_judge_correctness"] if s["judged"] else "n/a"),
    ]
    for label, fn in rows:
        out.append(f"| {label} | " + " | ".join(str(fn(summaries[n])) for n in names) + " |")

    out += ["", "## Verdict", ""] + _verdict(summaries)
    out += [
        "",
        "## Per-question results",
        "",
        "Keyword: ✅ all expected keywords found, 🟡 some, ❌ none. Judge column is relevance/correctness (1-5).",
        "",
        f"| # | Category | Question | {names[0]}: KW | Judge | ms | {names[1]}: KW | Judge | ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    by_id = {n: {r.question_id: r for r in results[n]} for n in names}
    ids = sorted(set().union(*[set(d) for d in by_id.values()]))
    for qid in ids:
        ra, rb = by_id[names[0]].get(qid), by_id[names[1]].get(qid)
        ref = ra or rb
        out.append(f"| {qid} | {ref.category} | {_esc(ref.question, 60)} | {_cell(ra)} | {_cell(rb)} |")

    out += ["", "## Answers", ""]
    for qid in ids:
        ref = by_id[names[0]].get(qid) or by_id[names[1]].get(qid)
        out.append(f"**Q{qid}. {ref.question}**")
        out.append("")
        for n in names:
            r = by_id[n].get(qid)
            if r is None:
                continue
            body = _esc(r.answer, 400) if r.status == "success" else f"_error: {r.status} - {_esc(r.error or '')}_"
            reason = f" (judge: {_esc(r.judge_reason, 120)})" if r.judge_reason else ""
            out.append(f"- `{n}`: {body}{reason}")
        out.append("")
    return "\n".join(out)


def save_report(out_dir: str, meta: dict, summaries: dict, results: dict) -> tuple[Path, Path]:
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path, json_path = folder / f"eval_{stamp}.md", folder / f"eval_{stamp}.json"
    md_path.write_text(render_markdown(meta, summaries, results), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {"meta": meta, "summaries": summaries,
             "results": {n: [asdict(r) for r in rs] for n, rs in results.items()}},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return md_path, json_path