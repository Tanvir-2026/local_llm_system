"""Append-only JSONL usage log + summary stats."""
import json
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


class UsageTracker:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, **fields) -> dict:
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **fields}
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def summary(self, recent: int = 10) -> dict:
        rows = self.read_all()
        per_model: dict = defaultdict(
            lambda: {"calls": 0, "success": 0, "failures": 0,
                     "total_latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0}
        )
        for r in rows:
            key = f'{r["provider"]}:{r["model"]}'
            m = per_model[key]
            m["calls"] += 1
            m["total_latency_ms"] += r.get("latency_ms", 0)
            m["input_tokens"] += r.get("input_tokens", 0)
            m["output_tokens"] += r.get("output_tokens", 0)
            if r["status"] == "success":
                m["success"] += 1
            else:
                m["failures"] += 1

        stats = {}
        for key, m in per_model.items():
            stats[key] = {
                **m,
                "avg_latency_ms": round(m["total_latency_ms"] / m["calls"], 1),
            }
            del stats[key]["total_latency_ms"]

        return {
            "total_attempts": len(rows),
            "fallbacks_used": sum(1 for r in rows if r.get("is_fallback")),
            "per_model": stats,
            "recent": rows[-recent:],
        }
