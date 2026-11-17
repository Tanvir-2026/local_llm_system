"""Problem 2 tests: simulate Ollama failures and assert Claude takes over."""
import asyncio
import tempfile
from pathlib import Path

import pytest

from app.model_fallback_system.fallback_manager import AllProvidersFailed, FallbackManager
from app.model_fallback_system.providers.base import (
    LLMProvider, ProviderConnectionError, ProviderResult, ProviderTimeout,
)
from app.model_fallback_system.usage_tracker import UsageTracker


class Fake(LLMProvider):
    def __init__(self, name, behaviour, timeout=1.0):
        self.name, self.model, self.timeout = name, f"{name}-model", timeout
        self.behaviour, self.calls = behaviour, 0

    async def generate(self, prompt, system=None, json_mode=False):
        self.calls += 1
        if isinstance(self.behaviour, Exception):
            raise self.behaviour
        if self.behaviour == "hang":
            await asyncio.sleep(10)
        return ProviderResult(text=self.behaviour, provider=self.name,
                              model=self.model, input_tokens=5, output_tokens=7)


def make(providers):
    path = Path(tempfile.mkdtemp()) / "usage.jsonl"
    tracker = UsageTracker(str(path))
    return FallbackManager(providers, tracker, max_retries=1, backoff=0.01), tracker


@pytest.mark.asyncio
async def test_primary_success_no_fallback():
    mgr, tracker = make([Fake("ollama", "hi"), Fake("claude", "x")])
    r = await mgr.generate("hello")
    assert r.provider == "ollama" and not r.fallback_used
    assert tracker.summary()["total_attempts"] == 1


@pytest.mark.asyncio
async def test_connection_error_falls_back_after_retry():
    ollama = Fake("ollama", ProviderConnectionError("down"))
    mgr, tracker = make([ollama, Fake("claude", "from claude")])
    r = await mgr.generate("hello")
    assert r.provider == "claude" and r.fallback_used
    assert ollama.calls == 2  # 1 try + 1 retry
    assert tracker.summary()["fallbacks_used"] == 1


@pytest.mark.asyncio
async def test_hard_timeout_falls_back():
    mgr, _ = make([Fake("ollama", "hang", timeout=0.05), Fake("claude", "ok")])
    r = await mgr.generate("hello")
    assert r.provider == "claude"
    assert any(a["status"] == "timeout" for a in r.attempts)


@pytest.mark.asyncio
async def test_invalid_output_triggers_fallback():
    def must_be_json(t):
        if not t.startswith("{"):
            raise ValueError("not json")
    mgr, _ = make([Fake("ollama", "garbage"), Fake("claude", '{"a":1}')])
    r = await mgr.generate("x", validator=must_be_json)
    assert r.provider == "claude"


@pytest.mark.asyncio
async def test_all_fail_raises():
    mgr, _ = make([Fake("ollama", ProviderTimeout("t")),
                   Fake("claude", ProviderConnectionError("c"))])
    with pytest.raises(AllProvidersFailed) as e:
        await mgr.generate("x")
    assert len(e.value.attempts) == 4
