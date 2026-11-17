"""Try the primary provider, then the fallback. Log every attempt."""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from app.model_fallback_system.providers.base import (
    InvalidOutputError,
    LLMProvider,
    ProviderError,
    ProviderResult,
    ProviderTimeout,
)
from app.model_fallback_system.usage_tracker import UsageTracker

log = logging.getLogger(__name__)


class AllProvidersFailed(Exception):
    def __init__(self, attempts: list[dict]):
        super().__init__("All providers failed")
        self.attempts = attempts


@dataclass
class FallbackResult:
    text: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    fallback_used: bool
    request_id: str
    attempts: list[dict] = field(default_factory=list)


class FallbackManager:
    def __init__(
        self,
        providers: list[LLMProvider],
        tracker: UsageTracker,
        max_retries: int = 1,
        backoff: float = 0.5,
    ):
        """`providers` is ordered: first = primary, rest = fallbacks."""
        self.providers = providers
        self.tracker = tracker
        self.max_retries = max_retries
        self.backoff = backoff

    async def _call(self, provider: LLMProvider, prompt, system, json_mode) -> ProviderResult:
        try:
            return await asyncio.wait_for(
                provider.generate(prompt, system=system, json_mode=json_mode),
                timeout=provider.timeout,  # hard total timeout
            )
        except asyncio.TimeoutError as e:
            raise ProviderTimeout(f"{provider.name} exceeded {provider.timeout}s") from e

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        validator: Callable[[str], None] | None = None,
    ) -> FallbackResult:
        """`validator(text)` should raise ValueError if the output is unusable;
        that counts as a failure and triggers the next provider."""
        request_id = uuid.uuid4().hex[:12]
        attempts: list[dict] = []

        for idx, provider in enumerate(self.providers):
            is_fallback = idx > 0
            for try_no in range(1 + self.max_retries):
                start = time.perf_counter()
                try:
                    result = await self._call(provider, prompt, system, json_mode)
                    if validator:
                        try:
                            validator(result.text)
                        except ValueError as e:
                            raise InvalidOutputError(str(e)) from e
                except ProviderError as err:
                    latency = (time.perf_counter() - start) * 1000
                    entry = self.tracker.record(
                        request_id=request_id, provider=provider.name,
                        model=provider.model, latency_ms=round(latency, 1),
                        input_tokens=0, output_tokens=0,
                        status=err.error_type, error=err.message[:300],
                        is_fallback=is_fallback, attempt=try_no + 1,
                    )
                    attempts.append(entry)
                    log.warning("%s attempt %d failed: %s (%s)",
                                provider.name, try_no + 1, err.error_type, err.message)
                    if err.retryable and try_no < self.max_retries:
                        await asyncio.sleep(self.backoff)
                        continue
                    break  # move on to next provider
                else:
                    latency = (time.perf_counter() - start) * 1000
                    entry = self.tracker.record(
                        request_id=request_id, provider=provider.name,
                        model=provider.model, latency_ms=round(latency, 1),
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        status="success", error=None,
                        is_fallback=is_fallback, attempt=try_no + 1,
                    )
                    attempts.append(entry)
                    return FallbackResult(
                        text=result.text, provider=result.provider, model=result.model,
                        latency_ms=round(latency, 1),
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        fallback_used=is_fallback, request_id=request_id,
                        attempts=attempts,
                    )

        raise AllProvidersFailed(attempts)
