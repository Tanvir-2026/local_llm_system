"""Builds the shared tracker + fallback manager once."""
from functools import lru_cache

from app.config import get_settings
from app.model_fallback_system.fallback_manager import FallbackManager
from app.model_fallback_system.providers.base import LLMProvider
from app.model_fallback_system.providers.claude_provider import ClaudeProvider
from app.model_fallback_system.providers.ollama_provider import OllamaProvider
from app.model_fallback_system.usage_tracker import UsageTracker


@lru_cache
def get_tracker() -> UsageTracker:
    return UsageTracker(get_settings().usage_log_path)


@lru_cache
def get_manager() -> FallbackManager:
    s = get_settings()
    available: dict[str, LLMProvider] = {
        "ollama": OllamaProvider(s.ollama_base_url, s.ollama_model, s.ollama_timeout_seconds),
        "claude": ClaudeProvider(s.anthropic_api_key, s.claude_model, s.claude_timeout_seconds),
    }
    order = [s.primary_provider]
    if s.fallback_provider != s.primary_provider:
        order.append(s.fallback_provider)
    return FallbackManager(
        providers=[available[n] for n in order],
        tracker=get_tracker(),
        max_retries=s.max_retries_per_provider,
        backoff=s.retry_backoff_seconds,
    )
