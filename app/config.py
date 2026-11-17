"""Settings loaded from .env (pydantic-settings)."""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama (local)
    ollama_base_url: str 
    ollama_model: str 
    ollama_timeout_seconds: float = 60.0  # first call can be slow (model load)

    # Claude (API)
    anthropic_api_key: str 
    claude_model: str 
    claude_timeout_seconds: float = 30.0

    # Fallback behaviour
    primary_provider: Literal["ollama", "claude"] = "ollama"
    fallback_provider: Literal["ollama", "claude"] = "claude"
    max_retries_per_provider: int = 1
    retry_backoff_seconds: float = 0.5
    usage_log_path: str = "data/logs/usage.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
