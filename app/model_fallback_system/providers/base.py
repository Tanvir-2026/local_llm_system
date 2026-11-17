"""Provider interface, result type, and a typed error hierarchy."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderResult:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class ProviderError(Exception):
    """Base error. `error_type` is what gets written to the usage log."""

    error_type = "provider_error"
    retryable = False

    def __init__(self, message: str = ""):
        super().__init__(message)
        self.message = message


class ProviderTimeout(ProviderError):
    error_type = "timeout"
    retryable = True


class ProviderConnectionError(ProviderError):
    error_type = "connection_error"
    retryable = True


class ProviderAPIError(ProviderError):
    error_type = "api_error"

    def __init__(self, message: str = "", status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
        # retry on rate limit / server errors only
        self.retryable = status_code in (429, 500, 502, 503, 504, 529)


class InvalidOutputError(ProviderError):
    """Raised by the manager when a validator rejects the model's output."""

    error_type = "invalid_output"


class LLMProvider(ABC):
    name: str
    model: str
    timeout: float

    @abstractmethod
    async def generate(
        self, prompt: str, system: str | None = None, json_mode: bool = False
    ) -> ProviderResult:
        """Return a ProviderResult or raise a ProviderError subclass."""
