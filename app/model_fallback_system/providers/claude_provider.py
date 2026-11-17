"""Claude provider using the official anthropic SDK."""
import anthropic

from app.model_fallback_system.providers.base import (
    LLMProvider,
    ProviderAPIError,
    ProviderConnectionError,
    ProviderResult,
    ProviderTimeout,
)

JSON_HINT = "Respond with a single valid JSON object only. No prose, no code fences."


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self, api_key: str, model: str, timeout: float):
        self.model = model
        self.timeout = timeout
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or "missing", timeout=timeout, max_retries=0
        )
        self._has_key = bool(api_key)

    async def generate(self, prompt, system=None, json_mode=False) -> ProviderResult:
        if not self._has_key:
            raise ProviderAPIError("ANTHROPIC_API_KEY is not set", status_code=401)

        system_text = system or ""
        if json_mode:
            system_text = f"{system_text}\n\n{JSON_HINT}".strip()

        kwargs = {
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_text:
            kwargs["system"] = system_text

        try:
            msg = await self._client.messages.create(**kwargs)
        except anthropic.APITimeoutError as e:  # must come before APIConnectionError
            raise ProviderTimeout(f"Claude timed out after {self.timeout}s") from e
        except anthropic.APIConnectionError as e:
            raise ProviderConnectionError(f"Cannot reach Claude API: {e}") from e
        except anthropic.APIStatusError as e:
            raise ProviderAPIError(
                f"Claude HTTP {e.status_code}: {e.message}", status_code=e.status_code
            ) from e

        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return ProviderResult(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )
