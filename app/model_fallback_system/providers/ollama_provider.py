"""Ollama provider: POST {base_url}/api/chat."""
import httpx

from app.model_fallback_system.providers.base import (
    LLMProvider,
    ProviderAPIError,
    ProviderConnectionError,
    ProviderResult,
    ProviderTimeout,
)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def generate(self, prompt, system=None, json_mode=False) -> ProviderResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        if json_mode:
            payload["format"] = "json"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException as e:
            raise ProviderTimeout(f"Ollama timed out after {self.timeout}s") from e
        except httpx.ConnectError as e:
            raise ProviderConnectionError(
                f"Cannot reach Ollama at {self.base_url} (is `ollama serve` running?)"
            ) from e
        except httpx.HTTPStatusError as e:
            raise ProviderAPIError(
                f"Ollama HTTP {e.response.status_code}: {e.response.text[:200]}",
                status_code=e.response.status_code,
            ) from e
        except httpx.HTTPError as e:
            raise ProviderConnectionError(f"Ollama network error: {e}") from e

        text = data.get("message", {}).get("content", "")
        return ProviderResult(
            text=text,
            provider=self.name,
            model=self.model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )
