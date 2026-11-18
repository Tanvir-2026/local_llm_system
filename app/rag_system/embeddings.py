"""Embeddings via Ollama (/api/embed). Default model: nomic-embed-text."""
import httpx


class EmbeddingError(Exception):
    pass


class OllamaEmbedder:
    BATCH = 16

    def __init__(self, base_url: str, model: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # nomic-embed-text works best with task prefixes
        self.use_prefix = "nomic" in model.lower()

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": inputs},
                )
                r.raise_for_status()
                return r.json()["embeddings"]
        except httpx.TimeoutException as e:
            raise EmbeddingError(f"Embedding request timed out after {self.timeout}s") from e
        except httpx.ConnectError as e:
            raise EmbeddingError(
                f"Cannot reach Ollama at {self.base_url} (is `ollama serve` running?)"
            ) from e
        except httpx.HTTPStatusError as e:
            hint = f" Run: ollama pull {self.model}" if e.response.status_code == 404 else ""
            raise EmbeddingError(
                f"Ollama embed HTTP {e.response.status_code}.{hint}"
            ) from e
        except (httpx.HTTPError, KeyError, ValueError) as e:
            raise EmbeddingError(f"Embedding failed: {e}") from e

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefix = "search_document: " if self.use_prefix else ""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            batch = [prefix + t for t in texts[i : i + self.BATCH]]
            out.extend(await self._embed(batch))
        return out

    async def embed_query(self, text: str) -> list[float]:
        prefix = "search_query: " if self.use_prefix else ""
        return (await self._embed([prefix + text]))[0]