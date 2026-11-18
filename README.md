# Assignments

- ** AI Model Fallback System.** Local Ollama model first, Claude API as fallback. Handles timeouts and errors, and logs model, response time and token usage.
- ** Smart Lead Extraction.** Extracts name, email, phone and requirements from a conversation as JSON, detects missing fields, and never re-asks for details already collected.

Built with FastAPI.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) installed locally
- An Anthropic API key (used as the fallback)

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Pull a local model
ollama pull qwen2.5:3b          # or: ollama pull llama3.2:3b

# 3. Configure environment
cp .env.example .env            # Windows: copy .env.example .env
```

Edit `.env` and set `ANTHROPIC_API_KEY`. To use Llama instead of Qwen, set `OLLAMA_MODEL=llama3.2:3b`.

## Run

Make sure Ollama is running (`ollama serve`), then:

```bash
uv run uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive API.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/fallback/chat` | Send a prompt. Ollama answers, Claude takes over if it fails |
| GET | `/fallback/usage` | Per-model calls, failures, average latency, token counts |
| POST | `/leads/extract` | Extract lead details from a conversation |

Example, lead extraction (pass the previous response's `lead` back as `existing_lead` on later turns):

```json
{
  "conversation": [
    {"role": "user", "content": "Hi, I'm Sara Khan and I need a website for my bakery."}
  ],
  "existing_lead": null
}
```

## Tests

```bash
uv run pytest                             # unit tests (fake providers)
```

## Try the fallback

1. Call `POST /fallback/chat` normally. The response shows `provider: ollama`.
2. Stop Ollama (or set `OLLAMA_MODEL=doesnotexist`) and call again. The response shows `provider: claude` and `fallback_used: true`.
3. Check `GET /fallback/usage`. The raw log is in `data/logs/usage.jsonl`.

## Project structure

```
app/
  main.py                    FastAPI entrypoint
  config.py                  Settings from .env
  dependencies.py            Builds the shared fallback manager and tracker
  model_fallback_system/     Providers, fallback manager, usage tracker, router
  lead_extraction_system/    Prompts, extractor, missing-info logic, router
  rag_system                 
  evaluation_system
tests/                       Unit tests
data/logs/                   Usage logs (JSONL)
```

## Configuration (.env)

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | None | Ollama server |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Local model |
| `OLLAMA_TIMEOUT_SECONDS` | `60` | Local call timeout |
| `ANTHROPIC_API_KEY` | None | Your API key |
| `CLAUDE_MODEL` | `your model` | Fallback model |
| `CLAUDE_TIMEOUT_SECONDS` | `30` | Claude call timeout |
| `PRIMARY_PROVIDER` / `FALLBACK_PROVIDER` | `ollama` / `claude` | Provider order |
| `MAX_RETRIES_PER_PROVIDER` | `1` | Retries on timeout, connection error, 429 or 5xx |
| `CHROMA_DIR` | None | data/ |
| `QA_TOP_K` | None | 3 |
| `QA_MAX_DISTANCE` | None | 0.34 |
