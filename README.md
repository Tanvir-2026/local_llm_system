# AI Assignments
Problem 2: AI Model Fallback System (Ollama primary -> Claude fallback)
Problem 3: Smart Lead Extraction

## Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY
ollama pull qwen2.5:3b
uvicorn app.main:app --reload
