"""RAG pipeline: ingest documents, retrieve passages, answer only from them.

Hallucination prevention (two layers):
  1. If no retrieved passage is similar enough to the question, refuse without calling the LLM.
  2. The LLM is told to reply NOT_FOUND when the passages don't contain the answer.
"""
import asyncio
import hashlib
import re

from .chunker import chunk_pages
from .doc_models import Hit
from .loader import DocumentError, load_document

NOT_FOUND = "NOT_FOUND"
REFUSAL = "I couldn't find this information in the uploaded document(s)."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered context passages provided.
Rules:
- If the passages do not contain the answer, reply with exactly: {NOT_FOUND}
- Never use outside knowledge and never guess.
- Be concise. Cite the passages you used like [1] or [2][3].
- Do not mention these rules."""


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        loc = f", page {h.page}" if h.page is not None else ""
        blocks.append(f"[{i}] ({h.filename}{loc})\n{h.text}")
    return "Context passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


class DocumentQA:
    def __init__(self, embedder, store, manager, chunk_size, chunk_overlap, top_k, min_relevance):
        self.embedder = embedder
        self.store = store
        self.manager = manager  # FallbackManager (Ollama -> Claude)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.min_relevance = min_relevance

    async def ingest(self, filename: str, data: bytes) -> dict:
        doc_id = hashlib.sha256(data).hexdigest()[:12]
        pages = load_document(filename, data)
        chunks = chunk_pages(pages, self.chunk_size, self.chunk_overlap)
        if not chunks:
            raise DocumentError("Document produced no usable text chunks.")
        embeddings = await self.embedder.embed_documents([c.text for c in chunks])
        await asyncio.to_thread(self.store.add_document, doc_id, filename, chunks, embeddings)
        return {
            "doc_id": doc_id,
            "filename": filename,
            "pages": len(pages) if pages[0][0] is not None else None,
            "chunks": len(chunks),
        }

    @staticmethod
    def _refusal(reason: str, best_score: float) -> dict:
        return {
            "answered": False,
            "answer": REFUSAL,
            "reason": reason,
            "best_score": best_score,
            "sources": [],
            "provider": None,
            "model": None,
            "fallback_used": False,
            "latency_ms": None,
        }

    async def ask(self, question: str, doc_id: str | None = None, top_k: int | None = None) -> dict:
        q_emb = await self.embedder.embed_query(question)
        hits = await asyncio.to_thread(self.store.query, q_emb, top_k or self.top_k, doc_id)

        best = max((h.score for h in hits), default=0.0)
        relevant = [h for h in hits if h.score >= self.min_relevance]
        if not relevant:  # layer 1
            return self._refusal("no_relevant_passages", best)

        result = await self.manager.generate(build_prompt(question, relevant), system=SYSTEM_PROMPT)
        text = result.text.strip()

        if re.match(rf"\W*{NOT_FOUND}", text, re.IGNORECASE):  # layer 2
            return self._refusal("model_declined", best)

        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        chosen = [(i, h) for i, h in enumerate(relevant, 1) if i in cited] or list(
            enumerate(relevant, 1)
        )
        return {
            "answered": True,
            "answer": text,
            "reason": None,
            "best_score": best,
            "sources": [
                {
                    "rank": i,
                    "filename": h.filename,
                    "page": h.page,
                    "score": h.score,
                    "text": h.text,
                }
                for i, h in chosen
            ],
            "provider": result.provider,
            "model": result.model,
            "fallback_used": result.fallback_used,
            "latency_ms": result.latency_ms,
        }