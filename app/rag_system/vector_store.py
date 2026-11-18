"""ChromaDB persistent vector store (cosine similarity)."""
import chromadb
from chromadb.config import Settings as ChromaSettings

from .doc_models import Chunk, Hit


class VectorStore:
    def __init__(self, path: str):
        self.client = chromadb.PersistentClient(
            path=path, settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.col = self.client.get_or_create_collection(
            "documents", metadata={"hnsw:space": "cosine"}
        )

    def add_document(
        self, doc_id: str, filename: str, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        self.delete_document(doc_id)  # re-uploading the same file replaces it
        self.col.add(
            ids=[f"{doc_id}:{c.index}" for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": c.page if c.page is not None else -1,
                    "chunk_index": c.index,
                }
                for c in chunks
            ],
        )

    def delete_document(self, doc_id: str) -> None:
        self.col.delete(where={"doc_id": doc_id})

    def list_documents(self) -> list[dict]:
        res = self.col.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for m in res.get("metadatas") or []:
            d = docs.setdefault(
                m["doc_id"], {"doc_id": m["doc_id"], "filename": m["filename"], "chunks": 0}
            )
            d["chunks"] += 1
        return list(docs.values())

    def query(self, embedding: list[float], top_k: int, doc_id: str | None = None) -> list[Hit]:
        total = self.col.count()
        if total == 0:
            return []
        kwargs = dict(
            query_embeddings=[embedding],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        if doc_id:
            kwargs["where"] = {"doc_id": doc_id}
        res = self.col.query(**kwargs)

        hits = []
        for text, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(
                Hit(
                    text=text,
                    doc_id=meta["doc_id"],
                    filename=meta["filename"],
                    page=None if meta["page"] == -1 else meta["page"],
                    chunk_index=meta["chunk_index"],
                    score=round(1 - dist, 4),  # cosine distance -> similarity
                )
            )
        return hits