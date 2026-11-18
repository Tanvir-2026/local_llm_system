"""Endpoints: POST /documents/upload, GET /documents, DELETE /documents/{id}, POST /documents/ask"""
from functools import lru_cache

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import get_settings
from app.dependencies import get_manager
from app.model_fallback_system.fallback_manager import AllProvidersFailed
from .schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    UploadResponse,
)
from .embeddings import EmbeddingError, OllamaEmbedder
from .loader import DocumentError
from .document_qa import DocumentQA
from .vector_store import VectorStore

router = APIRouter(prefix="/documents", tags=["Problem 4 - Document QA"])


@lru_cache
def get_qa() -> DocumentQA:
    s = get_settings()
    return DocumentQA(
        embedder=OllamaEmbedder(s.ollama_base_url, s.embed_model, s.embed_timeout_seconds),
        store=VectorStore(s.chroma_path),
        manager=get_manager(),
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        top_k=s.top_k,
        min_relevance=s.min_relevance,
    )


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), qa: DocumentQA = Depends(get_qa)):
    data = await file.read()
    limit = get_settings().max_upload_mb * 1024 * 1024
    if len(data) > limit:
        raise HTTPException(413, f"File too large (max {get_settings().max_upload_mb} MB)")
    try:
        return await qa.ingest(file.filename or "upload", data)
    except DocumentError as e:
        raise HTTPException(422, str(e))
    except EmbeddingError as e:
        raise HTTPException(503, f"Embedding service error: {e}")


@router.get("", response_model=list[DocumentInfo])
async def list_documents(qa: DocumentQA = Depends(get_qa)):
    return qa.store.list_documents()


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, qa: DocumentQA = Depends(get_qa)):
    qa.store.delete_document(doc_id)
    return {"deleted": doc_id}


@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, qa: DocumentQA = Depends(get_qa)):
    try:
        return await qa.ask(req.question, doc_id=req.doc_id, top_k=req.top_k)
    except EmbeddingError as e:
        raise HTTPException(503, f"Embedding service error: {e}")
    except AllProvidersFailed as e:
        raise HTTPException(
            503, detail={"error": "All LLM providers failed", "attempts": e.attempts}
        )