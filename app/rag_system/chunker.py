"""Split text into overlapping chunks, preferring paragraph/sentence boundaries."""
import re

from .doc_models import Chunk

MIN_CHUNK_CHARS = 20


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks: list[str] = []
    n, start = len(text), 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # try to end on a paragraph / sentence boundary in the last 40% of the window
            lo = start + int(chunk_size * 0.6)
            cut = max(text.rfind("\n\n", lo, end), text.rfind(". ", lo, end), text.rfind("\n", lo, end))
            if cut != -1:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(
    pages: list[tuple[int | None, str]], chunk_size: int = 1000, overlap: int = 150
) -> list[Chunk]:
    """Chunk each page separately so every chunk maps to exactly one page number."""
    chunks: list[Chunk] = []
    for page, text in pages:
        for piece in split_text(text, chunk_size, overlap):
            if len(piece) >= MIN_CHUNK_CHARS:
                chunks.append(Chunk(text=piece, page=page, index=len(chunks)))
    return chunks