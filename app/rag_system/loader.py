"""Extract text from PDF / TXT files. Returns a list of (page_number | None, text)."""
import io
from pathlib import Path


class DocumentError(Exception):
    """The file is unsupported, unreadable, or has no extractable text."""


def load_document(filename: str, data: bytes) -> list[tuple[int | None, str]]:
    ext = Path(filename).suffix.lower()

    if ext == ".txt":
        for enc in ("utf-8-sig", "utf-16", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeError:
                continue
        pages = [(None, text)]
    elif ext == ".pdf":
        from pypdf import PdfReader  # lazy import

        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise DocumentError("PDF is password-protected")
            pages = [(i + 1, (p.extract_text() or "")) for i, p in enumerate(reader.pages)]
        except DocumentError:
            raise
        except Exception as e:
            raise DocumentError(f"Could not read PDF: {e}") from e
    else:
        raise DocumentError(f"Unsupported file type '{ext}'. Upload a .pdf or .txt file.")

    if not any(t.strip() for _, t in pages):
        raise DocumentError(
            "No extractable text found (a scanned PDF needs OCR, which is not supported)."
        )
    return pages