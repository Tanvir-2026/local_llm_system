"""Problem 4: chunking and loading (no external services)."""
from app.rag_system.chunker import chunk_pages, split_text
from app.rag_system.loader import DocumentError, load_document


def test_short_text_is_one_chunk():
    assert split_text("Hello world, this is short.", 1000, 150) == ["Hello world, this is short."]


def test_long_text_is_split_with_overlap_and_nothing_lost():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = split_text(text, 300, 50)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    # every sentence appears in at least one chunk
    for i in range(200):
        assert any(f"Sentence number {i} is here." in c for c in chunks), i


def test_chunk_pages_keeps_page_numbers():
    chunks = chunk_pages([(1, "Page one has some real text in it."), (2, "Page two has other text here too.")])
    assert [c.page for c in chunks] == [1, 2]
    assert [c.index for c in chunks] == [0, 1]


def test_load_txt_and_errors():
    pages = load_document("a.txt", "Hello there".encode())
    assert pages == [(None, "Hello there")]
    for name, data in (("a.docx", b"x"), ("empty.txt", b"   \n ")):
        try:
            load_document(name, data)
            assert False, "should have raised"
        except DocumentError:
            pass