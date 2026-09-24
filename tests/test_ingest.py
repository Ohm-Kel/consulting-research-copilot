import pytest
from pypdf import PdfReader

from copilot import config
from copilot.ingest import chunk_document, chunk_page, clean_text


def test_chunk_page_windows_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(700))
    pieces = chunk_page(text, chunk_words=300, overlap_words=50)
    assert [len(p.split()) for p in pieces] == [300, 300, 200]
    # the last 50 words of one chunk open the next one
    assert pieces[0].split()[-50:] == pieces[1].split()[:50]
    assert pieces[-1].split()[-1] == "w699"


def test_chunk_page_short_page_is_one_chunk() -> None:
    assert chunk_page("only a few words here", chunk_words=300, overlap_words=50) == ["only a few words here"]


def test_chunk_page_rejects_overlap_not_smaller_than_chunk() -> None:
    with pytest.raises(ValueError):
        chunk_page("a b c", chunk_words=50, overlap_words=50)


def test_clean_text_collapses_whitespace_and_running_header() -> None:
    assert clean_text("Table of Contents\nNet   revenues\n\nincreased") == "Net revenues increased"


def test_chunk_document_on_real_report(data_available: bool) -> None:
    if not data_available:
        pytest.skip("reports not downloaded; run scripts/download_data.py")
    path = config.DATA_DIR / "Nike_FY2025_10K.pdf"
    chunks = chunk_document(path, "Nike")
    page_count = len(PdfReader(path).pages)

    assert len(chunks) > 100
    assert all(1 <= c.page <= page_count for c in chunks)
    assert all(len(c.text.split()) <= config.CHUNK_WORDS for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    gross_margin_pages = {c.page for c in chunks if "gross margin decreased 190 basis points" in c.text.lower()}
    assert 38 in gross_margin_pages
    assert chunks[0].citation.startswith("Nike_FY2025_10K.pdf, p. ")
