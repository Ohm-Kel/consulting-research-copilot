"""Stage 0.4: extract text from a PDF and split it into overlapping chunks.

Concept: models and embeddings work best on passages of a few hundred words,
so long documents are cut into chunks. Overlap keeps a sentence that straddles
a boundary intact in at least one chunk. We record the page number of each
chunk so answers can cite their source later.

Run: python stage0/04_chunk_pdf.py [path/to/file.pdf]
"""

import sys
from pathlib import Path

from pypdf import PdfReader

CHUNK_WORDS = 500
OVERLAP_WORDS = 50


def chunk_pdf(path: Path) -> list[dict]:
    """Return a list of {"page": int, "text": str} chunks, chunked page by page."""
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for page_number, page in enumerate(PdfReader(path).pages, start=1):
        words = (page.extract_text() or "").split()
        for start in range(0, len(words), step):
            piece = words[start : start + CHUNK_WORDS]
            chunks.append({"page": page_number, "text": " ".join(piece)})
            if start + CHUNK_WORDS >= len(words):
                break
    return chunks


if __name__ == "__main__":
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else next(Path("data").glob("*.pdf"))
    chunks = chunk_pdf(pdf_path)
    print(f"File: {pdf_path.name}")
    print(f"Chunks: {len(chunks)}")
    print(f"\nFirst chunk (page {chunks[0]['page']}):\n{chunks[0]['text'][:800]}...")
