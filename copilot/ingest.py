"""Ingestion: PDF -> page text -> overlapping chunks -> embeddings in Chroma."""

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from copilot import config


@dataclass(frozen=True)
class Chunk:
    """A passage of one page of one report, the unit we embed and retrieve."""

    chunk_id: str
    source: str  # PDF file name, used in citations
    company: str
    page: int  # 1-based PDF page number
    text: str

    @property
    def header(self) -> str:
        """Context prepended before embedding so a page that never names the
        company (e.g. a bare financial table) is still tied to it."""
        return f"{self.company} annual report, page {self.page}."

    @property
    def report(self) -> str:
        """Report name with its fiscal-year end, e.g. 'Nike FY2025 Form 10-K (fiscal year ended May 31, 2025)'.
        Fiscal years differ between companies, so the model is shown which period a passage covers."""
        return config.DOCUMENTS.get(self.source, {}).get("report", self.source)

    @property
    def citation(self) -> str:
        """Source reference used in answers, e.g. 'Nike_FY2025_10K.pdf, p. 38'."""
        return f"{self.source}, p. {self.page}"


def clean_text(text: str) -> str:
    """Collapse whitespace and drop the 'Table of Contents' running header."""
    text = re.sub(r"\bTable\s+of\s+Contents\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_pdf_pages(path: Path) -> list[str]:
    """Return cleaned text for each page of a PDF (index 0 is page 1)."""
    return [clean_text(page.extract_text() or "") for page in PdfReader(path).pages]


def chunk_page(
    text: str, chunk_words: int = config.CHUNK_WORDS, overlap_words: int = config.OVERLAP_WORDS
) -> list[str]:
    """Split one page into windows of `chunk_words` that overlap by `overlap_words`."""
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")
    words = text.split()
    pieces: list[str] = []
    step = chunk_words - overlap_words
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start : start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return pieces


def chunk_document(path: Path, company: str, min_words: int = 20) -> list[Chunk]:
    """Chunk every page of a report. Pages with fewer than `min_words` words
    (covers, blank separators) are skipped."""
    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(read_pdf_pages(path), start=1):
        if len(page_text.split()) < min_words:
            continue
        for i, piece in enumerate(chunk_page(page_text)):
            chunks.append(Chunk(f"{path.stem}-p{page_number}-{i}", path.name, company, page_number, piece))
    return chunks


def load_corpus(data_dir: Path = config.DATA_DIR) -> list[Chunk]:
    """Chunk every configured report found in `data_dir`."""
    chunks: list[Chunk] = []
    for filename, meta in config.DOCUMENTS.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run: python scripts/download_data.py")
        chunks.extend(chunk_document(path, meta["company"]))
    return chunks


def index_settings() -> dict[str, str | int]:
    """Settings an index depends on. They are stored with the index when it is built and
    checked when it is opened, so a config change cannot silently reuse a stale index."""
    return {
        "embedding_model": config.EMBEDDING_MODEL,
        "chunk_words": config.CHUNK_WORDS,
        "overlap_words": config.OVERLAP_WORDS,
    }


def build_index(chunks: list[Chunk], chroma_dir: Path = config.CHROMA_DIR) -> int:
    """Embed chunks and (re)write the Chroma collection. Returns the chunk count."""
    import chromadb

    from copilot.embeddings import embed_passages

    client = chromadb.PersistentClient(path=str(chroma_dir))
    if config.COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(config.COLLECTION_NAME)
    collection = client.create_collection(config.COLLECTION_NAME, metadata={"hnsw:space": "cosine", **index_settings()})

    vectors = embed_passages([f"{c.header} {c.text}" for c in chunks])
    batch = 1000  # Chroma caps how many records one add() call accepts
    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        collection.add(
            ids=[c.chunk_id for c in part],
            embeddings=vectors[start : start + batch].tolist(),
            documents=[c.text for c in part],
            metadatas=[{"source": c.source, "company": c.company, "page": c.page} for c in part],
        )
    return len(chunks)
