"""Retrieval: find the chunks most relevant to a question.

Stage 1: VectorRetriever (dense embeddings in Chroma).
Stage 2: BM25Retriever (keywords), HybridRetriever (reciprocal rank fusion of
both), and RerankedRetriever (a cross-encoder re-scores the fused candidates).
CompanyScopedRetriever restricts any of them to the companies a query names.

Every retriever's search() accepts `companies`: when given, only chunks from
those companies are considered.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import chromadb
from rank_bm25 import BM25Okapi

from copilot import config
from copilot.embeddings import embed_query
from copilot.ingest import Chunk, index_settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk and its relevance score (higher is better; scales
    differ between retrievers)."""

    chunk: Chunk
    score: float


class Retriever(Protocol):
    """Interface shared by all retrievers."""

    def search(self, query: str, k: int = config.TOP_K, companies: set[str] | None = None) -> list[Hit]:
        """Return the `k` best hits for `query`, optionally limited to `companies`."""


def _chunk(chunk_id: str, text: str, meta: Mapping[str, Any]) -> Chunk:
    """Rebuild a Chunk from a Chroma record and its metadata."""
    return Chunk(chunk_id, str(meta["source"]), str(meta["company"]), int(meta["page"]), text)


def _open_collection(chroma_dir: Path) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(chroma_dir))
    if config.COLLECTION_NAME not in [c.name for c in client.list_collections()]:
        raise RuntimeError(f"No index found in {chroma_dir}. Run: python -m copilot.cli ingest")
    collection = client.get_collection(config.COLLECTION_NAME)
    stored = collection.metadata or {}
    stale = {key: (stored.get(key), value) for key, value in index_settings().items() if stored.get(key) != value}
    if stale:
        details = ", ".join(f"{key}: index has {old!r}, config has {new!r}" for key, (old, new) in stale.items())
        raise RuntimeError(f"The index is out of date ({details}). Rebuild it: python -m copilot.cli ingest")
    return collection


class VectorRetriever:
    """Dense retrieval: cosine similarity between the query and chunk embeddings."""

    def __init__(self, chroma_dir: Path = config.CHROMA_DIR) -> None:
        self.collection = _open_collection(chroma_dir)

    def search(self, query: str, k: int = config.TOP_K, companies: set[str] | None = None) -> list[Hit]:
        """Return the `k` chunks closest to the query embedding."""
        where = cast(chromadb.Where, {"company": {"$in": sorted(companies)}}) if companies else None
        result = self.collection.query(query_embeddings=[embed_query(query).tolist()], n_results=k, where=where)
        documents, metadatas, distances = result["documents"], result["metadatas"], result["distances"]
        if documents is None or metadatas is None or distances is None:
            raise RuntimeError("Chroma returned no documents; rebuild the index with: python -m copilot.cli ingest")
        return [
            Hit(_chunk(chunk_id, text, meta), 1.0 - distance)  # Chroma returns cosine distance
            for chunk_id, text, meta, distance in zip(
                result["ids"][0], documents[0], metadatas[0], distances[0], strict=True
            )
        ]


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens. '4,689' and '4689' both yield '4689'."""
    return re.findall(r"[a-z0-9]+(?:[.%][a-z0-9]+)*", text.lower().replace(",", ""))


class BM25Retriever:
    """Keyword retrieval with BM25 over the same chunks stored in Chroma.
    Strong where embeddings are weak: exact names, figures and jargon."""

    def __init__(self, chroma_dir: Path = config.CHROMA_DIR) -> None:
        records = _open_collection(chroma_dir).get(include=["documents", "metadatas"])
        documents, metadatas = records["documents"], records["metadatas"]
        if documents is None or metadatas is None:
            raise RuntimeError("Chroma returned no documents; rebuild the index with: python -m copilot.cli ingest")
        self.chunks = [
            _chunk(cid, text, meta) for cid, text, meta in zip(records["ids"], documents, metadatas, strict=True)
        ]
        self.bm25 = BM25Okapi([tokenize(f"{c.header} {c.text}") for c in self.chunks])

    def search(self, query: str, k: int = config.TOP_K, companies: set[str] | None = None) -> list[Hit]:
        """Return the `k` chunks with the highest BM25 score for the query terms."""
        scores = self.bm25.get_scores(tokenize(query))
        allowed = [i for i in range(len(scores)) if not companies or self.chunks[i].company in companies]
        best = sorted(allowed, key=lambda i: scores[i], reverse=True)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in best]


def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Merge ranked lists: score = sum of 1 / (k + rank). Uses ranks only, so
    retrievers with incomparable score scales can be combined."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk.chunk_id] = scores.get(hit.chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[hit.chunk.chunk_id] = hit.chunk
    return [Hit(chunks[cid], s) for cid, s in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


class HybridRetriever:
    """Vector + BM25 candidates combined with reciprocal rank fusion."""

    def __init__(self, chroma_dir: Path = config.CHROMA_DIR, candidates: int = 30) -> None:
        self.vector = VectorRetriever(chroma_dir)
        self.bm25 = BM25Retriever(chroma_dir)
        self.candidates = candidates

    def search(self, query: str, k: int = config.TOP_K, companies: set[str] | None = None) -> list[Hit]:
        """Fuse the vector and BM25 candidate lists and return the top `k`."""
        fused = reciprocal_rank_fusion(
            [self.vector.search(query, self.candidates, companies), self.bm25.search(query, self.candidates, companies)]
        )
        return fused[:k]


@lru_cache(maxsize=1)
def get_reranker() -> "CrossEncoder":
    """Load the cross-encoder once per process."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANKER_MODEL, max_length=512)


class RerankedRetriever:
    """Re-scores the top candidates of a first-stage retriever with a
    cross-encoder, which reads the query and passage together. Too slow for
    the whole corpus, accurate on a short list."""

    cross_encoder_scores = True  # scores are on the scale RELEVANCE_THRESHOLD was calibrated on

    def __init__(self, first_stage: Retriever, candidates: int = config.RERANK_CANDIDATES) -> None:
        self.first_stage = first_stage
        self.candidates = candidates

    def search(self, query: str, k: int = config.TOP_K, companies: set[str] | None = None) -> list[Hit]:
        """Re-score the first-stage candidates with the cross-encoder and return the top `k`."""
        candidates = self.first_stage.search(query, self.candidates, companies)
        if not candidates:
            return []
        scores = get_reranker().predict([(query, f"{h.chunk.header} {h.chunk.text}") for h in candidates])
        ranked = sorted(zip(scores, candidates, strict=True), key=lambda pair: pair[0], reverse=True)
        return [Hit(hit.chunk, float(score)) for score, hit in ranked[:k]]


# Names, brands and subsidiaries that identify each company in a question.
COMPANY_ALIASES: dict[str, tuple[str, ...]] = {
    "Nike": ("nike", "converse", "jordan brand"),
    "Lululemon": ("lululemon",),
    "Under Armour": ("under armour", "underarmour"),
    "Columbia Sportswear": ("columbia", "sorel", "prana", "mountain hardwear"),
    "Deckers Brands": ("deckers", "ugg", "hoka", "teva", "koolaburra"),
}


def detect_companies(query: str) -> set[str]:
    """Companies whose name or brand appears in the query (whole words only)."""
    text = query.lower()
    return {
        company
        for company, aliases in COMPANY_ALIASES.items()
        if any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases)
    }


class CompanyScopedRetriever:
    """Searches only the reports of the companies a query names, so a question
    about Deckers cannot be answered from a similar-sounding Columbia passage.
    Queries naming no covered company (e.g. "Adidas revenue") search everything."""

    def __init__(self, inner: Retriever) -> None:
        self.inner = inner
        self.cross_encoder_scores = getattr(inner, "cross_encoder_scores", False)

    def search(self, query: str, k: int = config.TOP_K, companies: set[str] | None = None) -> list[Hit]:
        """Search only the companies named in the query, unless `companies` is given."""
        return self.inner.search(query, k, companies or detect_companies(query) or None)


RETRIEVER_MODES = ("vector", "bm25", "hybrid", "hybrid_rerank", "hybrid_rerank_company")


def build_retriever(mode: str = config.RETRIEVER_MODE, chroma_dir: Path = config.CHROMA_DIR) -> Retriever:
    """Factory for the retrieval pipeline named by `mode`."""
    if mode == "vector":
        return VectorRetriever(chroma_dir)
    if mode == "bm25":
        return BM25Retriever(chroma_dir)
    if mode == "hybrid":
        return HybridRetriever(chroma_dir)
    if mode == "hybrid_rerank":
        return RerankedRetriever(HybridRetriever(chroma_dir))
    if mode == "hybrid_rerank_company":
        return CompanyScopedRetriever(RerankedRetriever(HybridRetriever(chroma_dir)))
    raise ValueError(f"unknown retriever mode {mode!r}; choose from {RETRIEVER_MODES}")
