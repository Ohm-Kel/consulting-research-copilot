from pathlib import Path

import pytest

from copilot.retrieval import (
    RETRIEVER_MODES,
    BM25Retriever,
    CompanyScopedRetriever,
    HybridRetriever,
    RerankedRetriever,
    VectorRetriever,
    build_retriever,
)


@pytest.mark.parametrize(
    ("question", "expected_id"),
    [
        ("Why did Nike's gross margin fall?", "nike-p38-0"),
        ("How many employees does Columbia have?", "colm-p5-0"),
        ("How fast did HOKA grow?", "deck-p40-0"),
    ],
)
def test_vector_search_ranks_relevant_chunk_first(sample_index: Path, question: str, expected_id: str) -> None:
    hits = VectorRetriever(sample_index).search(question, k=3)
    assert hits[0].chunk.chunk_id == expected_id


def test_vector_search_returns_k_scored_hits_with_metadata(sample_index: Path) -> None:
    hits = VectorRetriever(sample_index).search("revenue growth", k=3)
    assert len(hits) == 3
    assert hits[0].score >= hits[1].score >= hits[2].score
    assert all(h.chunk.page > 0 and h.chunk.source.endswith(".pdf") for h in hits)


def test_missing_index_gives_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="copilot.cli ingest"):
        VectorRetriever(tmp_path / "empty")


def test_reranked_retriever_puts_the_relevant_chunk_first(sample_index: Path) -> None:
    # The default pipeline: hybrid candidates re-scored by the cross-encoder.
    retriever = RerankedRetriever(HybridRetriever(sample_index, candidates=5), candidates=5)
    hits = retriever.search("Why did Nike's gross margin fall?", k=3)
    assert hits[0].chunk.chunk_id == "nike-p38-0"
    assert len(hits) == 3 and hits[0].score >= hits[1].score >= hits[2].score
    assert retriever.cross_encoder_scores  # the scale the relevance floor was calibrated on


def test_reranked_scores_separate_relevant_from_unrelated(sample_index: Path) -> None:
    retriever = RerankedRetriever(HybridRetriever(sample_index, candidates=5), candidates=5)
    relevant = retriever.search("How fast did HOKA brand net sales grow?", k=1)[0].score
    unrelated = retriever.search("Who won the 2022 FIFA World Cup?", k=1)[0].score
    assert relevant > unrelated


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("vector", VectorRetriever),
        ("bm25", BM25Retriever),
        ("hybrid", HybridRetriever),
        ("hybrid_rerank", RerankedRetriever),
        ("hybrid_rerank_company", CompanyScopedRetriever),
    ],
)
def test_build_retriever_returns_each_mode(sample_index: Path, mode: str, expected: type) -> None:
    assert mode in RETRIEVER_MODES
    assert isinstance(build_retriever(mode, sample_index), expected)


def test_build_retriever_rejects_unknown_mode(sample_index: Path) -> None:
    with pytest.raises(ValueError, match="unknown retriever mode"):
        build_retriever("magic", sample_index)
