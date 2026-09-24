from pathlib import Path

import pytest

from copilot.retrieval import VectorRetriever


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
