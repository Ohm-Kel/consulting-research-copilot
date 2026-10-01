from pathlib import Path
from types import SimpleNamespace

import pytest

from copilot.ingest import Chunk
from copilot.retrieval import (
    RETRIEVER_MODES,
    BM25Retriever,
    CompanyScopedRetriever,
    Hit,
    HybridRetriever,
    RerankedRetriever,
    VectorRetriever,
    build_retriever,
    passage_windows,
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


def test_passage_windows_overlap_by_half_and_cover_the_text() -> None:
    words = [f"w{i}" for i in range(10)]
    assert passage_windows(" ".join(words), 4) == ["w0 w1 w2 w3", "w2 w3 w4 w5", "w4 w5 w6 w7", "w6 w7 w8 w9"]
    assert passage_windows(" ".join(words[:9]), 4)[-1] == "w5 w6 w7 w8"  # last window flush with the end
    assert passage_windows("a short passage", 4) == ["a short passage"]


class StubCrossEncoder:
    """Scores 5.0 for a passage containing 'needle', else 0.0; records what it was shown."""

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs += pairs
        return [5.0 if "needle" in passage else 0.0 for _, passage in pairs]


def test_reranker_scores_each_chunk_by_its_best_window(monkeypatch: pytest.MonkeyPatch) -> None:
    from copilot import retrieval

    needle = Chunk("a-p1-0", "a.pdf", "A", 1, " ".join(["filler"] * 250 + ["needle"] + ["filler"] * 49))
    plain = Chunk("b-p1-0", "b.pdf", "B", 1, " ".join(["filler"] * 300))
    first_stage = SimpleNamespace(search=lambda query, k, companies=None: [Hit(plain, 1.0), Hit(needle, 0.5)])
    stub = StubCrossEncoder()
    monkeypatch.setattr(retrieval, "get_reranker", lambda: stub)

    hits = RerankedRetriever(first_stage, candidates=2, window_words=128).search("q", k=2)
    assert [(h.chunk.chunk_id, h.score) for h in hits] == [("a-p1-0", 5.0), ("b-p1-0", 0.0)]
    header_words = len(needle.header.split())
    assert len(stub.pairs) > 2 and all(len(p.split()) <= 128 + header_words for _, p in stub.pairs)

    stub.pairs.clear()
    RerankedRetriever(first_stage, candidates=2, window_words=None).search("q", k=2)
    assert len(stub.pairs) == 2  # whole chunks: one pair per candidate


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("vector", VectorRetriever),
        ("bm25", BM25Retriever),
        ("hybrid", HybridRetriever),
        ("hybrid_rerank_whole", RerankedRetriever),
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


def test_index_records_its_settings(sample_index: Path) -> None:
    from copilot.ingest import index_settings
    from copilot.retrieval import _open_collection

    metadata = _open_collection(sample_index).metadata or {}
    assert all(metadata[key] == value for key, value in index_settings().items())


def test_stale_index_is_rejected(sample_index: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from copilot import config

    monkeypatch.setattr(config, "CHUNK_WORDS", config.CHUNK_WORDS + 100)
    with pytest.raises(RuntimeError, match="out of date.*chunk_words"):
        VectorRetriever(sample_index)
