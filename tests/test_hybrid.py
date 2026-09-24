from pathlib import Path

from copilot.evaluation import EvalQuestion, load_questions, retrieval_metrics
from copilot.retrieval import BM25Retriever, HybridRetriever, Hit, reciprocal_rank_fusion, tokenize
from tests.conftest import SAMPLE_CHUNKS


def test_tokenize_normalises_numbers() -> None:
    assert tokenize("Sales rose to $4,689 million (42.7%).") == ["sales", "rose", "to", "4689", "million", "42.7"]


def test_bm25_matches_exact_brand_name(sample_index: Path) -> None:
    assert BM25Retriever(sample_index).search("HOKA", k=1)[0].chunk.chunk_id == "deck-p40-0"


def test_rrf_rewards_chunks_ranked_well_by_both_lists() -> None:
    a, b, c = SAMPLE_CHUNKS[:3]
    fused = reciprocal_rank_fusion([[Hit(a, 1), Hit(b, 1)], [Hit(b, 1), Hit(c, 1)]])
    assert [h.chunk.chunk_id for h in fused][0] == b.chunk_id
    assert len(fused) == 3


def test_hybrid_returns_k_unique_hits(sample_index: Path) -> None:
    hits = HybridRetriever(sample_index, candidates=5).search("employees", k=3)
    assert len({h.chunk.chunk_id for h in hits}) == 3


def _question(source: str, pages: tuple[int, ...]) -> EvalQuestion:
    return EvalQuestion("q", "X", "lookup", "?", "", source, pages, "")


def test_retrieval_metrics() -> None:
    nike, lulu = SAMPLE_CHUNKS[0], SAMPLE_CHUNKS[1]
    q1 = _question(nike.source, (38,))
    q2 = _question(lulu.source, (99,))  # never retrieved
    results = [(q1, [Hit(lulu, 1), Hit(nike, 1)]), (q2, [Hit(nike, 1)])]
    m = retrieval_metrics(results, k=2)
    assert m["hit@1"] == 0.0
    assert m["hit@2"] == 0.5
    assert m["mrr"] == 0.25
    assert m["precision@2"] == 0.25


def test_eval_set_is_well_formed() -> None:
    questions = load_questions()
    assert len(questions) == 25
    assert len({q.id for q in questions}) == 25
    assert all(q.pages and q.evidence and q.expected_answer for q in questions)
