from pathlib import Path

from copilot.evaluation import EvalQuestion, normalize, retrieval_metrics, supports
from copilot.retrieval import BM25Retriever, Hit, HybridRetriever, reciprocal_rank_fusion, tokenize
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


def _question(source: str, facts: tuple[tuple[str, ...], ...]) -> EvalQuestion:
    return EvalQuestion("q", "X", "lookup", "?", "", source, facts, ())


def test_normalize_handles_pdf_spacing() -> None:
    assert normalize("Operating  Margin 23.7 %\nof $ 140 million") == "operating margin 23.7% of $140 million"


def test_supports_requires_every_fact_of_one_set() -> None:
    q = _question("x.pdf", (("46.3 billion", "51.4 billion"), ("46,309", "51,362")))
    assert supports("Revenues were $46.3 billion vs $51.4 billion", q)
    assert supports("Revenues $ 46,309 $ 51,362", q)
    assert not supports("Revenues were $46.3 billion", q)  # half a fact set is not evidence


def test_retrieval_metrics_are_evidence_level() -> None:
    nike, lulu = SAMPLE_CHUNKS[0], SAMPLE_CHUNKS[1]
    q1 = _question(nike.source, (("190 basis points",),))
    q2 = _question(lulu.source, (("not in any chunk",),))
    results = [(q1, [Hit(lulu, 1), Hit(nike, 1)]), (q2, [Hit(lulu, 1)])]
    m = retrieval_metrics(results, k=2)
    assert m["hit@1"] == 0.0
    assert m["hit@2"] == 0.5
    assert m["mrr"] == 0.25
    assert m["precision@2"] == 0.25


def test_bootstrap_ci_brackets_the_mean_and_narrows_with_more_questions() -> None:
    from copilot.evaluation import bootstrap_ci

    low, high = bootstrap_ci([1.0] * 17 + [0.0] * 8)  # 0.68 on 25 questions
    assert low < 0.68 < high and 0.3 < high - low < 0.45
    wide, narrow = bootstrap_ci([1.0, 0.0] * 10), bootstrap_ci([1.0, 0.0] * 200)
    assert narrow[1] - narrow[0] < wide[1] - wide[0]
    assert bootstrap_ci([1.0] * 5) == (1.0, 1.0)  # no variation, no uncertainty
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_detect_companies_by_name_and_brand() -> None:
    from copilot.retrieval import detect_companies

    assert detect_companies("How fast did HOKA grow?") == {"Deckers Brands"}
    assert detect_companies("Compare Nike and Under Armour margins") == {"Nike", "Under Armour"}
    assert detect_companies("What was Adidas's revenue?") == set()
    assert detect_companies("Is Nikel a brand?") == set()  # whole words only


def test_company_filter_restricts_every_retriever(sample_index: Path) -> None:
    from copilot.retrieval import VectorRetriever

    for retriever in (
        VectorRetriever(sample_index),
        BM25Retriever(sample_index),
        HybridRetriever(sample_index, candidates=5),
    ):
        hits = retriever.search("net revenue employees margin", k=3, companies={"Columbia Sportswear"})
        assert hits and {h.chunk.company for h in hits} == {"Columbia Sportswear"}


def test_company_scoped_retriever_passes_detected_companies() -> None:
    from copilot.retrieval import CompanyScopedRetriever

    seen: list[set[str] | None] = []

    class Recorder:
        cross_encoder_scores = True

        def search(self, query: str, k: int = 5, companies: set[str] | None = None) -> list[Hit]:
            seen.append(companies)
            return []

    scoped = CompanyScopedRetriever(Recorder())
    scoped.search("UGG sales growth")
    scoped.search("Adidas revenue")
    assert seen == [{"Deckers Brands"}, None]
    assert scoped.cross_encoder_scores is True
