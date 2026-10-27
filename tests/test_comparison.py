"""The comparison set: questions that need evidence from two reports."""

import json
import sys

import pytest

from copilot import config
from copilot.evaluation import COMPARE_PATH, ComparisonQuestion, covers_all_parts, load_comparisons, states_figures
from copilot.ingest import Chunk
from copilot.retrieval import Hit


def test_comparison_set_is_well_formed() -> None:
    questions = load_comparisons()
    assert len(questions) == 7
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        assert len(q.parts) >= 2 and len({p.source for p in q.parts}) == len(q.parts)
        assert all(p.source in config.DOCUMENTS and p.facts and p.pages for p in q.parts)
        assert len(q.figures) >= 2 and q.expected_answer


def test_comparison_facts_match_the_reports(data_available: bool) -> None:
    if not data_available:
        pytest.skip("reports not downloaded")
    sys.path.insert(0, str(config.ROOT / "evals"))
    from label_pages import derive_comparisons

    pages, chunk_counts = derive_comparisons()
    assert all(n > 0 for n in chunk_counts.values()), chunk_counts
    stored = {
        f"{item['id']}/{part['company']}": part["pages"]
        for item in json.loads(COMPARE_PATH.read_text(encoding="utf-8"))
        for part in item["evidence"]
    }
    assert stored == pages


def _hit(source: str, text: str) -> Hit:
    return Hit(Chunk(f"{source}-1", source, "", 1, text), 1.0)


def test_covers_all_parts_needs_evidence_from_every_report() -> None:
    question = load_comparisons()[0]  # Nike vs Lululemon headcount
    nike = _hit("Nike_FY2025_10K.pdf", "We had approximately 77,800 employees worldwide.")
    lulu = _hit("Lululemon_FY2024_10K.pdf", "We employed approximately 39,000 people.")
    other = _hit("Deckers_FY2025_AR.pdf", "Approximately 5,500 employees.")
    assert covers_all_parts([nike, other, lulu], question, k=3)
    assert not covers_all_parts([nike, other, lulu], question, k=2)  # Lululemon's evidence is third
    assert not covers_all_parts([nike, nike, other], question, k=3)


def test_covers_all_parts_ignores_evidence_text_from_the_wrong_report() -> None:
    question = load_comparisons()[0]
    wrong_report = _hit("Deckers_FY2025_AR.pdf", "approximately 39,000 people")
    nike = _hit("Nike_FY2025_10K.pdf", "77,800 employees")
    assert not covers_all_parts([nike, wrong_report], question, k=2)


@pytest.mark.parametrize(
    ("answer", "figures", "expected"),
    [
        ("Nike had about 77,800 employees; Lululemon had approximately 39,000.", (77800, 39000), True),
        ("Under Armour's inventories fell 1.3% while Deckers' rose 4.4%.", (1.3, 4.4), True),  # signs ignored
        ("Lululemon's gross margin was 59.2% against 50.2% at Columbia.", (59.2, 50.2), True),
        ("Footwear was about 23% at Under Armour and 66% at the NIKE Brand.", (23.4, 66.0), False),  # 23 is off
        ("Lululemon's gross margin was 59.2%.", (59.2, 50.2), False),  # one company missing
    ],
)
def test_states_figures(answer: str, figures: tuple[float, ...], expected: bool) -> None:
    assert states_figures(answer, figures) is expected


def test_comparison_question_parts_keep_their_own_evidence() -> None:
    q = load_comparisons()[1]
    assert isinstance(q, ComparisonQuestion)
    assert {p.company for p in q.parts} == {"Nike", "Columbia Sportswear"}
    assert all(p.id.startswith(q.id + "/") for p in q.parts)
