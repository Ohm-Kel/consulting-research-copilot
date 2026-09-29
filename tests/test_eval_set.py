"""The answer keys must match the reports: every question's facts appear in at
least one chunk, and the stored supporting pages are exactly the pages that
contain a fact set (regenerate with `python evals/label_pages.py`)."""

import json

import pytest

from copilot.evaluation import QUESTION_SETS, load_questions


@pytest.mark.parametrize(("name", "size"), [("dev", 25), ("heldout", 20)])
def test_eval_set_is_well_formed(name: str, size: int) -> None:
    questions = load_questions(QUESTION_SETS[name])
    assert len(questions) == size
    assert len({q.id for q in questions}) == size
    assert all(q.facts and all(q.facts) and q.expected_answer and q.pages for q in questions)
    assert {q.type for q in questions} == {"lookup", "explanation", "calculation"}


def test_heldout_questions_are_new() -> None:
    dev, heldout = (load_questions(QUESTION_SETS[name]) for name in ("dev", "heldout"))
    assert not {q.id for q in dev} & {q.id for q in heldout}
    assert not {q.question for q in dev} & {q.question for q in heldout}


@pytest.mark.parametrize("name", QUESTION_SETS)
def test_facts_match_the_reports(name: str, data_available: bool) -> None:
    if not data_available:
        pytest.skip("reports not downloaded")
    import sys

    from copilot import config

    sys.path.insert(0, str(config.ROOT / "evals"))
    from label_pages import derive

    path = QUESTION_SETS[name]
    pages, chunk_counts = derive(path)
    assert all(n > 0 for n in chunk_counts.values()), chunk_counts
    stored = {item["id"]: item["pages"] for item in json.loads(path.read_text(encoding="utf-8"))}
    assert stored == pages
