"""The answer key must match the reports: every question's facts appear in at
least one chunk, and the stored supporting pages are exactly the pages that
contain a fact set (regenerate with `python evals/label_pages.py`)."""

import json

import pytest

from copilot.evaluation import QUESTIONS_PATH, load_questions


def test_eval_set_is_well_formed() -> None:
    questions = load_questions()
    assert len(questions) == 25
    assert len({q.id for q in questions}) == 25
    assert all(q.facts and all(q.facts) and q.expected_answer and q.pages for q in questions)
    assert {q.type for q in questions} == {"lookup", "explanation", "calculation"}


def test_facts_match_the_reports(data_available: bool) -> None:
    if not data_available:
        pytest.skip("reports not downloaded")
    import sys

    from copilot import config

    sys.path.insert(0, str(config.ROOT / "evals"))
    from label_pages import derive

    pages, chunk_counts = derive()
    assert all(n > 0 for n in chunk_counts.values()), chunk_counts
    stored = {item["id"]: item["pages"] for item in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))}
    assert stored == pages
