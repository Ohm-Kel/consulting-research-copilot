"""The answer key must match the reports: each evidence snippet has to appear
verbatim on the question's primary page."""

import pytest

from copilot import config
from copilot.evaluation import load_questions
from copilot.ingest import read_pdf_pages


def test_evidence_appears_on_primary_page(data_available: bool) -> None:
    if not data_available:
        pytest.skip("reports not downloaded")
    pages_by_file: dict[str, list[str]] = {}
    for q in load_questions():
        if q.source not in pages_by_file:
            pages_by_file[q.source] = read_pdf_pages(config.DATA_DIR / q.source)
        assert q.evidence in pages_by_file[q.source][q.pages[0] - 1], q.id
