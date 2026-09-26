"""Derive each question's supporting pages from its fact sets and check that
every question is answerable from at least one chunk of the index.

    python evals/label_pages.py          # rewrite `pages` in evals/questions.json
    python evals/label_pages.py --check  # exit 1 if `pages` is stale or a question has no evidence chunk
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.evaluation import QUESTIONS_PATH, load_questions, supports  # noqa: E402
from copilot.ingest import chunk_document, read_pdf_pages  # noqa: E402


def derive() -> tuple[dict[str, list[int]], dict[str, int]]:
    """Map question id -> supporting pages, and question id -> number of evidence chunks."""
    questions = load_questions()
    pages: dict[str, list[int]] = {}
    chunk_counts: dict[str, int] = {}
    for source in sorted({q.source for q in questions}):
        path = config.DATA_DIR / source
        page_texts = read_pdf_pages(path)
        chunks = chunk_document(path, company="")
        for q in (q for q in questions if q.source == source):
            pages[q.id] = [n for n, text in enumerate(page_texts, start=1) if supports(text, q)]
            chunk_counts[q.id] = sum(supports(c.text, q) for c in chunks)
    return pages, chunk_counts


def main() -> None:
    """Rewrite or check the `pages` field of every question."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    pages, chunk_counts = derive()
    problems = [f"{qid}: no chunk contains a full fact set" for qid, n in chunk_counts.items() if n == 0]
    raw = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    for item in raw:
        print(f"{item['id']:8} pages {pages[item['id']]}  evidence chunks {chunk_counts[item['id']]}")
        if args.check and item["pages"] != pages[item["id"]]:
            problems.append(f"{item['id']}: stored pages {item['pages']} != derived {pages[item['id']]}")
        item["pages"] = pages[item["id"]]

    if problems:
        sys.exit("\n".join(problems))
    if not args.check:
        QUESTIONS_PATH.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {QUESTIONS_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
