"""Derive each question's supporting pages from its fact sets and check that
every question is answerable from at least one chunk of the index.

    python evals/label_pages.py          # rewrite `pages` in every question set and the comparison set
    python evals/label_pages.py --check  # exit 1 if `pages` is stale or a question has no evidence chunk
"""

import argparse
import json
import sys
from pathlib import Path

from copilot import config
from copilot.evaluation import (
    COMPARE_PATH,
    QUESTION_SETS,
    EvalQuestion,
    load_comparisons,
    load_questions,
    supports,
)
from copilot.ingest import chunk_document, read_pdf_pages


def derive(path: Path) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Map question id -> supporting pages, and question id -> number of evidence chunks."""
    return derive_for(load_questions(path))


def derive_comparisons(path: Path = COMPARE_PATH) -> tuple[dict[str, list[int]], dict[str, int]]:
    """The same as `derive`, for each report (part) of each comparison question, keyed 'id/company'."""
    return derive_for([part for q in load_comparisons(path) for part in q.parts])


def derive_for(questions: list[EvalQuestion]) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Supporting pages and evidence-chunk counts for any list of single-report questions."""
    pages: dict[str, list[int]] = {}
    chunk_counts: dict[str, int] = {}
    for source in sorted({q.source for q in questions}):
        report = config.DATA_DIR / source
        page_texts = read_pdf_pages(report)
        chunks = chunk_document(report, company="")
        for q in (q for q in questions if q.source == source):
            pages[q.id] = [n for n, text in enumerate(page_texts, start=1) if supports(text, q)]
            chunk_counts[q.id] = sum(supports(c.text, q) for c in chunks)
    return pages, chunk_counts


def main() -> None:
    """Rewrite or check the `pages` field of every question in every set."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    problems: list[str] = []
    updated: dict[Path, list[dict]] = {}
    for name, path in QUESTION_SETS.items():
        print(f"{name} set ({path.name})")
        pages, chunk_counts = derive(path)
        problems += [f"{qid}: no chunk contains a full fact set" for qid, n in chunk_counts.items() if n == 0]
        raw = json.loads(path.read_text(encoding="utf-8"))
        for item in raw:
            print(f"  {item['id']:8} pages {pages[item['id']]}  evidence chunks {chunk_counts[item['id']]}")
            if args.check and item["pages"] != pages[item["id"]]:
                problems.append(f"{item['id']}: stored pages {item['pages']} != derived {pages[item['id']]}")
            item["pages"] = pages[item["id"]]
        updated[path] = raw

    print(f"comparison set ({COMPARE_PATH.name})")
    part_pages, part_counts = derive_comparisons()
    problems += [f"{pid}: no chunk contains a full fact set" for pid, n in part_counts.items() if n == 0]
    raw_compare = json.loads(COMPARE_PATH.read_text(encoding="utf-8"))
    for item in raw_compare:
        for part in item["evidence"]:
            pid = f"{item['id']}/{part['company']}"
            print(f"  {pid:28} pages {part_pages[pid]}  evidence chunks {part_counts[pid]}")
            if args.check and part["pages"] != part_pages[pid]:
                problems.append(f"{pid}: stored pages {part['pages']} != derived {part_pages[pid]}")
            part["pages"] = part_pages[pid]
    updated[COMPARE_PATH] = raw_compare

    if problems:
        sys.exit("\n".join(problems))
    if not args.check:
        for path, raw in updated.items():
            path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
            print(f"Updated {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
