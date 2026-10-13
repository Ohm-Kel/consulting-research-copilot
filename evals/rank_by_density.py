"""Does reranking move number-dense (table-like) evidence up or down, compared with hybrid search?

Reads the per-question ranks saved by run_retrieval_eval.py, measures how much of each question's
evidence chunk is numbers, and lists the rank change from `hybrid` to `hybrid_rerank`.

    python evals/rank_by_density.py              # development set
    python evals/rank_by_density.py --set hard   # harder held-out set (look only, never tune on it)
"""

import argparse
import json
import math
import re

from copilot import config
from copilot.evaluation import QUESTION_SETS, load_questions, supports
from copilot.ingest import Chunk, chunk_document

NUMBER = re.compile(r"[$(]?-?[\d,.]+%?\)?")
DENSE = 0.3  # share of number tokens above which a chunk reads as a table


def numeric_share(text: str) -> float:
    """Share of whitespace-separated tokens that are numbers, amounts or percentages."""
    tokens = text.split()
    return sum(bool(NUMBER.fullmatch(t)) for t in tokens) / max(len(tokens), 1)


def rank_change(before: int | None, after: int | None) -> str:
    """Compare two 1-based ranks; None means the evidence was not in the ranked list."""
    if before is None and after is None:
        return "missed by both"
    b, a = before or math.inf, after or math.inf
    return "better" if a < b else "worse" if a > b else "same"


def main() -> None:
    """Print each question's evidence density and rank change, then a summary for dense questions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", dest="question_set", default="dev", choices=QUESTION_SETS)
    args = parser.parse_args()

    questions = load_questions(QUESTION_SETS[args.question_set])
    name = "retrieval.json" if args.question_set == "dev" else f"retrieval_{args.question_set}.json"
    rows = {r["mode"]: r for r in json.loads((config.ROOT / "evals" / "results" / name).read_text(encoding="utf-8"))}
    chunks: dict[str, list[Chunk]] = {
        source: chunk_document(config.DATA_DIR / source, company="") for source in {q.source for q in questions}
    }

    changes: dict[str, int] = {"better": 0, "worse": 0, "same": 0, "missed by both": 0}
    for q in questions:
        share = max(numeric_share(c.text) for c in chunks[q.source] if supports(c.text, q))
        before, after = rows["hybrid"]["ranks"].get(q.id), rows["hybrid_rerank"]["ranks"].get(q.id)
        change = rank_change(before, after)
        if share >= DENSE:
            changes[change] += 1
        print(f"{q.id:8} numbers {share:.2f}  hybrid rank {before}  reranked rank {after}  {change}")
    print(f"\nEvidence with at least {DENSE:.0%} numbers, reranked vs hybrid: {changes}")


if __name__ == "__main__":
    main()
