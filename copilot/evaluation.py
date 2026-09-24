"""Evaluation helpers: the question set and page-level retrieval metrics.

Retrieval metrics need no LLM: a retrieved chunk is relevant when it comes from
the question's source report and one of its supporting pages. They run in CI on
every push. LLM-judged answer metrics (RAGAS) live in evals/run_ragas_eval.py.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from copilot import config
from copilot.retrieval import Hit

QUESTIONS_PATH = config.ROOT / "evals" / "questions.json"


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    company: str
    type: str             # lookup | explanation | calculation
    question: str
    expected_answer: str
    source: str           # PDF file name
    pages: tuple[int, ...]  # every page that supports the answer; first is primary
    evidence: str         # verbatim snippet from the primary page


def load_questions(path: Path = QUESTIONS_PATH) -> list[EvalQuestion]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [EvalQuestion(**{**item, "pages": tuple(item["pages"])}) for item in raw]


def is_relevant(hit: Hit, question: EvalQuestion) -> bool:
    return hit.chunk.source == question.source and hit.chunk.page in question.pages


def first_relevant_rank(hits: list[Hit], question: EvalQuestion) -> int | None:
    """1-based rank of the first relevant hit, or None if none is relevant."""
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(hit, question):
            return rank
    return None


def retrieval_metrics(results: list[tuple[EvalQuestion, list[Hit]]], k: int = config.TOP_K) -> dict[str, float]:
    """Aggregate page-level metrics over (question, ranked hits) pairs.

    hit@1 / hit@k  share of questions with a supporting page at rank 1 / in the top k
    mrr            mean reciprocal rank of the first supporting page (0 if absent)
    precision@k    share of the top-k chunks that come from a supporting page
    """
    n = len(results)
    if n == 0:
        return {"hit@1": 0.0, f"hit@{k}": 0.0, "mrr": 0.0, f"precision@{k}": 0.0}
    ranks = [first_relevant_rank(hits, q) for q, hits in results]
    return {
        "hit@1": sum(r == 1 for r in ranks) / n,
        f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n,
        "mrr": sum(1.0 / r for r in ranks if r is not None) / n,
        f"precision@{k}": sum(sum(is_relevant(h, q) for h in hits[:k]) / k for q, hits in results) / n,
    }
