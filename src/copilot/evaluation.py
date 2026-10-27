"""Evaluation helpers: the question set and evidence-level retrieval metrics.

Each question lists one or more alternative *fact sets*: short strings copied
from the report (e.g. "46.3 billion" + "51.4 billion", or the table figures
"46,309" + "51,362"). A retrieved chunk is relevant when its own text contains
every fact of at least one set, i.e. the model was actually shown the evidence.
This needs no LLM and runs in CI. The `pages` field (pages containing a fact set)
is derived from the facts by evals/label_pages.py and used to check citations.
LLM-judged answer metrics (RAGAS) live in evals/run_ragas_eval.py.

There are three question sets. "dev" (25 questions) was used to choose the retrieval
settings; "heldout" (20 questions about facts the dev set never asks about) was
written afterwards and is only scored, never tuned on, so it shows how well those
choices generalise. "hard" (20 more held-out questions, written for v1.6 before any
v1.6 change was measured) targets the known weak spots: figures that sit in tables,
answers that need two facts, and wording that differs from the report's. Like
"heldout", it is never tuned on. With sets this small, results carry bootstrap
confidence intervals.

A separate comparison set (evals/compare_questions.json) asks questions that need
evidence from two reports, such as which of two companies had the higher margin.
Each report is one *part* of the question, scored like a single-report question.
"""

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from copilot import config
from copilot.retrieval import Hit

QUESTIONS_PATH = config.ROOT / "evals" / "questions.json"
HELDOUT_PATH = config.ROOT / "evals" / "heldout_questions.json"
HARD_PATH = config.ROOT / "evals" / "hard_questions.json"
QUESTION_SETS = {"dev": QUESTIONS_PATH, "heldout": HELDOUT_PATH, "hard": HARD_PATH}
COMPARE_PATH = config.ROOT / "evals" / "compare_questions.json"


@dataclass(frozen=True)
class EvalQuestion:
    """One evaluation question with its reference answer and evidence."""

    id: str
    company: str
    type: str  # lookup | explanation | calculation
    question: str
    expected_answer: str
    source: str  # PDF file name
    facts: tuple[tuple[str, ...], ...]  # alternative fact sets; a chunk needs all facts of one set
    pages: tuple[int, ...]  # pages containing a fact set (derived from `facts`)


def load_questions(path: Path = QUESTIONS_PATH) -> list[EvalQuestion]:
    """Load the evaluation set from JSON."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalQuestion(**{**item, "facts": tuple(tuple(s) for s in item["facts"]), "pages": tuple(item["pages"])})
        for item in raw
    ]


def normalize(text: str) -> str:
    """Lowercase, straighten quotes, collapse whitespace and drop the spaces PDF
    extraction leaves in "23.7 %" and "$ 140", so facts match however they were typeset."""
    text = text.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\$ (?=\d)", "$", re.sub(r" %", "%", text))


def supports(text: str, question: EvalQuestion) -> bool:
    """True if `text` contains every fact of at least one of the question's fact sets."""
    norm = normalize(text)
    return any(all(normalize(fact) in norm for fact in fact_set) for fact_set in question.facts)


def is_relevant(hit: Hit, question: EvalQuestion) -> bool:
    """True if `hit` comes from the question's report and contains one of its fact sets."""
    return hit.chunk.source == question.source and supports(hit.chunk.text, question)


def first_relevant_rank(hits: list[Hit], question: EvalQuestion) -> int | None:
    """1-based rank of the first relevant hit, or None if none is relevant."""
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(hit, question):
            return rank
    return None


def retrieval_metrics(results: list[tuple[EvalQuestion, list[Hit]]], k: int = config.TOP_K) -> dict[str, float]:
    """Aggregate evidence-level metrics over (question, ranked hits) pairs.

    hit@1 / hit@k  share of questions with an evidence chunk at rank 1 / in the top k
    mrr            mean reciprocal rank of the first evidence chunk (0 if absent)
    precision@k    share of the top-k chunks that contain the evidence
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


def bootstrap_ci(
    values: Sequence[float], level: float = 0.95, samples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval for the mean of per-question `values`.

    Resamples the questions with replacement. Pass per-question differences between
    two pipelines (scored on the same questions) to get a paired interval for the gap:
    if it excludes 0, the difference is unlikely to be noise from this particular set."""
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(samples))
    tail = (1 - level) / 2
    return means[int(tail * samples)], means[int((1 - tail) * samples) - 1]


@dataclass(frozen=True)
class ComparisonQuestion:
    """A question that needs evidence from several reports: one part per report."""

    id: str
    question: str
    expected_answer: str
    figures: tuple[float, ...]  # numbers a correct answer states, such as both margins
    parts: tuple[EvalQuestion, ...]  # per report: its source, fact sets and supporting pages


def load_comparisons(path: Path = COMPARE_PATH) -> list[ComparisonQuestion]:
    """Load the comparison set from JSON; each evidence entry becomes one part."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        ComparisonQuestion(
            id=item["id"],
            question=item["question"],
            expected_answer=item["expected_answer"],
            figures=tuple(item["figures"]),
            parts=tuple(
                EvalQuestion(
                    id=f"{item['id']}/{part['company']}",
                    company=part["company"],
                    type="comparison",
                    question=item["question"],
                    expected_answer=item["expected_answer"],
                    source=part["source"],
                    facts=tuple(tuple(s) for s in part["facts"]),
                    pages=tuple(part["pages"]),
                )
                for part in item["evidence"]
            ),
        )
        for item in raw
    ]


def covers_all_parts(hits: list[Hit], question: ComparisonQuestion, k: int) -> bool:
    """True if the top `k` hits hold evidence for every report the question needs."""
    return all(first_relevant_rank(hits[:k], part) is not None for part in question.parts)


def states_figures(answer: str, figures: Sequence[float]) -> bool:
    """True if `answer` states every expected figure.

    Signs are ignored ("fell 1.3%" states 1.3), and a figure matches within 0.2 or 0.2% of
    its value, whichever is larger, so rounded percentages and exact counts both pass."""
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", answer.replace(",", ""))]
    return all(any(abs(n - abs(f)) <= max(0.2, 0.002 * abs(f)) for n in numbers) for f in figures)
