"""Compare retrieval pipelines on a question set. No API key needed.

python evals/run_retrieval_eval.py                  # dev set, all modes, writes evals/results/retrieval.json
python evals/run_retrieval_eval.py --set heldout    # held-out set, writes evals/results/retrieval_heldout.json
python evals/run_retrieval_eval.py --modes hybrid_rerank --min-hit 0.8   # CI regression gate
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.evaluation import (  # noqa: E402
    QUESTION_SETS,
    EvalQuestion,
    bootstrap_ci,
    first_relevant_rank,
    load_questions,
    retrieval_metrics,
)
from copilot.retrieval import RETRIEVER_MODES, build_retriever  # noqa: E402

RESULTS_DIR = config.ROOT / "evals" / "results"


def found(rank: int | None, k: int) -> float:
    """1.0 if the evidence was ranked within the top `k`, else 0.0."""
    return float(rank is not None and rank <= k)


def evaluate(mode: str, k: int, questions: list[EvalQuestion]) -> dict:
    """Score one retrieval pipeline on a question set."""
    retriever = build_retriever(mode)
    start = time.perf_counter()
    results = [(q, retriever.search(q.question, k=max(k, 10))) for q in questions]
    seconds = time.perf_counter() - start
    metrics = retrieval_metrics(results, k=k)
    ranks = {q.id: first_relevant_rank(hits, q) for q, hits in results}
    low, high = bootstrap_ci([found(r, k) for r in ranks.values()])
    return {
        "mode": mode,
        **{m: round(v, 3) for m, v in metrics.items()},
        f"hit@{k}_ci95": [round(low, 2), round(high, 2)],
        "sec_per_query": round(seconds / len(questions), 3),
        "misses": [qid for qid, r in ranks.items() if found(r, k) == 0.0],
        "ranks": ranks,  # rank of the first evidence chunk per question (None if not retrieved)
    }


def paired_difference(row: dict, base: dict, k: int) -> list[float]:
    """Hit@k gain of `row` over `base` with a paired 95% bootstrap interval: [mean, low, high]."""
    diffs = [found(row["ranks"][qid], k) - found(base["ranks"][qid], k) for qid in base["ranks"]]
    low, high = bootstrap_ci(diffs)
    return [round(sum(diffs) / len(diffs), 2), round(low, 2), round(high, 2)]


def main() -> None:
    """Compare retrieval pipelines and save the results, or act as a CI gate."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=list(RETRIEVER_MODES), choices=RETRIEVER_MODES)
    parser.add_argument("--set", dest="question_set", default="dev", choices=QUESTION_SETS)
    parser.add_argument("-k", type=int, default=config.TOP_K)
    parser.add_argument(
        "--min-hit", type=float, default=None, help="fail (exit 1) if hit@k of any mode is below this value"
    )
    args = parser.parse_args()

    questions = load_questions(QUESTION_SETS[args.question_set])
    rows = [evaluate(mode, args.k, questions) for mode in args.modes]
    k = args.k
    print(f"\n{args.question_set} set, {len(questions)} questions")
    print(f"| Retriever | Hit@1 | Hit@{k} (95% CI) | MRR | Precision@{k} | sec/query |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        low, high = r[f"hit@{k}_ci95"]
        hit = f"{r[f'hit@{k}']:.2f} ({low:.2f}-{high:.2f})"
        cells = [
            r["mode"],
            f"{r['hit@1']:.2f}",
            hit,
            *(f"{r[m]:.2f}" for m in ("mrr", f"precision@{k}", "sec_per_query")),
        ]
        print("| " + " | ".join(cells) + " |")
    for r in rows:
        if r["misses"]:
            print(f"{r['mode']} misses (not in top {k}): {', '.join(r['misses'])}")
    base = rows[0]
    if len(rows) > 1:
        print(f"\nHit@{k} gain over {base['mode']}, same questions (paired 95% bootstrap CI):")
    for r in rows[1:]:
        gain = paired_difference(r, base, k)
        r[f"hit@{k}_gain_vs_{base['mode']}"] = gain
        print(f"  {r['mode']:22} {gain[0]:+.2f}  [{gain[1]:+.2f}, {gain[2]:+.2f}]")

    if args.min_hit is None:
        name = "retrieval.json" if args.question_set == "dev" else f"retrieval_{args.question_set}.json"
        out = RESULTS_DIR / name
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nSaved {out.relative_to(config.ROOT)}")
    else:
        failing = [r["mode"] for r in rows if r[f"hit@{k}"] < args.min_hit]
        if failing:
            sys.exit(f"Retrieval regression: hit@{k} below {args.min_hit} for {failing}")
        print(f"Retrieval gate passed: hit@{k} >= {args.min_hit}")


if __name__ == "__main__":
    main()
