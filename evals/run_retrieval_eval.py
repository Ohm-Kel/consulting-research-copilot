"""Compare retrieval pipelines on the 25-question set. No API key needed.

python evals/run_retrieval_eval.py                         # all modes, writes evals/results/retrieval.json
python evals/run_retrieval_eval.py --modes hybrid_rerank --min-hit 0.8   # CI regression gate
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.evaluation import first_relevant_rank, load_questions, retrieval_metrics  # noqa: E402
from copilot.retrieval import RETRIEVER_MODES, build_retriever  # noqa: E402

RESULTS_PATH = config.ROOT / "evals" / "results" / "retrieval.json"


def evaluate(mode: str, k: int) -> dict:
    retriever = build_retriever(mode)
    questions = load_questions()
    start = time.perf_counter()
    results = [(q, retriever.search(q.question, k=max(k, 10))) for q in questions]
    seconds = time.perf_counter() - start
    metrics = retrieval_metrics(results, k=k)
    misses = [q.id for q, hits in results if (r := first_relevant_rank(hits, q)) is None or r > k]
    return {
        "mode": mode,
        **{m: round(v, 3) for m, v in metrics.items()},
        "sec_per_query": round(seconds / len(questions), 3),
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=list(RETRIEVER_MODES), choices=RETRIEVER_MODES)
    parser.add_argument("-k", type=int, default=config.TOP_K)
    parser.add_argument(
        "--min-hit", type=float, default=None, help="fail (exit 1) if hit@k of any mode is below this value"
    )
    args = parser.parse_args()

    rows = [evaluate(mode, args.k) for mode in args.modes]
    k = args.k
    print(f"\n| Retriever | Hit@1 | Hit@{k} | MRR | Precision@{k} | sec/query |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        metrics = ("hit@1", f"hit@{k}", "mrr", f"precision@{k}", "sec_per_query")
        print("| " + " | ".join([r["mode"], *(f"{r[m]:.2f}" for m in metrics)]) + " |")
    for r in rows:
        if r["misses"]:
            print(f"{r['mode']} misses (not in top {k}): {', '.join(r['misses'])}")

    if args.min_hit is None:
        RESULTS_PATH.parent.mkdir(exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nSaved {RESULTS_PATH.relative_to(config.ROOT)}")
    else:
        failing = [r["mode"] for r in rows if r[f"hit@{k}"] < args.min_hit]
        if failing:
            sys.exit(f"Retrieval regression: hit@{k} below {args.min_hit} for {failing}")
        print(f"Retrieval gate passed: hit@{k} >= {args.min_hit}")


if __name__ == "__main__":
    main()
