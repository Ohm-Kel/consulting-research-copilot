"""Check the relevance-floor guardrail on real retrieval. No API key needed.

Every answerable eval question must clear the floor (otherwise the agent would
wrongly decline it); out-of-scope questions should fall below it. On-topic
out-of-scope questions (another company's figures, a future year) can clear
the floor by design: the model-judgement guardrail handles them.

    python evals/run_guardrail_eval.py            # report
    python evals/run_guardrail_eval.py --strict   # exit 1 if any answerable question is blocked (CI)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config  # noqa: E402
from copilot.evaluation import load_questions  # noqa: E402
from copilot.retrieval import build_retriever  # noqa: E402

OUT_OF_SCOPE_PATH = config.ROOT / "evals" / "out_of_scope.json"


def main() -> None:
    """Report how the relevance floor treats answerable and out-of-scope questions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    retriever = build_retriever("hybrid_rerank")
    floor = config.RELEVANCE_THRESHOLD
    top = lambda q: retriever.search(q, k=1)[0].score  # noqa: E731

    answerable = [(q.id, top(q.question)) for q in load_questions()]
    blocked = [(qid, s) for qid, s in answerable if s < floor]
    print(f"Relevance floor: {floor}")
    print(
        f"Answerable questions passing the floor: {len(answerable) - len(blocked)}/{len(answerable)} "
        f"(lowest score {min(s for _, s in answerable):.2f})"
    )
    for qid, s in blocked:
        print(f"  BLOCKED {qid} ({s:.2f})")

    print("\nOut-of-scope questions:")
    oos = json.loads(OUT_OF_SCOPE_PATH.read_text(encoding="utf-8"))
    caught = 0
    for item in oos:
        s = top(item["question"])
        caught += s < floor
        verdict = "declined by floor" if s < floor else "passes floor -> model must decline"
        print(f"  {s:7.2f}  {verdict:34}  [{item['kind']}] {item['question']}")
    print(f"Caught by the floor alone: {caught}/{len(oos)}")

    if args.strict and blocked:
        sys.exit(f"Guardrail regression: {len(blocked)} answerable question(s) fall below the relevance floor")


if __name__ == "__main__":
    main()
