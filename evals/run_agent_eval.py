"""End-to-end agent check. Needs OPENAI_API_KEY.

Reports, for the agent (Stage 3):
    answer rate        answerable questions that got a cited answer (not a decline)
    citation hit rate  answered questions citing at least one expected source page
    calc accuracy      calculation questions whose answer contains the right percentage (+/-0.2 pts)
    decline rate       out-of-scope questions correctly declined

    python evals/run_agent_eval.py            # gpt-5.6-luna
    python evals/run_agent_eval.py --final    # gpt-5.6-terra (final)
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from copilot import config  # noqa: E402
from copilot.agent import ResearchAgent  # noqa: E402
from copilot.evaluation import load_questions  # noqa: E402

OUT_OF_SCOPE_PATH = config.ROOT / "evals" / "out_of_scope.json"
# Expected percentage for each calculation question (verified with copilot.tools.calculate).
CALC_EXPECTED = {"nike-05": -43.5, "lulu-05": 17.1, "colm-02": -3.4, "deck-05": 44.8}


def percentages(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"(-?\d+(?:\.\d+)?)\s?%", text.replace(",", ""))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help=f"use {config.EVAL_MODEL}")
    args = parser.parse_args()
    load_dotenv()
    agent = ResearchAgent(model=config.EVAL_MODEL if args.final else config.DEV_MODEL)

    answered = cited_ok = calc_ok = 0
    rows = []
    questions = load_questions()
    for q in questions:
        r = agent.run(q.question)
        expected = {f"{q.source}, p. {p}" for p in q.pages}
        hit = bool(expected & set(r.sources))
        answered += not r.fallback_triggered
        cited_ok += hit
        if q.id in CALC_EXPECTED:
            # Accept the magnitude too: "decreased 43.5%" is correct for -43.5.
            calc_ok += any(abs(abs(p) - abs(CALC_EXPECTED[q.id])) <= 0.2 for p in percentages(r.answer))
        rows.append(
            {
                "id": q.id,
                "answer": r.answer,
                "sources": r.sources,
                "tool_calls": r.tool_calls,
                "fallback": r.fallback_triggered,
                "reason": r.fallback_reason,
                "cited_expected_page": hit,
            }
        )
        print(
            f"{q.id:8} {'DECLINED' if r.fallback_triggered else 'answered':9} cited_ok={hit!s:5} tools={r.tool_calls}"
        )

    oos = json.loads(OUT_OF_SCOPE_PATH.read_text(encoding="utf-8"))
    declined = 0
    for item in oos:
        r = agent.run(item["question"])
        declined += r.fallback_triggered
        rows.append(
            {
                "id": "oos",
                "question": item["question"],
                "answer": r.answer,
                "fallback": r.fallback_triggered,
                "reason": r.fallback_reason,
                "tool_calls": r.tool_calls,
            }
        )
        print(
            f"oos      {'DECLINED' if r.fallback_triggered else 'ANSWERED':9} {item['question']}  ({r.fallback_reason})"
        )

    n = len(questions)
    summary = {
        "model": agent.model,
        "answer_rate": round(answered / n, 3),
        "citation_hit_rate": round(cited_ok / max(answered, 1), 3),
        "calc_accuracy": round(calc_ok / len(CALC_EXPECTED), 3),
        "out_of_scope_decline_rate": round(declined / len(oos), 3),
    }
    print("\n" + json.dumps(summary, indent=2))
    out = config.ROOT / "evals" / "results" / f"agent_{'final' if args.final else 'dev'}.json"
    out.write_text(json.dumps({**summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"Saved {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
