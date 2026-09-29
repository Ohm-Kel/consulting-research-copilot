"""End-to-end agent check. Needs OPENAI_API_KEY.

Reports, for the agent (Stage 3):
    answer rate        answerable questions that got a cited answer (not a decline)
    citation hit rate  answered questions citing at least one expected source page
    calc accuracy      calculation questions whose answer contains the right percentage (+/-0.2 pts)
    decline rate       out-of-scope questions correctly declined

    python evals/run_agent_eval.py                    # dev set, gpt-5.6-luna
    python evals/run_agent_eval.py --final            # dev set, gpt-5.6-terra (final)
    python evals/run_agent_eval.py --final --set heldout
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from copilot import config  # noqa: E402
from copilot.agent import ResearchAgent, Usage  # noqa: E402
from copilot.evaluation import QUESTION_SETS, load_questions  # noqa: E402

OUT_OF_SCOPE_PATH = config.ROOT / "evals" / "out_of_scope.json"
# Expected percentage for each calculation question (verified with copilot.tools.calculate).
CALC_EXPECTED = {
    "nike-05": -43.5,
    "lulu-05": 17.1,
    "colm-02": -3.4,
    "deck-05": 44.8,
    "nike-h4": 66.7,
    "lulu-h2": 7.9,
    "ua-h2": 8.4,
    "colm-h2": 72.7,
    "deck-h2": 26.3,
}


DECREASE_WORDS = re.compile(r"\b(decreas|declin|fell|fall|drop|down|lower|shrank|contract)", re.IGNORECASE)
INCREASE_WORDS = re.compile(r"\b(increas|grew|grow|rose|rise|up\b|higher|gain)", re.IGNORECASE)


PERCENT = re.compile(r"(-?\d+(?:\.\d+)?)\s?%")


def percentages(text: str) -> list[float]:
    """Extract every percentage figure from `text`."""
    return [float(x) for x in PERCENT.findall(text.replace(",", ""))]


def calc_correct(answer: str, expected: float, tolerance: float = 0.2) -> bool:
    """True if `answer` states `expected` (a percentage) with the right direction.

    "-43.5%" and "fell 43.5%" both match -43.5, but "rose 43.5%" does not; likewise "fell
    17.1%" does not match +17.1. The direction is read from the words just before the number."""
    text = answer.replace(",", "")
    for match in PERCENT.finditer(text):
        value = float(match.group(1))
        before = text[max(0, match.start() - 60) : match.start()]
        says_decrease = bool(DECREASE_WORDS.search(before)) and not INCREASE_WORDS.search(before)
        if abs(value - expected) <= tolerance and not (expected > 0 and says_decrease):
            return True
        if expected < 0 and abs(-value - expected) <= tolerance and says_decrease:
            return True
    return False


def usage_summary(usages: list[Usage]) -> dict[str, float | int | None]:
    """Totals and per-question averages of LLM calls, tokens, time and cost."""
    n = max(len(usages), 1)
    costs = [u.cost_usd for u in usages]
    total_cost = None if any(c is None for c in costs) else round(sum(c for c in costs if c is not None), 4)
    return {
        "questions": len(usages),
        "llm_calls": sum(u.llm_calls for u in usages),
        "input_tokens": sum(u.input_tokens for u in usages),
        "output_tokens": sum(u.output_tokens for u in usages),
        "mean_seconds_per_question": round(sum(u.seconds for u in usages) / n, 2),
        "total_cost_usd": total_cost,
    }


def main() -> None:
    """Run the agent on the evaluation and out-of-scope sets and save a summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help=f"use {config.EVAL_MODEL}")
    parser.add_argument("--set", dest="question_set", default="dev", choices=QUESTION_SETS)
    args = parser.parse_args()
    load_dotenv()
    agent = ResearchAgent(model=config.EVAL_MODEL if args.final else config.DEV_MODEL)

    answered = cited_ok = calc_ok = 0
    rows = []
    usages: list[Usage] = []
    questions = load_questions(QUESTION_SETS[args.question_set])
    calc_ids = [q.id for q in questions if q.id in CALC_EXPECTED]
    for q in questions:
        r = agent.run(q.question)
        usages.append(r.usage)
        expected = {f"{q.source}, p. {p}" for p in q.pages}
        hit = bool(expected & set(r.sources))
        answered += not r.fallback_triggered
        cited_ok += hit
        if q.id in CALC_EXPECTED:
            calc_ok += calc_correct(r.answer, CALC_EXPECTED[q.id])
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

    # The out-of-scope questions run with the dev set only, so a held-out run does not pay for them twice.
    oos = json.loads(OUT_OF_SCOPE_PATH.read_text(encoding="utf-8")) if args.question_set == "dev" else []
    declined = 0
    for item in oos:
        r = agent.run(item["question"])
        usages.append(r.usage)
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
        "question_set": args.question_set,
        "answer_rate": round(answered / n, 3),
        "citation_hit_rate": round(cited_ok / max(answered, 1), 3),
        "calc_accuracy": round(calc_ok / max(len(calc_ids), 1), 3),
        "out_of_scope_decline_rate": round(declined / len(oos), 3) if oos else None,
        "usage": usage_summary(usages),
    }
    print("\n" + json.dumps(summary, indent=2))
    suffix = "" if args.question_set == "dev" else f"_{args.question_set}"
    out = config.ROOT / "evals" / "results" / f"agent_{'final' if args.final else 'dev'}{suffix}.json"
    out.write_text(json.dumps({**summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"Saved {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
