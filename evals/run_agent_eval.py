"""End-to-end agent check. Needs OPENAI_API_KEY.

Reports, for the agent (Stage 3):
    answer rate        answerable questions that got a cited answer (not a decline)
    citation hit rate  answered questions citing at least one expected source page
    calc accuracy      calculation questions whose answer contains the right percentage (+/-0.2 pts)
    decline rate       out-of-scope questions correctly declined

    python evals/run_agent_eval.py                    # dev set, gpt-5.6-luna
    python evals/run_agent_eval.py --final            # dev set, gpt-5.6-terra (final)
    python evals/run_agent_eval.py --final --set heldout
    python evals/run_agent_eval.py --limit 2          # cost check: tokens and time for two questions

Results are saved even if the run stops early (e.g. the API account runs out of
credit), so the questions already paid for are not lost.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from copilot import config  # noqa: E402
from copilot.agent import AgentResult, ResearchAgent, Usage  # noqa: E402
from copilot.evaluation import QUESTION_SETS, load_questions  # noqa: E402

OUT_OF_SCOPE_PATH = config.ROOT / "evals" / "out_of_scope.json"
RETRY_PAUSE_SECONDS = 20.0
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


def run_with_retry(agent: ResearchAgent, question: str) -> AgentResult:
    """Run one question, retrying once after a pause, so a brief network drop does not end a paid run."""
    try:
        return agent.run(question)
    except Exception as exc:
        print(f"  failed ({type(exc).__name__}); retrying once in {RETRY_PAUSE_SECONDS:.0f} s")
        time.sleep(RETRY_PAUSE_SECONDS)
        return agent.run(question)


def summarise(rows: list[dict[str, Any]], usages: list[Usage]) -> dict[str, Any]:
    """Agent metrics over the questions that were run."""
    answerable = [r for r in rows if r["id"] != "oos"]
    answered = [r for r in answerable if not r["fallback"]]
    calcs = [r for r in answerable if "calc_correct" in r]
    oos = [r for r in rows if r["id"] == "oos"]
    return {
        "questions": len(answerable),
        "answer_rate": round(len(answered) / max(len(answerable), 1), 3),
        "citation_hit_rate": round(sum(r["cited_expected_page"] for r in answered) / max(len(answered), 1), 3),
        "calc_accuracy": round(sum(r["calc_correct"] for r in calcs) / len(calcs), 3) if calcs else None,
        "out_of_scope_decline_rate": round(sum(r["fallback"] for r in oos) / len(oos), 3) if oos else None,
        "usage": usage_summary(usages),
    }


def main() -> None:
    """Run the agent on a question set (the full dev set also runs the out-of-scope questions)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help=f"use {config.EVAL_MODEL}")
    parser.add_argument("--set", dest="question_set", default="dev", choices=QUESTION_SETS)
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions, no out-of-scope set")
    args = parser.parse_args()
    load_dotenv()
    agent = ResearchAgent(model=config.EVAL_MODEL if args.final else config.DEV_MODEL)

    questions = load_questions(QUESTION_SETS[args.question_set])[: args.limit]
    # The out-of-scope questions run with the full dev set only, so other runs do not pay for them again.
    run_oos = args.question_set == "dev" and args.limit is None
    oos = json.loads(OUT_OF_SCOPE_PATH.read_text(encoding="utf-8")) if run_oos else []
    rows: list[dict[str, Any]] = []
    usages: list[Usage] = []
    stopped = None
    try:
        for q in questions:
            r = run_with_retry(agent, q.question)
            usages.append(r.usage)
            hit = bool({f"{q.source}, p. {p}" for p in q.pages} & set(r.sources))
            row: dict[str, Any] = {
                "id": q.id,
                "answer": r.answer,
                "sources": r.sources,
                "tool_calls": r.tool_calls,
                "fallback": r.fallback_triggered,
                "reason": r.fallback_reason,
                "cited_expected_page": hit,
            }
            if q.id in CALC_EXPECTED:
                row["calc_correct"] = calc_correct(r.answer, CALC_EXPECTED[q.id])
            rows.append({**row, "usage": r.usage.as_dict()})
            status = "DECLINED" if r.fallback_triggered else "answered"
            print(f"{q.id:8} {status:9} cited_ok={hit!s:5} tools={r.tool_calls} tokens={r.usage.output_tokens}")

        for item in oos:
            r = run_with_retry(agent, item["question"])
            usages.append(r.usage)
            rows.append(
                {
                    "id": "oos",
                    "question": item["question"],
                    "answer": r.answer,
                    "fallback": r.fallback_triggered,
                    "reason": r.fallback_reason,
                    "tool_calls": r.tool_calls,
                    "usage": r.usage.as_dict(),
                }
            )
            status = "DECLINED" if r.fallback_triggered else "ANSWERED"
            print(f"oos      {status:9} {item['question']}  ({r.fallback_reason})")
    except Exception as exc:  # e.g. the account ran out of credit: keep the answers already paid for
        stopped = f"{type(exc).__name__}: {exc}"
        print(f"\nStopped early after {len(rows)} questions: {stopped}")

    summary = {"model": agent.model, "question_set": args.question_set, **summarise(rows, usages)}
    if stopped:
        summary["stopped_early"] = stopped
    print("\n" + json.dumps(summary, indent=2))
    suffix = ("" if args.question_set == "dev" else f"_{args.question_set}") + (
        f"_first{args.limit}" if args.limit else ""
    )
    out = config.ROOT / "evals" / "results" / f"agent_{'final' if args.final else 'dev'}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"Saved {out.relative_to(config.ROOT)}")
    if stopped:
        sys.exit(f"Incomplete run: {stopped}")


if __name__ == "__main__":
    main()
