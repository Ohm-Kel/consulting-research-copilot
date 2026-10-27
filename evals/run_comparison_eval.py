"""Cross-company comparison set: questions that need evidence from two reports.

Retrieval (free): one search, run on the question as asked. Does its top 5 or top 10 hold
evidence from every report the question needs?
Agent (OpenAI API, about $0.002 a question): does the answer state the expected figures for
both companies and cite a supporting page from each report? The agent can search more than
once, for example once per company, so it can succeed where a single search falls short.

    python evals/run_comparison_eval.py           # retrieval only, writes evals/results/comparison.json
    python evals/run_comparison_eval.py --agent   # also the agent (gpt-5.6-luna), comparison_agent_dev.json
    python evals/run_comparison_eval.py --rescore # re-score the saved agent answers after a key correction (free)
"""

import argparse
import json
import time
from typing import Any

from dotenv import load_dotenv
from run_agent_eval import run_with_retry, usage_summary  # evals/ is on the path when run as a script

from copilot import config
from copilot.agent import ResearchAgent, Usage
from copilot.evaluation import (
    ComparisonQuestion,
    bootstrap_ci,
    covers_all_parts,
    first_relevant_rank,
    load_comparisons,
    states_figures,
)
from copilot.retrieval import RETRIEVER_MODES, build_retriever

RESULTS_DIR = config.ROOT / "evals" / "results"


def evaluate_retrieval(mode: str, questions: list[ComparisonQuestion]) -> dict[str, Any]:
    """Share of questions whose top 5 / top 10 hold evidence from every report, for one pipeline."""
    retriever = build_retriever(mode)
    start = time.perf_counter()
    hits = {q.id: retriever.search(q.question, k=10) for q in questions}
    seconds = time.perf_counter() - start
    row: dict[str, Any] = {"mode": mode}
    for k in (5, 10):
        covered = [float(covers_all_parts(hits[q.id], q, k)) for q in questions]
        low, high = bootstrap_ci(covered)
        row[f"all_reports@{k}"] = round(sum(covered) / len(covered), 3)
        row[f"all_reports@{k}_ci95"] = [round(low, 2), round(high, 2)]
    parts = [(q, part) for q in questions for part in q.parts]
    missing = [part.id for q, part in parts if first_relevant_rank(hits[q.id][:10], part) is None]
    row["reports_found@10"] = round(1 - len(missing) / len(parts), 3)
    row["missing@10"] = missing  # reports whose evidence was not in the top 10
    row["sec_per_query"] = round(seconds / len(questions), 3)
    return row


def score_answer(question: ComparisonQuestion, answer: str, sources: list[str]) -> dict[str, Any]:
    """Whether an answer cites a supporting page from each report and states every expected figure."""
    cited = {
        part.company: bool({f"{part.source}, p. {page}" for page in part.pages} & set(sources))
        for part in question.parts
    }
    return {
        "cited_each_report": all(cited.values()),
        "cited": cited,
        "states_figures": states_figures(answer, question.figures),
    }


def summarise(model: str, rows: list[dict[str, Any]], usage: dict[str, Any]) -> dict[str, Any]:
    """Counts over the scored rows, with the run's usage."""
    for row in rows:
        status = "DECLINED" if row["fallback"] else "answered"
        print(
            f"{row['id']}  {status}  cited_each_report={row['cited_each_report']!s:5}"
            f"  figures={row['states_figures']!s:5}  sources={row['sources']}"
        )
    return {
        "model": model,
        "questions": len(rows),
        "answered": sum(not r["fallback"] for r in rows),
        "cited_each_report": sum(r["cited_each_report"] for r in rows),
        "states_figures": sum(r["states_figures"] for r in rows),
        "usage": usage,
        "rows": rows,
    }


def evaluate_agent(model: str, questions: list[ComparisonQuestion]) -> dict[str, Any]:
    """Run the agent on every comparison question and score its answers and citations."""
    agent = ResearchAgent(model=model)
    rows: list[dict[str, Any]] = []
    usages: list[Usage] = []
    for q in questions:
        result = run_with_retry(agent, q.question)
        usages.append(result.usage)
        rows.append(
            {
                "id": q.id,
                "answer": result.answer,
                "sources": result.sources,
                "tool_calls": result.tool_calls,
                "fallback": result.fallback_triggered,
                **score_answer(q, result.answer, result.sources),
                "usage": result.usage.as_dict(),
            }
        )
    return summarise(model, rows, usage_summary(usages))


def rescore(saved: dict[str, Any], questions: list[ComparisonQuestion]) -> dict[str, Any]:
    """Score saved answers again against the current answer key, without calling the API."""
    by_id = {q.id: q for q in questions}
    rows = [{**row, **score_answer(by_id[row["id"]], row["answer"], row["sources"])} for row in saved["rows"]]
    return summarise(saved["model"], rows, saved["usage"])


def main() -> None:
    """Score retrieval on the comparison set, and optionally the agent."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["vector", "hybrid", "hybrid_rerank"], choices=RETRIEVER_MODES)
    parser.add_argument("--agent", action="store_true", help="also run the agent (uses the OpenAI API)")
    parser.add_argument("--final", action="store_true", help=f"run the agent with {config.EVAL_MODEL}")
    parser.add_argument("--rescore", action="store_true", help="re-score the saved agent answers (no API calls)")
    args = parser.parse_args()

    questions = load_comparisons()
    agent_out = RESULTS_DIR / f"comparison_agent_{'final' if args.final else 'dev'}.json"
    if args.rescore:
        result = rescore(json.loads(agent_out.read_text(encoding="utf-8")), questions)
        agent_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
        print(f"Saved {agent_out.relative_to(config.ROOT)}")
        return
    rows = [evaluate_retrieval(mode, questions) for mode in args.modes]
    print(f"\ncomparison set, {len(questions)} questions, {sum(len(q.parts) for q in questions)} reports needed")
    print("| Retriever | All reports in top 5 (95% CI) | All reports in top 10 | Reports found in top 10 |")
    print("|---|---|---|---|")
    for r in rows:
        low, high = r["all_reports@5_ci95"]
        print(
            f"| {r['mode']} | {r['all_reports@5']:.2f} ({low:.2f}-{high:.2f}) | {r['all_reports@10']:.2f} "
            f"| {r['reports_found@10']:.2f} |"
        )
    out = RESULTS_DIR / "comparison.json"
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {out.relative_to(config.ROOT)}")

    if args.agent:
        load_dotenv()
        result = evaluate_agent(config.EVAL_MODEL if args.final else config.DEV_MODEL, questions)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
        agent_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {agent_out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
