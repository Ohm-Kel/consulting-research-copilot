"""Score end-to-end answers with RAGAS (LLM-judged). Needs ANTHROPIC_API_KEY.

    python evals/run_ragas_eval.py --modes vector hybrid_rerank          # development run, Haiku 4.5
    python evals/run_ragas_eval.py --modes vector hybrid_rerank --final  # final run, Sonnet 4.6
    python evals/run_ragas_eval.py --limit 3                             # quick smoke test

Metrics (0-1, higher is better):
    faithfulness       every claim in the answer is supported by the retrieved context
    answer_relevancy   the answer addresses the question
    context_precision  relevant passages are ranked above irrelevant ones
    context_recall     the retrieved context contains what the reference answer needs
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anthropic import AsyncAnthropic  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from copilot import config  # noqa: E402
from copilot.evaluation import EvalQuestion, load_questions  # noqa: E402
from copilot.generate import answer_question, make_client  # noqa: E402
from copilot.retrieval import RETRIEVER_MODES, build_retriever  # noqa: E402

RESULTS_DIR = config.ROOT / "evals" / "results"
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def build_metrics(judge_model: str) -> dict:
    """RAGAS metrics judged by Claude, with local embeddings for answer relevancy."""
    from ragas.embeddings.base import embedding_factory
    from ragas.llms.base import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, Faithfulness

    llm = llm_factory(judge_model, provider="anthropic", client=AsyncAnthropic(), max_tokens=4096)
    embeddings = embedding_factory("huggingface", model=config.EMBEDDING_MODEL)
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


async def score_one(metrics: dict, q: EvalQuestion, answer: str, contexts: list[str]) -> dict[str, float]:
    results = await asyncio.gather(
        metrics["faithfulness"].ascore(user_input=q.question, response=answer, retrieved_contexts=contexts),
        metrics["answer_relevancy"].ascore(user_input=q.question, response=answer),
        metrics["context_precision"].ascore(user_input=q.question, reference=q.expected_answer, retrieved_contexts=contexts),
        metrics["context_recall"].ascore(user_input=q.question, retrieved_contexts=contexts, reference=q.expected_answer),
    )
    return {name: float(r.value) for name, r in zip(METRIC_NAMES, results)}


def run_mode(mode: str, questions: list[EvalQuestion], answer_model: str, metrics: dict) -> dict:
    retriever = build_retriever(mode)
    client = make_client()
    rows = []
    for q in questions:
        hits = retriever.search(q.question)
        answer = answer_question(q.question, hits, client=client, model=answer_model)
        scores = asyncio.run(score_one(metrics, q, answer.text, answer.contexts))
        rows.append({"id": q.id, "answer": answer.text, "sources": answer.sources, **scores})
        print(f"  {q.id}: " + "  ".join(f"{m}={scores[m]:.2f}" for m in METRIC_NAMES))
    means = {m: round(sum(r[m] for r in rows) / len(rows), 3) for m in METRIC_NAMES}
    return {"mode": mode, "answer_model": answer_model, **means, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["vector", "hybrid_rerank"], choices=RETRIEVER_MODES)
    parser.add_argument("--final", action="store_true", help=f"use {config.EVAL_MODEL} to answer and judge")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    args = parser.parse_args()

    load_dotenv()
    model = config.EVAL_MODEL if args.final else config.DEV_MODEL
    questions = load_questions()[: args.limit]
    metrics = build_metrics(model)

    summaries = []
    for mode in args.modes:
        print(f"\n== {mode} ({len(questions)} questions, model {model})")
        summaries.append(run_mode(mode, questions, model, metrics))

    print("\n| Metric | " + " | ".join(s["mode"] for s in summaries) + " |")
    print("|---|" + "---|" * len(summaries))
    for m in METRIC_NAMES:
        print(f"| {m} | " + " | ".join(f"{s[m]:.2f}" for s in summaries) + " |")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"ragas_{'final' if args.final else 'dev'}.json"
    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nSaved {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
