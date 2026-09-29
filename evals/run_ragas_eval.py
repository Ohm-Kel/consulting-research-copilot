"""Score end-to-end answers with RAGAS (LLM-judged). Needs OPENAI_API_KEY.

    python evals/run_ragas_eval.py --modes vector hybrid_rerank          # development run, gpt-5.6-luna
    python evals/run_ragas_eval.py --modes vector hybrid_rerank --final  # final run, gpt-5.6-terra
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
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from copilot import config  # noqa: E402
from copilot.evaluation import EvalQuestion, load_questions  # noqa: E402
from copilot.generate import answer_question, make_client  # noqa: E402
from copilot.retrieval import RETRIEVER_MODES, build_retriever  # noqa: E402

RESULTS_DIR = config.ROOT / "evals" / "results"
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def use_reasoning_model_params(llm) -> None:
    """Workaround for ragas 0.4.3: its GPT-5 detection parses the version as an
    integer, so 'gpt-5.6-luna' is not recognised and gets `max_tokens`, which the
    API rejects. Apply ragas's own GPT-5 rules: `max_completion_tokens`,
    temperature 1.0, no top_p."""
    original = llm._map_openai_params

    def mapped() -> dict:
        params = original()
        if "max_tokens" in params:
            params["max_completion_tokens"] = params.pop("max_tokens")
        params["temperature"] = 1.0
        params.pop("top_p", None)
        return params

    llm._map_openai_params = mapped


def build_metrics(judge_model: str) -> dict:
    """RAGAS metrics judged by an OpenAI model, with local embeddings for answer relevancy."""
    from ragas.embeddings.base import BaseRagasEmbedding, embedding_factory
    from ragas.llms.base import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithReference, ContextRecall, Faithfulness

    llm = llm_factory(
        judge_model,
        provider="openai",
        client=AsyncOpenAI(timeout=config.LLM_TIMEOUT_SECONDS, max_retries=config.LLM_MAX_RETRIES),
        max_tokens=4096,
    )
    use_reasoning_model_params(llm)
    embeddings = embedding_factory("huggingface", model=config.EMBEDDING_MODEL)
    if not isinstance(embeddings, BaseRagasEmbedding):  # AnswerRelevancy needs the modern interface
        raise TypeError(f"unexpected ragas embeddings type: {type(embeddings).__name__}")
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


async def score_one(metrics: dict, q: EvalQuestion, answer: str, contexts: list[str]) -> dict[str, float]:
    """Score one answer on the four RAGAS metrics concurrently."""
    results = await asyncio.gather(
        metrics["faithfulness"].ascore(user_input=q.question, response=answer, retrieved_contexts=contexts),
        metrics["answer_relevancy"].ascore(user_input=q.question, response=answer),
        metrics["context_precision"].ascore(
            user_input=q.question, reference=q.expected_answer, retrieved_contexts=contexts
        ),
        metrics["context_recall"].ascore(
            user_input=q.question, retrieved_contexts=contexts, reference=q.expected_answer
        ),
    )
    return {name: float(r.value) for name, r in zip(METRIC_NAMES, results, strict=True)}


async def run_mode(mode: str, questions: list[EvalQuestion], answer_model: str, metrics: dict) -> dict:
    """Answer and score every question with one retrieval pipeline."""
    retriever = build_retriever(mode)
    client = make_client()
    rows: list[dict[str, Any]] = []
    all_scores: list[dict[str, float]] = []
    for q in questions:
        hits = retriever.search(q.question)
        answer = answer_question(q.question, hits, client=client, model=answer_model)
        scores = await score_one(metrics, q, answer.text, answer.contexts)
        rows.append({"id": q.id, "answer": answer.text, "sources": answer.sources, **scores})
        all_scores.append(scores)
        print(f"  {q.id}: " + "  ".join(f"{m}={scores[m]:.2f}" for m in METRIC_NAMES))
    means = {m: round(sum(s[m] for s in all_scores) / len(all_scores), 3) for m in METRIC_NAMES}
    return {"mode": mode, "answer_model": answer_model, **means, "rows": rows}


async def run_all(modes: list[str], questions: list[EvalQuestion], model: str) -> list[dict]:
    """One event loop for the whole run: the async OpenAI client used by the
    judge metrics is bound to the loop it was first used on."""
    metrics = build_metrics(model)
    summaries = []
    for mode in modes:
        print(f"\n== {mode} ({len(questions)} questions, model {model})")
        summaries.append(await run_mode(mode, questions, model, metrics))
    return summaries


def main() -> None:
    """Run the RAGAS evaluation and save the results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["vector", "hybrid_rerank"], choices=RETRIEVER_MODES)
    parser.add_argument("--final", action="store_true", help=f"use {config.EVAL_MODEL} to answer and judge")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    args = parser.parse_args()

    load_dotenv()
    model = config.EVAL_MODEL if args.final else config.DEV_MODEL
    questions = load_questions()[: args.limit]
    summaries = asyncio.run(run_all(args.modes, questions, model))

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
