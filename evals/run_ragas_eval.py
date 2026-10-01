"""Score end-to-end answers with RAGAS (LLM-judged). Needs OPENAI_API_KEY.

    python evals/run_ragas_eval.py --modes vector hybrid_rerank          # development run, gpt-5.6-luna
    python evals/run_ragas_eval.py --modes vector hybrid_rerank --final  # final run: gpt-5.6-terra answers
    python evals/run_ragas_eval.py --limit 3                             # quick smoke test
    python evals/run_ragas_eval.py --set heldout --final                 # held-out questions

The judge defaults to the development model, so in a final run the answers are
scored by a different model from the one that wrote them (a model tends to rate
its own output highly). Each mean comes with a 95% bootstrap interval, and the
modes are compared question by question with a paired interval.

Metrics (0-1, higher is better):
    faithfulness       every claim in the answer is supported by the retrieved context
    answer_relevancy   the answer addresses the question
    context_precision  relevant passages are ranked above irrelevant ones
    context_recall     the retrieved context contains what the reference answer needs
"""

import argparse
import asyncio
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from openai import AsyncOpenAI, OpenAI  # noqa: E402

from copilot import config  # noqa: E402
from copilot.evaluation import QUESTION_SETS, EvalQuestion, bootstrap_ci, load_questions  # noqa: E402
from copilot.generate import answer_question  # noqa: E402
from copilot.retrieval import RETRIEVER_MODES, build_retriever  # noqa: E402

RESULTS_DIR = config.ROOT / "evals" / "results"
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
RETRY_PAUSE_SECONDS = 20.0


class TokenCounter:
    """Adds up the tokens of every chat completion that passes through an OpenAI client,
    so a run reports what it cost. RAGAS makes its judge calls internally, so the count
    is taken from the HTTP responses."""

    def __init__(self) -> None:
        self.calls = self.input_tokens = self.output_tokens = 0

    def add(self, body: bytes) -> None:
        """Count one response body; anything without a `usage` block (errors) is ignored."""
        try:
            usage = json.loads(body)["usage"]
            tokens_in, tokens_out = int(usage["prompt_tokens"]), int(usage["completion_tokens"])
        except (ValueError, KeyError, TypeError):
            return
        self.calls += 1
        self.input_tokens += tokens_in
        self.output_tokens += tokens_out

    def on_response(self, response: httpx.Response) -> None:
        """httpx response hook for the synchronous client."""
        self.add(response.read())

    async def on_async_response(self, response: httpx.Response) -> None:
        """httpx response hook for the asynchronous client."""
        self.add(await response.aread())

    def as_dict(self) -> dict[str, int]:
        """Totals so far."""
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}

    def since(self, earlier: dict[str, int]) -> dict[str, int]:
        """Totals added since an earlier `as_dict()` snapshot."""
        return {key: value - earlier[key] for key, value in self.as_dict().items()}


ANSWER_USAGE = TokenCounter()
JUDGE_USAGE = TokenCounter()


def make_client() -> OpenAI:
    """Client for the answering model, with its tokens counted."""
    return OpenAI(
        timeout=config.LLM_TIMEOUT_SECONDS,
        max_retries=config.LLM_MAX_RETRIES,
        http_client=httpx.Client(event_hooks={"response": [ANSWER_USAGE.on_response]}),
    )


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
        client=AsyncOpenAI(
            timeout=config.LLM_TIMEOUT_SECONDS,
            max_retries=config.LLM_MAX_RETRIES,
            http_client=httpx.AsyncClient(event_hooks={"response": [JUDGE_USAGE.on_async_response]}),
        ),
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


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean and 95% bootstrap interval of each metric over the questions. A question a
    metric could not score (NaN) is left out of that metric's mean."""
    summary: dict[str, Any] = {}
    for m in METRIC_NAMES:
        values = [r[m] for r in rows if not math.isnan(r[m])]
        low, high = bootstrap_ci(values)
        summary[m] = round(sum(values) / len(values), 3) if values else None
        summary[f"{m}_ci95"] = [round(low, 2), round(high, 2)]
    return summary


def paired_gain(rows: list[dict[str, Any]], base_rows: list[dict[str, Any]], metric: str) -> list[float]:
    """Mean per-question difference in `metric` over `base_rows` (same questions),
    with a paired 95% bootstrap interval: [mean, low, high]."""
    base = {r["id"]: r[metric] for r in base_rows}
    diffs = [r[metric] - base[r["id"]] for r in rows if not (math.isnan(r[metric]) or math.isnan(base[r["id"]]))]
    low, high = bootstrap_ci(diffs)
    return [round(sum(diffs) / len(diffs), 3) if diffs else 0.0, round(low, 3), round(high, 3)]


async def run_mode(
    mode: str,
    questions: list[EvalQuestion],
    answer_model: str,
    judge_model: str,
    metrics: dict,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Answer and score every question with one retrieval pipeline. If an API call
    fails (e.g. the account runs out of credit), the questions scored so far are kept;
    `checkpoint` receives them after every question, in case the process itself dies."""
    retriever = build_retriever(mode)
    client = make_client()
    rows: list[dict[str, Any]] = []
    stopped = None
    answers_before, judge_before = ANSWER_USAGE.as_dict(), JUDGE_USAGE.as_dict()
    for q in questions:
        for attempt in (1, 2):  # one retry, so a brief network drop does not end a paid run
            try:
                hits = retriever.search(q.question)
                answer = answer_question(q.question, hits, client=client, model=answer_model)
                scores = await score_one(metrics, q, answer.text, answer.contexts)
                stopped = None
                break
            except Exception as exc:
                stopped = f"{type(exc).__name__}: {exc}"
                if attempt == 1:
                    print(f"  {q.id} failed ({type(exc).__name__}); retrying once in {RETRY_PAUSE_SECONDS:.0f} s")
                    await asyncio.sleep(RETRY_PAUSE_SECONDS)
        if stopped:
            print(f"  stopped early at {q.id}: {stopped}")
            break
        rows.append({"id": q.id, "answer": answer.text, "sources": answer.sources, **scores})
        print(f"  {q.id}: " + "  ".join(f"{m}={scores[m]:.2f}" for m in METRIC_NAMES))
        if checkpoint:
            checkpoint({"mode": mode, "answer_model": answer_model, "judge_model": judge_model, "rows": rows})
    summary = {"mode": mode, "answer_model": answer_model, "judge_model": judge_model, **summarise(rows)}
    summary["usage"] = {"answers": ANSWER_USAGE.since(answers_before), "judge": JUDGE_USAGE.since(judge_before)}
    print(f"  tokens: {summary['usage']}")
    if stopped:
        summary["stopped_early"] = stopped
    return {**summary, "rows": rows}


async def run_all(
    modes: list[str],
    questions: list[EvalQuestion],
    answer_model: str,
    judge_model: str,
    checkpoint_path: Path | None = None,
) -> list[dict]:
    """One event loop for the whole run: the async OpenAI client used by the
    judge metrics is bound to the loop it was first used on. With `checkpoint_path`,
    everything scored so far is written there after each question."""
    metrics = build_metrics(judge_model)
    summaries: list[dict] = []

    def checkpoint(partial: dict[str, Any]) -> None:
        if checkpoint_path:
            checkpoint_path.write_text(json.dumps([*summaries, partial], indent=2), encoding="utf-8")

    for mode in modes:
        print(f"\n== {mode} ({len(questions)} questions, answers by {answer_model}, judged by {judge_model})")
        summaries.append(await run_mode(mode, questions, answer_model, judge_model, metrics, checkpoint))
        if "stopped_early" in summaries[-1]:
            break
    return summaries


def main() -> None:
    """Run the RAGAS evaluation and save the results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["vector", "hybrid_rerank"], choices=RETRIEVER_MODES)
    parser.add_argument("--final", action="store_true", help=f"answer with {config.EVAL_MODEL}")
    parser.add_argument("--judge", default=config.DEV_MODEL, help="model that scores the answers (%(default)s)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--set", dest="question_set", default="dev", choices=QUESTION_SETS)
    args = parser.parse_args()

    load_dotenv()
    answer_model = config.EVAL_MODEL if args.final else config.DEV_MODEL
    if args.judge == answer_model:
        print(f"Note: {answer_model} judges its own answers, so scores may be optimistic.")
    questions = load_questions(QUESTION_SETS[args.question_set])[: args.limit]
    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = ("" if args.question_set == "dev" else f"_{args.question_set}") + (
        f"_first{args.limit}" if args.limit else ""
    )
    out = RESULTS_DIR / f"ragas_{'final' if args.final else 'dev'}{suffix}.json"
    partial = out.with_suffix(".partial.json")  # survives a crash; removed once the full results are saved
    summaries = asyncio.run(run_all(args.modes, questions, answer_model, args.judge, partial))

    def cell(s: dict[str, Any], m: str) -> str:
        low, high = s[f"{m}_ci95"]
        return "n/a" if s[m] is None else f"{s[m]:.2f} ({low:.2f}-{high:.2f})"

    print("\n| Metric (95% CI) | " + " | ".join(s["mode"] for s in summaries) + " |")
    print("|---|" + "---|" * len(summaries))
    for m in METRIC_NAMES:
        print(f"| {m} | " + " | ".join(cell(s, m) for s in summaries) + " |")
    base = summaries[0]
    for s in summaries[1:]:
        print(f"\n{s['mode']} minus {base['mode']}, same questions (paired 95% bootstrap CI):")
        for m in METRIC_NAMES:
            gain = paired_gain(s["rows"], base["rows"], m)
            s[f"{m}_gain_vs_{base['mode']}"] = gain
            print(f"  {m:18} {gain[0]:+.3f}  [{gain[1]:+.3f}, {gain[2]:+.3f}]")

    out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(f"\nSaved {out.relative_to(config.ROOT)}")
    stopped = [s["stopped_early"] for s in summaries if "stopped_early" in s]
    if stopped:
        sys.exit(f"Incomplete run: {stopped[0]}")


if __name__ == "__main__":
    main()
