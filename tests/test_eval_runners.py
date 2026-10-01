"""The eval runner scripts, with the LLM and RAGAS replaced by stubs."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from copilot import config
from copilot.generate import Answer
from copilot.retrieval import Hit
from tests.conftest import SAMPLE_CHUNKS


def load_script(name: str):
    path = config.ROOT / "evals" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class StubMetric:
    """Async metric like RAGAS's; records the event loop it ran on."""

    loops: set[int] = set()

    def __init__(self, value: float) -> None:
        self.value = value

    async def ascore(self, **_: object) -> SimpleNamespace:
        StubMetric.loops.add(id(asyncio.get_running_loop()))
        return SimpleNamespace(value=self.value)


def test_ragas_runner_scores_all_modes_in_one_event_loop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ragas = load_script("run_ragas_eval")
    stub_metrics = {name: StubMetric(v) for name, v in zip(ragas.METRIC_NAMES, (0.9, 0.8, 0.7, 0.6), strict=True)}
    monkeypatch.setattr(ragas, "build_metrics", lambda model: stub_metrics)
    monkeypatch.setattr(ragas, "make_client", lambda: None)
    monkeypatch.setattr(
        ragas, "build_retriever", lambda mode: SimpleNamespace(search=lambda q: [Hit(SAMPLE_CHUNKS[0], 1.0)])
    )
    monkeypatch.setattr(ragas, "answer_question", lambda q, hits, client, model: Answer("A [1].", ["x, p. 1"], ["ctx"]))
    StubMetric.loops.clear()

    questions = ragas.load_questions()[:3]
    progress = tmp_path / "ragas.partial.json"
    summaries = asyncio.run(
        ragas.run_all(["vector", "hybrid_rerank"], questions, "answer-model", "judge-model", progress)
    )

    saved = json.loads(progress.read_text(encoding="utf-8"))  # written after every question, in case of a crash
    assert [len(mode["rows"]) for mode in saved] == [3, 3]
    assert [s["mode"] for s in summaries] == ["vector", "hybrid_rerank"]
    assert summaries[0]["faithfulness"] == 0.9 and summaries[1]["context_recall"] == 0.6
    assert summaries[0]["faithfulness_ci95"] == [0.9, 0.9]  # identical scores, no spread
    assert (summaries[0]["answer_model"], summaries[0]["judge_model"]) == ("answer-model", "judge-model")
    assert len(summaries[0]["rows"]) == 3
    assert len(StubMetric.loops) == 1  # every judge call shared one loop


def test_ragas_runner_keeps_scored_questions_when_the_api_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    ragas = load_script("run_ragas_eval")
    monkeypatch.setattr(ragas, "build_metrics", lambda model: {name: StubMetric(0.5) for name in ragas.METRIC_NAMES})
    monkeypatch.setattr(ragas, "make_client", lambda: None)
    monkeypatch.setattr(
        ragas, "build_retriever", lambda mode: SimpleNamespace(search=lambda q: [Hit(SAMPLE_CHUNKS[0], 1.0)])
    )
    calls: list[str] = []

    def answer_then_fail(question: str, hits: list[Hit], client: object, model: str) -> Answer:
        calls.append(question)
        if len(calls) > 1:
            raise RuntimeError("insufficient_quota")
        return Answer("A [1].", ["x, p. 1"], ["ctx"])

    monkeypatch.setattr(ragas, "answer_question", answer_then_fail)
    monkeypatch.setattr(ragas, "RETRY_PAUSE_SECONDS", 0)
    summaries = asyncio.run(ragas.run_all(["vector", "hybrid_rerank"], ragas.load_questions()[:3], "a", "j"))

    assert len(summaries) == 1  # the second mode never started
    assert len(summaries[0]["rows"]) == 1 and "insufficient_quota" in summaries[0]["stopped_early"]


def test_ragas_summary_skips_unscored_questions_and_pairs_by_id() -> None:
    ragas = load_script("run_ragas_eval")
    nan = float("nan")
    rows = [
        {"id": "a", "faithfulness": 1.0, "answer_relevancy": 0.8, "context_precision": 1.0, "context_recall": 1.0},
        {"id": "b", "faithfulness": nan, "answer_relevancy": 0.6, "context_precision": 0.0, "context_recall": 0.5},
    ]
    summary = ragas.summarise(rows)
    assert summary["faithfulness"] == 1.0  # the NaN is left out, not averaged in
    assert summary["answer_relevancy"] == 0.7
    base = [{**rows[1], "id": "b", "context_recall": 0.0}, {**rows[0], "id": "a", "context_recall": 0.5}]
    assert ragas.paired_gain(rows, base, "context_recall")[0] == 0.5  # matched by id, not by position


def test_ragas_token_counter_adds_up_completions_and_ignores_errors() -> None:
    ragas = load_script("run_ragas_eval")
    counter = ragas.TokenCounter()
    counter.add(json.dumps({"usage": {"prompt_tokens": 1200, "completion_tokens": 80}}).encode())
    snapshot = counter.as_dict()
    counter.add(json.dumps({"usage": {"prompt_tokens": 300, "completion_tokens": 20}}).encode())
    counter.add(json.dumps({"error": {"code": "insufficient_quota"}}).encode())
    counter.add(b"not json")
    assert counter.as_dict() == {"calls": 2, "input_tokens": 1500, "output_tokens": 100}
    assert counter.since(snapshot) == {"calls": 1, "input_tokens": 300, "output_tokens": 20}


@pytest.mark.parametrize("name", ["run_retrieval_eval", "run_guardrail_eval", "run_agent_eval", "run_ragas_eval"])
def test_eval_scripts_import(name: str) -> None:
    assert callable(load_script(name).main)


def test_retrieval_eval_paired_gain_compares_the_same_questions() -> None:
    retrieval_eval = load_script("run_retrieval_eval")
    base = {"ranks": {"a": None, "b": 7, "c": 1, "d": 2}}
    better = {"ranks": {"a": 3, "b": 2, "c": 1, "d": 2}}  # two more questions found in the top 5
    mean, low, high = retrieval_eval.paired_difference(better, base, k=5)
    assert mean == 0.5 and 0.0 <= low <= mean <= high <= 1.0


def test_agent_eval_percentage_parser() -> None:
    agent_eval = load_script("run_agent_eval")
    assert agent_eval.percentages("fell 43.5% to $3,219 million, or -3.4 %") == [43.5, -3.4]


def test_every_calculation_question_has_an_expected_value() -> None:
    agent_eval = load_script("run_agent_eval")
    calc_ids = {
        q.id
        for path in agent_eval.QUESTION_SETS.values()
        for q in agent_eval.load_questions(path)
        if q.type == "calculation"
    }
    assert set(agent_eval.CALC_EXPECTED) == calc_ids


def test_reasoning_model_params_workaround() -> None:
    ragas = load_script("run_ragas_eval")
    llm = SimpleNamespace(_map_openai_params=lambda: {"max_tokens": 4096, "temperature": 0.01, "top_p": 0.9})
    ragas.use_reasoning_model_params(llm)
    assert llm._map_openai_params() == {"max_completion_tokens": 4096, "temperature": 1.0}


@pytest.mark.parametrize(
    ("answer", "expected", "correct"),
    [
        ("Net income fell 43.5% in fiscal 2025.", -43.5, True),
        ("Net income changed by -43.5%.", -43.5, True),
        ("Net income decreased by **43.5%**, while revenue increased 2%.", -43.5, True),
        ("Net income rose 43.5% in fiscal 2025.", -43.5, False),  # right number, wrong direction
        ("Net income grew 17.1%.", 17.1, True),
        ("Net income fell 17.1%.", 17.1, False),  # a rise described as a fall
        ("HOKA was 44.8% of net sales.", 44.8, True),
        ("HOKA was 40.0% of net sales.", 44.8, False),
    ],
)
def test_agent_eval_calc_check_uses_direction(answer: str, expected: float, correct: bool) -> None:
    agent_eval = load_script("run_agent_eval")
    assert agent_eval.calc_correct(answer, expected) is correct


def test_agent_eval_summary_counts_only_questions_that_ran() -> None:
    agent_eval = load_script("run_agent_eval")
    from copilot.agent import Usage

    rows = [
        {"id": "nike-05", "fallback": False, "cited_expected_page": True, "calc_correct": True},
        {"id": "lulu-01", "fallback": True, "cited_expected_page": False},
        {"id": "oos", "fallback": True},
    ]
    summary = agent_eval.summarise(rows, [Usage(), Usage(), Usage()])
    assert (summary["questions"], summary["answer_rate"], summary["citation_hit_rate"]) == (2, 0.5, 1.0)
    assert summary["calc_accuracy"] == 1.0 and summary["out_of_scope_decline_rate"] == 1.0


def test_agent_eval_saves_partial_results_when_the_api_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    agent_eval = load_script("run_agent_eval")
    from copilot.agent import Usage

    class FlakyAgent:
        """Answers the first question, then fails like an account out of credit."""

        def __init__(self, model: str) -> None:
            self.model, self.calls = model, 0

        def run(self, question: str) -> SimpleNamespace:
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("insufficient_quota")
            return SimpleNamespace(
                answer="A [1].", sources=[], tool_calls={}, fallback_triggered=False, fallback_reason="", usage=Usage()
            )

    monkeypatch.setattr(agent_eval, "ResearchAgent", FlakyAgent)
    monkeypatch.setattr(agent_eval, "RETRY_PAUSE_SECONDS", 0)
    monkeypatch.setattr(agent_eval, "load_dotenv", lambda: None)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_agent_eval.py", "--set", "heldout"])
    with pytest.raises(SystemExit, match="Incomplete run"):
        agent_eval.main()

    saved = json.loads((tmp_path / "evals" / "results" / "agent_dev_heldout.json").read_text(encoding="utf-8"))
    assert saved["questions"] == 1 and len(saved["rows"]) == 1
    assert "insufficient_quota" in saved["stopped_early"]


def test_agent_eval_usage_summary() -> None:
    agent_eval = load_script("run_agent_eval")
    from copilot.agent import Usage

    summary = agent_eval.usage_summary([Usage(2, 100, 20, 3.0), Usage(4, 300, 60, 5.0)])
    assert summary["llm_calls"] == 6 and summary["input_tokens"] == 400 and summary["output_tokens"] == 80
    assert summary["mean_seconds_per_question"] == 4.0
    assert summary["total_cost_usd"] is None  # prices not configured
