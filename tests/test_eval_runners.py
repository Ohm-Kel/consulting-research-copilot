"""The eval runner scripts, with the LLM and RAGAS replaced by stubs."""

import asyncio
import importlib.util
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


def test_ragas_runner_scores_all_modes_in_one_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    ragas = load_script("run_ragas_eval")
    stub_metrics = {name: StubMetric(v) for name, v in zip(ragas.METRIC_NAMES, (0.9, 0.8, 0.7, 0.6))}
    monkeypatch.setattr(ragas, "build_metrics", lambda model: stub_metrics)
    monkeypatch.setattr(ragas, "make_client", lambda: None)
    monkeypatch.setattr(ragas, "build_retriever", lambda mode: SimpleNamespace(search=lambda q: [Hit(SAMPLE_CHUNKS[0], 1.0)]))
    monkeypatch.setattr(ragas, "answer_question", lambda q, hits, client, model: Answer("A [1].", ["x, p. 1"], ["ctx"]))
    StubMetric.loops.clear()

    questions = ragas.load_questions()[:3]
    summaries = asyncio.run(ragas.run_all(["vector", "hybrid_rerank"], questions, "test-model"))

    assert [s["mode"] for s in summaries] == ["vector", "hybrid_rerank"]
    assert summaries[0]["faithfulness"] == 0.9 and summaries[1]["context_recall"] == 0.6
    assert len(summaries[0]["rows"]) == 3
    assert len(StubMetric.loops) == 1  # every judge call shared one loop


@pytest.mark.parametrize("name", ["run_retrieval_eval", "run_guardrail_eval", "run_agent_eval", "run_ragas_eval"])
def test_eval_scripts_import(name: str) -> None:
    assert callable(load_script(name).main)


def test_agent_eval_percentage_parser() -> None:
    agent_eval = load_script("run_agent_eval")
    assert agent_eval.percentages("fell 43.5% to $3,219 million, or -3.4 %") == [43.5, -3.4]
    assert set(agent_eval.CALC_EXPECTED) <= {q.id for q in agent_eval.load_questions() if q.type == "calculation"}
