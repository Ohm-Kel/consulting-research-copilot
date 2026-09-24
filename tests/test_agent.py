"""Agent loop and fallback guardrails, with a scripted fake Claude and retriever."""

from types import SimpleNamespace
from typing import Any

from copilot.agent import DECLINE_MESSAGE, ResearchAgent, renumber_citations
from copilot.retrieval import Hit
from tests.conftest import SAMPLE_CHUNKS, FakeClaude, text_response

NIKE = SAMPLE_CHUNKS[0]


class FakeRetriever:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.queries: list[str] = []

    def search(self, query: str, k: int = 5) -> list[Hit]:
        self.queries.append(query)
        return self.hits[:k]


def tool_use(name: str, tool_input: dict[str, Any], block_id: str = "tu_1") -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)
    return SimpleNamespace(content=[block], stop_reason="tool_use")


def make_agent(responses: list[Any], hits: list[Hit]) -> tuple[ResearchAgent, FakeClaude, FakeRetriever]:
    fake, retriever = FakeClaude(responses), FakeRetriever(hits)
    return ResearchAgent(retriever=retriever, client=fake, model="test", relevance_threshold=2.0), fake, retriever


def test_agent_retrieves_calculates_and_cites() -> None:
    agent, fake, retriever = make_agent(
        [
            tool_use("retrieve_documents", {"query": "Nike gross margin fiscal 2025"}),
            tool_use("calculate", {"expression": "pct_change(44.6, 42.7)"}, "tu_2"),
            text_response("Gross margin fell 190 bps to 42.7% [S1], a 4.3% relative decline."),
        ],
        [Hit(NIKE, 7.5)],
    )
    result = agent.run("How did Nike's gross margin change?")

    assert not result.fallback_triggered
    assert result.answer == "Gross margin fell 190 bps to 42.7% [1], a 4.3% relative decline."
    assert result.sources == ["Nike_FY2025_10K.pdf, p. 38"]
    assert result.tool_calls == {"retrieve_documents": 1, "calculate": 1}
    assert retriever.queries == ["Nike gross margin fiscal 2025"]
    # the calculator result went back to the model as a tool_result
    calc_result = fake.requests[2]["messages"][-1]["content"][0]
    assert calc_result["tool_use_id"] == "tu_2" and '"result": -4.2601' in calc_result["content"]


def test_fallback_when_no_passage_is_relevant() -> None:
    agent, fake, _ = make_agent(
        [tool_use("retrieve_documents", {"query": "World Cup winner"}), text_response("Argentina won [S1].")],
        [Hit(NIKE, -8.5)],
    )
    result = agent.run("Who won the 2022 World Cup?")
    assert result.fallback_triggered
    assert result.answer == DECLINE_MESSAGE
    assert result.sources == []
    assert "not relevant" in result.fallback_reason or "no retrieved passage" in result.fallback_reason
    assert "weakly related" in fake.requests[1]["messages"][-1]["content"][0]["content"]


def test_fallback_when_model_reports_insufficient_context() -> None:
    agent, _, _ = make_agent(
        [tool_use("retrieve_documents", {"query": "Nike revenue fiscal 2030"}),
         text_response("INSUFFICIENT_CONTEXT: the reports cover fiscal 2025, not 2030.")],
        [Hit(NIKE, 8.0)],
    )
    result = agent.run("What was Nike's revenue in fiscal 2030?")
    assert result.fallback_triggered
    assert result.fallback_reason == "the reports cover fiscal 2025, not 2030"


def test_fallback_when_answer_cites_nothing() -> None:
    agent, _, _ = make_agent(
        [tool_use("retrieve_documents", {"query": "Nike margin"}), text_response("Margins fell.")],
        [Hit(NIKE, 8.0)],
    )
    assert agent.run("Nike margin?").fallback_triggered


def test_fallback_when_step_limit_reached() -> None:
    agent, _, _ = make_agent([tool_use("retrieve_documents", {"query": "q"}, f"tu_{i}") for i in range(6)], [Hit(NIKE, 8.0)])
    result = agent.run("loop forever")
    assert result.fallback_triggered and "step limit" in result.fallback_reason


def test_fallback_when_answer_is_truncated() -> None:
    truncated = SimpleNamespace(content=[SimpleNamespace(type="text", text="Gross margin fell [S1] because")],
                                stop_reason="max_tokens")
    agent, _, _ = make_agent([tool_use("retrieve_documents", {"query": "q"}), truncated], [Hit(NIKE, 8.0)])
    result = agent.run("Nike margin?")
    assert result.fallback_triggered and "max_tokens" in result.fallback_reason


def test_tool_errors_are_reported_to_the_model() -> None:
    agent, fake, _ = make_agent(
        [tool_use("calculate", {"expression": "import os"}), text_response("INSUFFICIENT_CONTEXT")], [Hit(NIKE, 8.0)]
    )
    agent.run("?")
    result_block = fake.requests[1]["messages"][-1]["content"][0]
    assert result_block["is_error"] is True and result_block["content"].startswith("Error:")


def test_renumber_citations_merges_same_page_and_drops_unknown_ids() -> None:
    other_chunk_same_page = SAMPLE_CHUNKS[0].__class__("nike-p38-1", NIKE.source, NIKE.company, NIKE.page, "more")
    sources = {"S1": Hit(SAMPLE_CHUNKS[1], 5), "S2": Hit(NIKE, 5), "S3": Hit(other_chunk_same_page, 5)}
    text, cited = renumber_citations("A [S2]. B [S3][S2]. C [S1, S2]. D [S9].", sources)
    assert text == "A [1]. B [1]. C [2][1]. D ."
    assert cited == ["Nike_FY2025_10K.pdf, p. 38", "Lululemon_FY2024_10K.pdf, p. 31"]


def test_relevance_floor_only_applies_to_reranker_scores() -> None:
    # A plain retriever returns cosine-like scores (0-1); the 2.0 floor must not decline everything.
    fake = FakeClaude([tool_use("retrieve_documents", {"query": "Nike margin"}), text_response("Fell to 42.7% [S1].")])
    agent = ResearchAgent(retriever=FakeRetriever([Hit(NIKE, 0.8)]), client=fake, model="test")
    assert agent.relevance_threshold is None
    result = agent.run("Nike margin?")
    assert not result.fallback_triggered and result.sources == ["Nike_FY2025_10K.pdf, p. 38"]
