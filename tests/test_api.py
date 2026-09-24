"""API tests: the retriever and agent are replaced through FastAPI dependency overrides."""

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

os.environ["COPILOT_SKIP_WARMUP"] = "1"

from copilot.agent import AgentResult  # noqa: E402
from copilot.api import app, get_agent, get_retriever  # noqa: E402
from copilot.retrieval import Hit  # noqa: E402
from tests.conftest import SAMPLE_CHUNKS  # noqa: E402


class StubRetriever:
    def search(self, query: str, k: int = 5) -> list[Hit]:
        return [Hit(c, 5.0 - i) for i, c in enumerate(SAMPLE_CHUNKS[:k])]


class StubAgent:
    def __init__(self, result: AgentResult) -> None:
        self.result = result

    def run(self, question: str) -> AgentResult:
        return self.result


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_retriever] = StubRetriever
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and "retriever" in body


def test_query_returns_answer_with_sources(client: TestClient) -> None:
    result = AgentResult("Gross margin fell to 42.7% [1].", ["Nike_FY2025_10K.pdf, p. 38"], {"retrieve_documents": 1})
    app.dependency_overrides[get_agent] = lambda: StubAgent(result)
    body = client.post("/query", json={"question": "Nike gross margin?"}).json()
    assert body == {"answer": "Gross margin fell to 42.7% [1].", "sources": ["Nike_FY2025_10K.pdf, p. 38"],
                    "tool_calls": {"retrieve_documents": 1}, "fallback_triggered": False, "fallback_reason": None}


def test_query_reports_fallback(client: TestClient) -> None:
    result = AgentResult("The available reports do not contain this information.", [], {}, True, "no relevant passage")
    app.dependency_overrides[get_agent] = lambda: StubAgent(result)
    body = client.post("/query", json={"question": "Who won the World Cup?"}).json()
    assert body["fallback_triggered"] is True and body["sources"] == []


def test_query_without_api_key_returns_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    response = client.post("/query", json={"question": "Nike revenue?"})
    assert response.status_code == 503


def test_query_validates_input(client: TestClient) -> None:
    app.dependency_overrides[get_agent] = lambda: StubAgent(AgentResult("unused"))
    assert client.post("/query", json={"question": ""}).status_code == 422
    assert client.post("/query", json={}).status_code == 422


def test_search_returns_passages(client: TestClient) -> None:
    body = client.post("/search", json={"question": "margin", "k": 2}).json()
    assert [p["citation"] for p in body] == ["Nike_FY2025_10K.pdf, p. 38", "Lululemon_FY2024_10K.pdf, p. 31"]
    assert body[0]["score"] == 5.0


def test_claude_api_errors_become_502(client: TestClient) -> None:
    import anthropic
    import httpx2

    class FailingAgent:
        def run(self, question: str) -> AgentResult:
            request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.APIConnectionError(request=request)

    app.dependency_overrides[get_agent] = FailingAgent
    response = client.post("/query", json={"question": "Nike revenue?"})
    assert response.status_code == 502 and "APIConnectionError" in response.json()["detail"]
