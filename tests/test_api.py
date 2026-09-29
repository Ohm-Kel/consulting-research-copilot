"""API tests: the retriever and agent are replaced through FastAPI dependency overrides."""

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

os.environ["COPILOT_SKIP_WARMUP"] = "1"

from copilot import api, config  # noqa: E402
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
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(config, "API_KEYS", [])  # open API unless a test configures keys
    api.query_limiter.reset()
    api.search_limiter.reset()
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
    assert body == {
        "answer": "Gross margin fell to 42.7% [1].",
        "sources": ["Nike_FY2025_10K.pdf, p. 38"],
        "tool_calls": {"retrieve_documents": 1},
        "fallback_triggered": False,
        "fallback_reason": None,
    }


def test_query_reports_fallback(client: TestClient) -> None:
    result = AgentResult("The available reports do not contain this information.", [], {}, True, "no relevant passage")
    app.dependency_overrides[get_agent] = lambda: StubAgent(result)
    body = client.post("/query", json={"question": "Who won the World Cup?"}).json()
    assert body["fallback_triggered"] is True and body["sources"] == []


def test_query_without_api_key_returns_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
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


def test_llm_api_errors_become_502(client: TestClient) -> None:
    import httpx
    import openai

    class FailingAgent:
        def run(self, question: str) -> AgentResult:
            request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
            raise openai.APIConnectionError(request=request)

    app.dependency_overrides[get_agent] = FailingAgent
    response = client.post("/query", json={"question": "Nike revenue?"})
    assert response.status_code == 502 and "APIConnectionError" in response.json()["detail"]


def test_api_reports_the_package_version(client: TestClient) -> None:
    from copilot import __version__

    assert client.get("/health").json()["version"] == __version__
    assert app.version == __version__


def test_api_keys_protect_query_and_search(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "API_KEYS", ["secret-1", "secret-2"])
    body = {"question": "margin", "k": 1}
    assert client.post("/search", json=body).status_code == 401
    assert client.post("/search", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/search", json=body, headers={"X-API-Key": "secret-2"}).status_code == 200
    assert client.post("/query", json={"question": "Nike revenue?"}).status_code == 401
    assert client.get("/health").status_code == 200  # health stays open for load balancers


def test_rate_limit_returns_429_with_retry_after(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.search_limiter, "per_minute", 2)
    body = {"question": "margin", "k": 1}
    assert [client.post("/search", json=body).status_code for _ in range(2)] == [200, 200]
    blocked = client.post("/search", json=body)
    assert blocked.status_code == 429
    assert 0 < int(blocked.headers["Retry-After"]) <= 60


def test_rate_limits_are_per_client(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "API_KEYS", ["alice", "bob"])
    monkeypatch.setattr(api.search_limiter, "per_minute", 1)
    body = {"question": "margin", "k": 1}
    assert client.post("/search", json=body, headers={"X-API-Key": "alice"}).status_code == 200
    assert client.post("/search", json=body, headers={"X-API-Key": "alice"}).status_code == 429
    assert client.post("/search", json=body, headers={"X-API-Key": "bob"}).status_code == 200


def test_rate_limiter_window_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(api.time, "monotonic", lambda: clock[0])
    limiter = api.RateLimiter(per_minute=1)
    assert limiter.seconds_until_allowed("c") == 0
    assert limiter.seconds_until_allowed("c") == 60
    clock[0] += 60  # a minute later the first request has left the window
    assert limiter.seconds_until_allowed("c") == 0
