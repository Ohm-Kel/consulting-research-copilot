"""FastAPI service.

uvicorn copilot.api:app --port 8000
POST /query   {"question": "..."}  -> agent answer with citations (needs OPENAI_API_KEY)
POST /search  {"question": "..."}  -> retrieved passages only (no key needed)
GET  /health

If COPILOT_API_KEYS is set, /query and /search require a matching X-API-Key header.
Both are rate limited per client (API key, or IP address when keys are not used).
"""

import hmac
import math
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import lru_cache

import openai
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from copilot import __version__, config
from copilot.agent import ResearchAgent
from copilot.retrieval import Retriever, build_retriever


class QueryRequest(BaseModel):
    """Body of POST /query."""

    question: str = Field(min_length=3, max_length=500, examples=["How did Nike's gross margin change in fiscal 2025?"])


class QueryResponse(BaseModel):
    """Agent answer with citations, tool usage and fallback status."""

    answer: str
    sources: list[str]
    tool_calls: dict[str, int]
    fallback_triggered: bool
    fallback_reason: str | None = None


class SearchRequest(QueryRequest):
    """Body of POST /search: a question and how many passages to return."""

    k: int = Field(default=config.TOP_K, ge=1, le=20)


class Passage(BaseModel):
    """One retrieved passage with its citation and relevance score."""

    citation: str
    company: str
    score: float
    text: str


class RateLimiter:
    """Allows `per_minute` requests per client in any rolling 60-second window.

    State is kept in memory, so the limit applies per server process."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def seconds_until_allowed(self, client_id: str) -> float:
        """Record a request and return 0 if it is allowed, else how long to wait."""
        now = time.monotonic()
        with self._lock:
            window = self._requests[client_id]
            while window and now - window[0] >= 60:
                window.popleft()
            if len(window) >= self.per_minute:
                return 60 - (now - window[0])
            window.append(now)
            return 0.0

    def reset(self) -> None:
        """Forget all recorded requests."""
        with self._lock:
            self._requests.clear()


query_limiter = RateLimiter(config.QUERY_RATE_LIMIT)
search_limiter = RateLimiter(config.SEARCH_RATE_LIMIT)


def client_identity(request: Request, x_api_key: str | None = Header(default=None)) -> str:
    """Authenticate the caller and return an ID to rate-limit on.

    With no API keys configured the API is open and callers are identified by IP."""
    if not config.API_KEYS:
        return f"ip:{request.client.host if request.client else 'unknown'}"
    if x_api_key is None or not any(hmac.compare_digest(x_api_key, key) for key in config.API_KEYS):
        raise HTTPException(401, "Missing or invalid X-API-Key header.")
    return f"key:{x_api_key}"


def rate_limit(limiter: RateLimiter) -> Callable[[str], None]:
    """Dependency that rejects a caller who is over `limiter`'s quota with 429."""

    def check(client_id: str = Depends(client_identity)) -> None:
        wait = limiter.seconds_until_allowed(client_id)
        if wait > 0:
            raise HTTPException(429, "Rate limit exceeded.", headers={"Retry-After": str(math.ceil(wait))})

    return check


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Build the retriever once and share it across requests."""
    return build_retriever()


def get_agent(retriever: Retriever = Depends(get_retriever)) -> ResearchAgent:
    """Create an agent for a request; respond 503 if no API key is configured."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(503, "OPENAI_API_KEY is not set; /query is unavailable. /search still works.")
    return ResearchAgent(retriever=retriever)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load .env and warm up the models before serving the first request."""
    load_dotenv()
    if os.getenv("COPILOT_SKIP_WARMUP") != "1":
        get_retriever().search("warm up", k=1)  # load models before the first request
    yield


app = FastAPI(title="Consulting Research Copilot", version=__version__, lifespan=lifespan)


@app.exception_handler(openai.APIError)
async def llm_error(_: Request, exc: openai.APIError) -> JSONResponse:
    """Upstream OpenAI failures (rate limits, auth, outages) become a clear 502."""
    return JSONResponse(status_code=502, content={"detail": f"OpenAI API error: {type(exc).__name__}"})


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Report service status, retriever mode, model and whether the LLM is configured."""
    return {
        "status": "ok",
        "version": __version__,
        "retriever": config.RETRIEVER_MODE,
        "model": config.LLM_MODEL,
        "llm_available": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(rate_limit(query_limiter))])
def query(request: QueryRequest, agent: ResearchAgent = Depends(get_agent)) -> QueryResponse:
    """Answer a question with the agent, including citations and fallback status."""
    result = agent.run(request.question)
    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        tool_calls=result.tool_calls,
        fallback_triggered=result.fallback_triggered,
        fallback_reason=result.fallback_reason,
    )


@app.post("/search", response_model=list[Passage], dependencies=[Depends(rate_limit(search_limiter))])
def search(request: SearchRequest, retriever: Retriever = Depends(get_retriever)) -> list[Passage]:
    """Return the top passages for a question without calling the LLM."""
    return [
        Passage(citation=h.chunk.citation, company=h.chunk.company, score=round(h.score, 3), text=h.chunk.text)
        for h in retriever.search(request.question, k=request.k)
    ]
