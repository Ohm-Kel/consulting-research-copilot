"""FastAPI service.

uvicorn copilot.api:app --port 8000
POST /query   {"question": "..."}  -> agent answer with citations (needs OPENAI_API_KEY)
POST /search  {"question": "..."}  -> retrieved passages only (no key needed)
GET  /health
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

import openai
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
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


@app.post("/query", response_model=QueryResponse)
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


@app.post("/search", response_model=list[Passage])
def search(request: SearchRequest, retriever: Retriever = Depends(get_retriever)) -> list[Passage]:
    """Return the top passages for a question without calling the LLM."""
    return [
        Passage(citation=h.chunk.citation, company=h.chunk.company, score=round(h.score, 3), text=h.chunk.text)
        for h in retriever.search(request.question, k=request.k)
    ]
