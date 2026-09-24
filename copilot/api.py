"""FastAPI service.

    uvicorn copilot.api:app --port 8000
    POST /query   {"question": "..."}  -> agent answer with citations (needs ANTHROPIC_API_KEY)
    POST /search  {"question": "..."}  -> retrieved passages only (no key needed)
    GET  /health
"""

import os
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from copilot import config
from copilot.agent import ResearchAgent
from copilot.retrieval import Retriever, build_retriever


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["How did Nike's gross margin change in fiscal 2025?"])


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    tool_calls: dict[str, int]
    fallback_triggered: bool
    fallback_reason: str | None = None


class SearchRequest(QueryRequest):
    k: int = Field(default=config.TOP_K, ge=1, le=20)


class Passage(BaseModel):
    citation: str
    company: str
    score: float
    text: str


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return build_retriever()


def get_agent(retriever: Retriever = Depends(get_retriever)) -> ResearchAgent:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "ANTHROPIC_API_KEY is not set; /query is unavailable. /search still works.")
    return ResearchAgent(retriever=retriever)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    load_dotenv()
    if os.getenv("COPILOT_SKIP_WARMUP") != "1":
        get_retriever().search("warm up", k=1)  # load models before the first request
    yield


app = FastAPI(title="Consulting Research Copilot", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "retriever": config.RETRIEVER_MODE, "model": config.LLM_MODEL,
            "llm_available": bool(os.getenv("ANTHROPIC_API_KEY"))}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, agent: ResearchAgent = Depends(get_agent)) -> QueryResponse:
    result = agent.run(request.question)
    return QueryResponse(answer=result.answer, sources=result.sources, tool_calls=result.tool_calls,
                         fallback_triggered=result.fallback_triggered, fallback_reason=result.fallback_reason)


@app.post("/search", response_model=list[Passage])
def search(request: SearchRequest, retriever: Retriever = Depends(get_retriever)) -> list[Passage]:
    return [Passage(citation=h.chunk.citation, company=h.chunk.company, score=round(h.score, 3), text=h.chunk.text)
            for h in retriever.search(request.question, k=request.k)]
