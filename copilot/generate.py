"""Generation: answer a question from retrieved chunks, with page citations."""

import re
from dataclasses import dataclass, field

import anthropic
from dotenv import load_dotenv

from copilot import config
from copilot.retrieval import Hit

SYSTEM_PROMPT = """You are a research assistant for management consultants. Answer questions \
using only the numbered excerpts from company annual reports that the user provides.

Rules:
- Cite every factual claim with the excerpt number in square brackets, e.g. [2].
- Quote figures exactly as they appear, with units and the fiscal year they refer to.
- If the excerpts do not contain the answer, say so plainly instead of guessing.
- Be concise: a direct answer first, then brief supporting detail."""


@dataclass
class Answer:
    text: str
    sources: list[str] = field(default_factory=list)  # e.g. "Nike_FY2025_10K.pdf, p. 34"
    contexts: list[str] = field(default_factory=list)  # retrieved passages, kept for evaluation


def format_context(hits: list[Hit]) -> str:
    """Render hits as numbered excerpts the model can cite."""
    return "\n\n".join(
        f"[{i}] {hit.chunk.citation} ({hit.chunk.company})\n{hit.chunk.text}" for i, hit in enumerate(hits, start=1)
    )


def cited_sources(text: str, hits: list[Hit]) -> list[str]:
    """Map the [n] markers the model used back to document/page citations."""
    numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    sources = [hits[n - 1].chunk.citation for n in numbers if 1 <= n <= len(hits)]
    return list(dict.fromkeys(sources))  # de-duplicate, keep order


def make_client() -> anthropic.Anthropic:
    load_dotenv()
    return anthropic.Anthropic()


def answer_question(
    question: str, hits: list[Hit], client: anthropic.Anthropic | None = None, model: str = config.LLM_MODEL
) -> Answer:
    """Ask Claude to answer `question` from `hits` and extract its citations."""
    client = client or make_client()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Excerpts:\n\n{format_context(hits)}\n\nQuestion: {question}"}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return Answer(text=text, sources=cited_sources(text, hits), contexts=[h.chunk.text for h in hits])
