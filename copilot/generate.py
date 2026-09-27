"""Generation: answer a question from retrieved chunks, with page citations."""

import re
from dataclasses import dataclass, field

from openai import OpenAI

from copilot import config
from copilot.retrieval import Hit

SYSTEM_PROMPT = """You are a research assistant for management consultants. Answer questions \
using only the numbered excerpts from company annual reports that the user provides.

Rules:
- Cite every factual claim with the excerpt number in square brackets, e.g. [2].
- Quote figures exactly as they appear, with units and the fiscal year they refer to.
- Each excerpt names its report and when that company's fiscal year ended; fiscal years differ
  between companies, so say which fiscal year each figure belongs to.
- If the excerpts do not contain the answer, say so plainly instead of guessing.
- Be concise: a direct answer first, then brief supporting detail."""


@dataclass
class Answer:
    """A generated answer with its citations and the passages it was based on."""

    text: str
    sources: list[str] = field(default_factory=list)  # e.g. "Nike_FY2025_10K.pdf, p. 34"
    contexts: list[str] = field(default_factory=list)  # retrieved passages, kept for evaluation


def format_context(hits: list[Hit]) -> str:
    """Render hits as numbered excerpts the model can cite."""
    return "\n\n".join(
        f"[{i}] {hit.chunk.citation} | {hit.chunk.report}\n{hit.chunk.text}" for i, hit in enumerate(hits, start=1)
    )


def cited_sources(text: str, hits: list[Hit]) -> tuple[str, list[str]]:
    """Map the excerpt numbers the model cited to document/page citations.

    Returns the text renumbered so that [1] is the first source listed, [2] the
    second, and so on (excerpts from the same page share a number), plus that
    source list. Markers that point at no excerpt are removed."""
    citations: list[str] = []

    def renumber(match: re.Match[str]) -> str:
        out = []
        for n in (int(x) for x in re.findall(r"\d+", match.group(0))):
            if 1 <= n <= len(hits):
                citation = hits[n - 1].chunk.citation
                if citation not in citations:
                    citations.append(citation)
                out.append(f"[{citations.index(citation) + 1}]")
        return "".join(dict.fromkeys(out))

    renumbered = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", renumber, text)
    return re.sub(r"(\[\d+\])(\1)+", r"\1", renumbered), citations


def make_client() -> OpenAI:
    """OpenAI client with a bounded timeout; reads OPENAI_API_KEY (loaded from .env by copilot.config)."""
    return OpenAI(timeout=config.LLM_TIMEOUT_SECONDS, max_retries=config.LLM_MAX_RETRIES)


def answer_question(
    question: str, hits: list[Hit], client: OpenAI | None = None, model: str = config.LLM_MODEL
) -> Answer:
    """Ask the model to answer `question` from `hits` and extract its citations."""
    client = client or make_client()
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=4096,  # reasoning models spend part of this budget thinking
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n\n{format_context(hits)}\n\nQuestion: {question}"},
        ],
    )
    text, sources = cited_sources(response.choices[0].message.content or "", hits)
    return Answer(text=text, sources=sources, contexts=[h.chunk.text for h in hits])
