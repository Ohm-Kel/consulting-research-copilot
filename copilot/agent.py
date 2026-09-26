"""Stage 3: a tool-calling research agent with a fallback guardrail.

Loop: the model decides which tools to call (retrieve_documents, calculate), we run
them and return the results, until the model produces a final answer. Three
guardrails decide whether that answer is returned or replaced by a decline:

1. Relevance floor: if no retrieved passage ever scored above
   RELEVANCE_THRESHOLD, the reports do not cover the question.
2. Model judgement: the model replies INSUFFICIENT_CONTEXT when the passages are
   on-topic but do not contain the answer (e.g. a future fiscal year).
3. Citations: an answer that cites no retrieved source is not returned.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from copilot import config
from copilot.generate import make_client
from copilot.retrieval import Hit, Retriever, build_retriever
from copilot.tools import TOOL_SCHEMAS, ToolError, calculate, retrieve_documents

DECLINE_MESSAGE = "The available reports do not contain this information."
INSUFFICIENT_MARKER = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""You are a research assistant for management consultants. You answer questions \
about five athletic apparel and footwear companies using only their annual reports, which you \
search with the retrieve_documents tool.

How to work:
- Always search before answering. Use specific queries that name the company.
- If the passages do not answer the question, search again with different wording (at most 3 searches).
- Use the calculate tool for any growth rate, margin, share or other derived figure.
- Cite every factual claim with the source IDs of the passages it comes from, e.g. [S2] or [S1][S4].
- Quote figures exactly, with units and fiscal year.
- Answer in a few sentences: the direct answer first, then the supporting detail or drivers.

If the passages do not contain enough information to answer, reply with exactly \
{INSUFFICIENT_MARKER} followed by one sentence on what is missing. Never guess or use outside knowledge."""


def renumber_citations(text: str, sources: dict[str, Hit]) -> tuple[str, list[str]]:
    """Replace [S7]-style source IDs with [1], [2]... in order of first use and
    return the matching list of 'file, p. N' citations. Chunks from the same page
    share a number; IDs that were never retrieved are dropped."""
    citations: list[str] = []

    def numbers_for(match: re.Match[str]) -> str:
        out = []
        for sid in re.findall(r"S\d+", match.group(0)):
            hit = sources.get(sid)
            if hit is None:
                continue
            if hit.chunk.citation not in citations:
                citations.append(hit.chunk.citation)
            out.append(f"[{citations.index(hit.chunk.citation) + 1}]")
        return "".join(dict.fromkeys(out))

    renumbered = re.sub(r"\[S\d+(?:\s*,\s*S\d+)*\]", numbers_for, text)
    return re.sub(r"(\[\d+\])(\1)+", r"\1", renumbered), citations


@dataclass
class AgentResult:
    answer: str
    sources: list[str] = field(default_factory=list)
    tool_calls: dict[str, int] = field(default_factory=dict)
    fallback_triggered: bool = False
    fallback_reason: str | None = None
    contexts: list[str] = field(default_factory=list)


class ResearchAgent:
    def __init__(
        self,
        retriever: Retriever | None = None,
        client: OpenAI | None = None,
        model: str = config.LLM_MODEL,
        relevance_threshold: float | None = None,
        max_turns: int = config.MAX_AGENT_TURNS,
    ) -> None:
        """`relevance_threshold` defaults to config.RELEVANCE_THRESHOLD when the
        retriever returns cross-encoder scores (the scale it was calibrated on) and
        is disabled for other retrievers, whose scores use different scales."""
        self.retriever = retriever or build_retriever()
        if relevance_threshold is None and getattr(self.retriever, "cross_encoder_scores", False):
            relevance_threshold = config.RELEVANCE_THRESHOLD
        self.client = client or make_client()
        self.model = model
        self.relevance_threshold = relevance_threshold
        self.max_turns = max_turns

    def run(self, question: str) -> AgentResult:
        sources: dict[str, Hit] = {}  # source ID (S1, S2...) -> passage, across all searches
        tool_calls: dict[str, int] = {}
        best_score = float("-inf")
        # Responses API: function tools work together with reasoning on GPT-5.6
        # models (Chat Completions rejects that combination).
        items: list[Any] = [{"role": "user", "content": question}]
        final_text = ""

        for _ in range(self.max_turns):
            response = self.client.responses.create(
                model=self.model, instructions=SYSTEM_PROMPT, input=items, tools=TOOL_SCHEMAS, max_output_tokens=4096
            )
            if response.status == "incomplete":
                reason = getattr(response.incomplete_details, "reason", None) or "unknown"
                return self._decline(f"the model stopped early ({reason})", tool_calls, sources)
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                final_text = (response.output_text or "").strip()
                break

            # Send back everything the model produced (reasoning and calls), then one output per call.
            items.extend(item.model_dump(exclude_none=True) for item in response.output)
            for call in calls:
                tool_calls[call.name] = tool_calls.get(call.name, 0) + 1
                try:
                    arguments = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    content, hits = "Error: arguments were not valid JSON", []
                else:
                    content, _, hits = self._execute(call.name, arguments, sources)
                for hit in hits:
                    best_score = max(best_score, hit.score)
                items.append({"type": "function_call_output", "call_id": call.call_id, "output": content})
        else:
            return self._decline("the agent reached its step limit without answering", tool_calls, sources)

        # Guardrails, cheapest and most certain first.
        if self.relevance_threshold is not None and best_score < self.relevance_threshold:
            return self._decline("no retrieved passage was relevant to the question", tool_calls, sources)
        if INSUFFICIENT_MARKER in final_text:
            reason = (
                final_text.replace(INSUFFICIENT_MARKER, "").strip(" :.-\n")
                or "the model judged the context insufficient"
            )
            return self._decline(reason, tool_calls, sources)
        answer, cited = renumber_citations(final_text, sources)
        if not cited:
            return self._decline("the draft answer cited no retrieved source", tool_calls, sources)
        return AgentResult(answer, cited, tool_calls, False, None, [h.chunk.text for h in sources.values()])

    def _execute(self, name: str, tool_input: dict[str, Any], sources: dict[str, Hit]) -> tuple[str, bool, list[Hit]]:
        """Run one tool call. Returns (content for the model, is_error, hits)."""
        try:
            if name == "retrieve_documents":
                hits = retrieve_documents(self.retriever, str(tool_input.get("query", "")))
                lines = []
                for hit in hits:
                    sid = next((k for k, v in sources.items() if v.chunk.chunk_id == hit.chunk.chunk_id), None)
                    if sid is None:
                        sid = f"S{len(sources) + 1}"
                        sources[sid] = hit
                    label = f"[{sid}] {hit.chunk.citation} ({hit.chunk.company}) relevance={hit.score:.2f}"
                    lines.append(f"{label}\n{hit.chunk.text}")
                if not hits or (
                    self.relevance_threshold is not None and max(h.score for h in hits) < self.relevance_threshold
                ):
                    lines.append("NOTE: these results look weakly related to the query. Try different wording.")
                return "\n\n".join(lines) or "No passages found.", False, hits
            if name == "calculate":
                value = calculate(str(tool_input.get("expression", "")))
                return json.dumps({"expression": tool_input.get("expression"), "result": round(value, 4)}), False, []
            return f"Unknown tool {name!r}", True, []
        except ToolError as exc:
            return f"Error: {exc}", True, []

    @staticmethod
    def _decline(reason: str, tool_calls: dict[str, int], sources: dict[str, Hit]) -> AgentResult:
        return AgentResult(DECLINE_MESSAGE, [], tool_calls, True, reason, [h.chunk.text for h in sources.values()])
