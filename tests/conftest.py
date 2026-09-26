"""Shared fixtures. No test calls the real OpenAI API."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from copilot import config
from copilot.ingest import Chunk, build_index

SAMPLE_CHUNKS = [
    Chunk(
        "nike-p38-0",
        "Nike_FY2025_10K.pdf",
        "Nike",
        38,
        "For fiscal 2025, gross margin decreased 190 basis points to 42.7% due to higher discounts.",
    ),
    Chunk(
        "lulu-p31-0",
        "Lululemon_FY2024_10K.pdf",
        "Lululemon",
        31,
        "Company-operated store net revenue increased 14% and e-commerce net revenue increased 6%.",
    ),
    Chunk(
        "colm-p5-0",
        "Columbia_FY2024_10K.pdf",
        "Columbia Sportswear",
        5,
        "We employed approximately 9,450 full-time employees as of December 31, 2024.",
    ),
    Chunk(
        "deck-p40-0",
        "Deckers_FY2025_AR.pdf",
        "Deckers Brands",
        40,
        "HOKA brand net sales increased 23.6% to $2,233 million, driven by strong wholesale demand.",
    ),
    Chunk(
        "ua-p30-0",
        "UnderArmour_FY2025_10K.pdf",
        "Under Armour",
        30,
        "Under Armour recorded restructuring charges related to its 2025 restructuring plan.",
    ),
]


@pytest.fixture(scope="session")
def sample_index(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny Chroma index built from SAMPLE_CHUNKS with the real embedding model."""
    chroma_dir = tmp_path_factory.mktemp("chroma")
    build_index(SAMPLE_CHUNKS, chroma_dir)
    return chroma_dir


@pytest.fixture(scope="session")
def data_available() -> bool:
    return all((config.DATA_DIR / name).exists() for name in config.DOCUMENTS)


class FakeLLM:
    """Stands in for openai.OpenAI. Returns queued responses in order and
    records every request so tests can inspect what was sent."""

    def __init__(self, responses: list[Any]) -> None:
        self.queue = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))  # generate.py
        self.responses = SimpleNamespace(create=self._create)  # agent.py (Responses API)

    def _create(self, **kwargs: Any) -> Any:
        # snapshot the conversation as it was when this request was sent
        self.requests.append(
            {**kwargs, "messages": list(kwargs.get("messages", [])), "input": list(kwargs.get("input", []))}
        )
        return self.queue.pop(0)


def completion(
    content: str | None = None, tool_calls: list[Any] | None = None, finish_reason: str | None = None
) -> SimpleNamespace:
    """A Chat Completions response with one choice."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    reason = finish_reason or ("tool_calls" if tool_calls else "stop")
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=reason)])


def text_response(text: str) -> SimpleNamespace:
    """A response containing only a text answer."""
    return completion(content=text)


def _item(**fields: Any) -> SimpleNamespace:
    """A Responses API output item (supports .model_dump like the SDK objects)."""
    return SimpleNamespace(**fields, model_dump=lambda **_: dict(fields))


def agent_response(
    text: str = "",
    calls: list[tuple[str, str, str]] | None = None,
    status: str = "completed",
    reason: str | None = None,
) -> SimpleNamespace:
    """A Responses API response. `calls` are (call_id, tool name, JSON arguments)."""
    output = [_item(type="function_call", call_id=cid, name=name, arguments=args) for cid, name, args in calls or []]
    if text:
        output.append(_item(type="message", content=text))
    details = SimpleNamespace(reason=reason) if reason else None
    return SimpleNamespace(output=output, output_text=text, status=status, incomplete_details=details)
