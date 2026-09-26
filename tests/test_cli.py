"""CLI error handling: no tracebacks for a missing key or an API failure."""

import argparse
import sys

import httpx
import openai
import pytest

from copilot import cli


@pytest.mark.parametrize("command", [cli.cmd_ask, cli.cmd_agent])
def test_llm_commands_explain_a_missing_api_key(monkeypatch: pytest.MonkeyPatch, command) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY is not set"):
        command(argparse.Namespace(question="q", k=5, retriever="vector"))


def test_api_errors_become_a_one_line_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_search(_: argparse.Namespace) -> None:
        raise openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))

    monkeypatch.setattr(cli, "cmd_search", failing_search)
    monkeypatch.setattr(sys, "argv", ["copilot", "search", "anything"])
    with pytest.raises(SystemExit, match="OpenAI API error \\(APIConnectionError\\)"):
        cli.main()
