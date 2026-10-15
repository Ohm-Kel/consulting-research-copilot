# 0008. OpenAI Responses API and a plain-Python agent loop

- Status: Accepted
- Date: 2026-09-24 (agent loop); 2026-09-26 (OpenAI provider)

## Context

The agent decides when to search, searches again when results are weak, and calls a
calculator for growth rates and margins. The first LLM provider could not be used because its
account verification failed, so the project moved to OpenAI in v1.1. GPT-5.6 models reject
function tools combined with reasoning on the Chat Completions API.

## Decision

- Use OpenAI through the `openai` package: `gpt-5.6-luna` for the app and development, and
  `gpt-5.6-terra` only for final evaluation answers, with luna as the independent RAGAS judge.
  Model IDs can be changed with `OPENAI_MODEL` and `OPENAI_EVAL_MODEL`.
- Call the Responses API, which supports function tools together with reasoning.
- Write the tool-calling loop in plain Python (at most 6 turns, a 3-minute budget per question
  and a 90-second timeout per call) instead of using an agent framework.

## Consequences

- Every step of the loop is visible and unit-tested with a scripted fake LLM, so the test suite
  runs without an API key.
- Switching provider again means rewriting the request and response handling in `agent.py`;
  there is no abstraction layer.
- Using one model to write and another to judge lowered faithfulness from 0.97 (self-judged)
  to 0.91, the number reported.
