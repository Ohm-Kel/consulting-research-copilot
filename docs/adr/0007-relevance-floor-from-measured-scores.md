# 0007. Relevance floor set from measured scores

- Status: Accepted
- Date: 2026-09-24 (recalibrated 2026-10-01 for windowed scoring)

## Context

The assistant must decline questions the reports cannot answer instead of guessing. The
model's own judgement is one check, but a cheap, deterministic check before the LLM can be
tested in CI without an API key.

## Decision

Decline when no retrieved passage's reranker score clears a floor of 2.0, and require every
cited passage to clear it too. With windowed scoring, every answerable development question
scores at least 5.35, and unrelated or other-company questions score at most 1.54. The floor
sits near the low end on purpose: a question that wrongly passes still meets the model's own
check, while one that is wrongly blocked is simply refused.

## Consequences

- In testing, the agent declined 8 of 8 out-of-scope questions (the floor alone stopped 6 of 8
  in the CI check) and answered all 45 answerable ones.
- The floor applies only to the reranker's score scale. Changing `RERANKER_MODEL` or
  `RERANK_WINDOW_WORDS` means recalibrating it, and `evals/run_guardrail_eval.py --strict` fails
  in CI if it no longer separates the two sets.
- On-topic but unanswerable questions (Adidas revenue, Nike fiscal 2030) pass the floor and
  rely on the model's check.
