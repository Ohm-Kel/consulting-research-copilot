# 0005. Evidence-level retrieval metric and a held-out set

- Status: Accepted
- Date: 2026-09-26 (held-out set added 2026-09-29)

## Context

The first metric counted a hit when any chunk from a "correct" page was retrieved. It
over-counted (right page, wrong chunk) and under-counted (the hand-made page list missed valid
pages). It reported 56% → 76% Hit@5, and the LLM-judged scores did not agree. All settings had
also been chosen on the same 25 questions, so there was no unbiased check.

## Decision

- Each question lists one or more fact sets: short strings copied from the report that together
  answer it. A retrieved chunk counts only if its own text contains a complete fact set.
  Supporting pages are derived from the facts by `evals/label_pages.py`, and a test fails if
  they drift from the documents.
- Report 95% bootstrap confidence intervals, and compare pipelines question by question
  (paired).
- Add a 20-question held-out set, written after the design was fixed, that is scored but never
  tuned on. CI reports it but does not gate on it.

## Consequences

- The honest numbers are lower than the first metric's, and they are the only ones reported.
- An answer key can be too narrow. The held-out key was audited once against the corpus,
  without looking at retriever output, so that other valid wordings count.
- The metric needs no LLM, so CI re-measures it on every push and fails if development Hit@5
  drops below 0.84.
- 45 questions give wide intervals; more held-out questions are the most useful next step.
