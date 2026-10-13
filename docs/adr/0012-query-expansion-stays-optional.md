# 0012. Query expansion stays an optional mode

- Status: Accepted
- Date: 2026-10-13

## Context

One development question asks how many people Under Armour "employs"; the report counts
"approximately 14,400 teammates". No retriever found it in the top 5, because neither the
keywords nor the embedding connect the two words strongly enough. The agent still answers it,
by searching again with different wording.

## Options

- Rewrite every query with the LLM before searching: general, but it adds cost and latency to
  `/search` and makes the free retrieval evaluation depend on an LLM.
- Expand the keyword (BM25) side of the query with a small list of synonym groups: free,
  deterministic and easy to test, but each group has to be written by hand.
- Do nothing and rely on the agent's second search.

## Decision

Add synonym expansion as the `hybrid_rerank_expanded` mode, with one group
(`config.SYNONYM_GROUPS`: "employ", "employees", "workforce", "headcount", "teammates" and their
forms), and keep `hybrid_rerank` as the default. Only development-set misses may add words to
the list, so the held-out sets stay a fair test.

| Hit@5 | Default | Expanded |
|---|---|---|
| Development (25) | 0.88 | 0.92 (+0.04, 95% CI +0.00 to +0.12) |
| Held-out (20) | 0.90 | 0.90 |
| Harder held-out (20) | 0.90 | 0.90 |

## Consequences

- The development gain is the one question that shaped the list, so it is a gain by
  construction. The held-out sets show only that expansion does no harm; three held-out
  questions use the listed words, and their results did not change.
- Keeping it optional keeps the reported default numbers free of a hand-made fix for one dev
  question. Switching the default is a one-line change (`COPILOT_RETRIEVER=hybrid_rerank_expanded`)
  if more wording gaps turn up.
- A hand-written list does not scale. If gaps become common, LLM query rewriting in the agent
  is the better place to handle them.
