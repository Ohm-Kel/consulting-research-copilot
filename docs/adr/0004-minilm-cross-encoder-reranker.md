# 0004. `ms-marco-MiniLM-L-6-v2` reranker over 30 candidates

- Status: Accepted. The scoring unit was changed by [0006](0006-rerank-by-best-window.md).
- Date: 2026-09-24

## Context

The fused candidates ([0003](0003-hybrid-retrieval-with-rank-fusion.md)) often hold the answer
outside the top 5. A cross-encoder reads the question and the passage together and ranks more
accurately than either first-stage retriever, at the cost of CPU time per candidate.

## Options measured (development set, whole-chunk scoring)

| Reranker (candidates) | Hit@1 | Hit@5 | MRR | sec/query |
|---|---|---|---|---|
| bge-reranker-base (20) | 0.36 | 0.68 | 0.50 | 16.2 |
| bge-reranker-base (10) | 0.40 | 0.64 | 0.49 | 9.7 |
| ms-marco-MiniLM-L-6-v2 (20) | 0.48 | 0.64 | 0.57 | 3.1 |
| **ms-marco-MiniLM-L-6-v2 (30)** | **0.48** | **0.68** | **0.57** | 4.7 |

## Decision

Rerank the top 30 fused candidates with `cross-encoder/ms-marco-MiniLM-L-6-v2` and keep the
best 5. It matched `bge-reranker-base` on Hit@5, beat it on Hit@1 and MRR, and ran about 3.5×
faster on CPU.

## Consequences

- The reranker's scores also drive the relevance floor
  ([0007](0007-relevance-floor-from-measured-scores.md)), so changing the model means
  recalibrating that floor.
- A later held-out check showed that the whole-chunk gain (0.60 → 0.68) was within noise, which
  led to [0006](0006-rerank-by-best-window.md).
