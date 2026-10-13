# 0006. Rerank each chunk by its best 150-word window

- Status: Accepted
- Date: 2026-10-01

## Context

The cross-encoder ([0004](0004-minilm-cross-encoder-reranker.md)) was trained on short web
passages. A 300-word chunk buries the one sentence that answers the question, and the
whole-chunk reranker's gain over vector search (+0.08 Hit@5, 95% CI −0.12 to +0.28) was within
noise.

## Decision

Split each candidate into 150-word windows that overlap by half, score every window, and give
the chunk its highest window score (MaxP). The index, the citations and the text the LLM reads
are unchanged. The size follows the BERT-MaxP setup (Dai & Callan, 2019) rather than the best
cell of a local grid; every size tried landed on the same plateau:

| Window (30 candidates, dev set) | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| Whole chunk (300 words) | 0.48 | 0.68 | 0.57 |
| 96 words | 0.60 | 0.84 | 0.70 |
| 128 words | 0.44 | 0.88 | 0.63 |
| **150 words** | **0.48** | **0.88** | **0.64** |
| 160 words | 0.48 | 0.88 | 0.64 |

## Consequences

- Development Hit@5 rose from 0.60 (vector baseline) to 0.88: a paired gain of +0.28, 95% CI
  +0.04 to +0.52. On the easier held-out set the pipeline is level with the baseline (0.95 →
  0.90, within noise).
- Search takes about 7 seconds on a laptop CPU instead of 4.
- 20 candidates instead of 30 was faster but lost answers (Hit@5 0.76–0.80).
- The relevance floor was recalibrated on the new score scale
  ([0007](0007-relevance-floor-from-measured-scores.md)).
