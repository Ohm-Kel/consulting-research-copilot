# 0003. Hybrid BM25 + vector retrieval with reciprocal rank fusion

- Status: Accepted
- Date: 2026-09-24

## Context

Questions mix meaning ("what drove the margin decline?") with exact tokens: brand names
("HOKA"), fiscal years and figures ("$4,689 million"). Vector search handles the first well
and the second poorly; keyword search is the reverse.

## Decision

- Run vector search (Chroma, in-process `PersistentClient`) and BM25 keyword search
  (`rank_bm25`) for every query, 30 candidates each.
- Merge the two rankings with reciprocal rank fusion (k = 60), which needs no calibration
  between the two systems' scores.

## Consequences

- On the development set, hybrid search alone reached Hit@5 0.64 against 0.60 for vector
  search and 0.56 for BM25; on the held-out set it found 20 of 20. The larger gain came from
  reranking the fused candidates ([0004](0004-minilm-cross-encoder-reranker.md),
  [0006](0006-rerank-by-best-window.md)).
- BM25 keeps its own index in memory, built from the stored chunks at start-up.
- Chroma runs in-process only and its HTTP server is never started, which is why the open
  chromadb server advisories do not apply.
