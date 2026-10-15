# Architecture decision records

Each record explains one decision: the problem, the options considered, what was chosen and
what it costs. Records 0001 to 0009 were written on 2026-10-13 from the design notes kept in
the README since v0.1; the date in each record is when the decision was made.

| # | Decision | Status |
|---|---|---|
| [0001](0001-page-level-chunks-with-company-header.md) | Page-level chunks with a company header | Accepted |
| [0002](0002-local-bge-small-embeddings.md) | Local `bge-small-en-v1.5` embeddings | Accepted |
| [0003](0003-hybrid-retrieval-with-rank-fusion.md) | Hybrid BM25 + vector retrieval with reciprocal rank fusion | Accepted |
| [0004](0004-minilm-cross-encoder-reranker.md) | `ms-marco-MiniLM-L-6-v2` reranker over 30 candidates | Accepted, scoring unit changed by 0006 |
| [0005](0005-evidence-level-retrieval-metric.md) | Evidence-level retrieval metric and a held-out set | Accepted |
| [0006](0006-rerank-by-best-window.md) | Rerank each chunk by its best 150-word window | Accepted |
| [0007](0007-relevance-floor-from-measured-scores.md) | Relevance floor set from measured scores | Accepted |
| [0008](0008-openai-responses-api-plain-agent-loop.md) | OpenAI Responses API and a plain-Python agent loop | Accepted |
| [0009](0009-10k-print-editions.md) | 10-K print editions for two reports | Accepted |
| [0010](0010-uv-and-a-lockfile.md) | uv and a lockfile for dependencies | Accepted |
| [0011](0011-src-layout.md) | `src/` package layout | Accepted |
| [0012](0012-query-expansion-stays-optional.md) | Query expansion stays an optional mode | Accepted |

To add a record, copy the structure of an existing one, take the next number, and link it here.
