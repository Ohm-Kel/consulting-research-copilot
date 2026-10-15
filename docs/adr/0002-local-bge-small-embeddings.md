# 0002. Local `bge-small-en-v1.5` embeddings

- Status: Accepted
- Date: 2026-09-24

## Context

Chunks are about 300 words (see [0001](0001-page-level-chunks-with-company-header.md)).
Embeddings are computed for every chunk at ingestion and for every query, so a paid API would
add cost and a network dependency to the free part of the evaluation that CI runs.

## Options

- `all-MiniLM-L6-v2`: small and fast, but truncates input at 256 tokens, so it would ignore
  most of each passage.
- `BAAI/bge-small-en-v1.5`: a similar size, reads 512 tokens, and is trained to use a query
  prefix for retrieval.
- A hosted embedding API: a higher quality ceiling, but paid and online.

## Decision

Use `BAAI/bge-small-en-v1.5` through sentence-transformers, run locally on CPU, with the query
prefix the model expects ("Represent this sentence for searching relevant passages: ").

## Consequences

- Embedding is free and offline, so retrieval tests and evaluations run in CI without an API
  key.
- The model downloads once; CI caches it and the Docker image includes it.
- A larger embedding model was not measured. Vector search alone is the 0.60 Hit@5 baseline.
