# Engineering

What is in place around the retrieval and agent code. Design decisions and their trade-offs are
in [adr/](adr/README.md); how the copilot is measured is in [evaluation.md](evaluation.md).

| Area | What is in place |
|---|---|
| API | FastAPI: `GET /` (demo page), `POST /query` (agent), `POST /search` (retrieval only, no LLM), `GET /health`, OpenAPI docs at `/docs`. Optional API-key auth (`X-API-Key`), per-client rate limits and an optional daily cap on `/query` (429 with `Retry-After`), LLM errors returned as 502 |
| Tests | Unit tests for ingestion, retrieval, tools, agent, metrics and evaluation runners, and integration tests for the API. The LLM is replaced by a scripted fake, so the suite runs without an API key. CI fails below 93% line coverage |
| Safety | Calculator built on Python's AST (no `eval`), with a size cap and checks for overflow and complex results. A 90-second timeout on every OpenAI call and a 3-minute budget per question. The demo page inserts server responses as text, never as HTML |
| Observability | One JSON log line per question: model, tool calls, LLM calls, tokens, time, estimated cost and any fallback reason. `/query` returns the same usage. The paid evaluation runners count tokens, retry a failed question once and save progress as they go |
| Reproducibility | `uv.lock` pins every dependency, including transitive ones, for local runs, CI and Docker. Every report is checked against a SHA-256 checksum, and comes from a byte-identical copy in the [reports-v1 release](https://github.com/Ohm-Kel/consulting-research-copilot/releases/tag/reports-v1) if its original host refuses the download. The index stores the settings it was built with and refuses to load if the configuration has changed. Bootstrap intervals use a fixed seed |
| Security | API keys only through `.env`, which is never committed. GitHub secret scanning with push protection, and a pre-commit check for private keys. Dependency alerts reviewed: the open chromadb and ragas advisories need components this project does not use (notes in [`pyproject.toml`](../pyproject.toml)) |
| Docker | One image containing the reports, the models and a pre-built index, built from the same lockfile |
| CI | GitHub Actions on every push and pull request: Ruff and mypy → tests with a coverage gate → build index → **retrieval regression gate** (development Hit@5 ≥ 0.84) → held-out and comparison retrieval reports → **guardrail gate**, and in parallel a Docker build with a `/health` and `/search` smoke test |
| Repository | `main` is protected: changes arrive through pull requests with passing `test-and-eval` and `docker` checks. Conventional Commits, a pull-request template, pre-commit hooks (Ruff, mypy), semantic-version tags and a [changelog](../CHANGELOG.md). [CONTRIBUTING.md](../CONTRIBUTING.md) has the workflow and the evaluation rules |
