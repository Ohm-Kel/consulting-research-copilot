# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). From v1.5.0 the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); earlier tags have two-part names
such as `v1.4` and are kept as published.

## [Unreleased]

### Added

- Test coverage is measured in CI and the build fails below 93% (94% when the gate was added).
- Pre-commit hooks for Ruff (lint and format) and mypy, plus checks for YAML, TOML, merge
  conflicts, private keys and large files.
- `CONTRIBUTING.md` and this changelog.

### Changed

- Dependencies are managed with uv. `pyproject.toml` lists them, `uv.lock` pins all 160 packages
  for Windows, macOS and Linux, and local setup, CI and the Docker image install from the lock.
  `requirements.txt` and `requirements-dev.txt` are gone.
- The package moved to `src/copilot/`, and the evaluation scripts import the installed package
  instead of adding the repository to `sys.path`.
- The Docker image is built and smoke-tested on every branch and pull request, not only on `main`.
- README restructured for readers skimming the repo; the CLI help lists every retriever mode.
- CI actions updated to their Node 24 versions.

### Security

- Documented why the open chromadb and ragas advisories do not affect this project.

## [1.4] - 2026-10-01

### Added

- Held-out set of 20 questions, scored but never tuned on.
- 95% bootstrap confidence intervals on every retrieval metric.
- API-key authentication (`X-API-Key`) and per-client rate limits on `/query` and `/search`.
- Token, time and cost tracking per question in logs, API responses and evaluation summaries.
- mypy type checking in CI.

### Changed

- The reranker scores each candidate by its best 150-word window (MaxP). Hit@5 on the
  development set rose from 0.60 (vector baseline) to 0.88; on the held-out set it is level
  with the baseline (0.95 → 0.90, within noise).
- RAGAS answers are judged by a separate model instead of the one that wrote them.
- The index records the settings it was built with and refuses to load if they change.
- Paid evaluation runners save partial results, retry a failed question once and count tokens.

### Fixed

- The held-out answer key accepts other valid wordings of the same fact.

## [1.3] - 2026-09-29

### Added

- SHA-256 checksums for every downloaded report.
- A timeout on every OpenAI call and a time budget per question.
- Fiscal-year labels on retrieved passages.
- Tests for the reranked retriever and the retriever factory.

### Changed

- Cited passages must clear the relevance floor.
- Calculation answers are checked for direction (increase or decrease).
- The API reads its version from the package.
- CI runs on branch pushes only and cancels superseded runs.

### Removed

- An unused config value and tool error flag.

## [1.2] - 2026-09-26

### Added

- MIT licence, `pyproject.toml` packaging with a `copilot` command, and a Makefile.
- Ruff formatting and linting, and docstrings on public functions and classes.
- A clear CLI error when the API key is missing.

### Changed

- Project renamed to consulting-research-copilot; README rewritten and its diagram redrawn.

## [1.1] - 2026-09-26

### Added

- Full LLM-judged results for the agent and RAGAS.

### Changed

- LLM provider switched to OpenAI (gpt-5.6-luna for development, gpt-5.6-terra for final runs).
- Retrieval is scored at evidence level: a passage counts only if it contains the full answer.
  This replaced a page-level metric that both over- and under-counted.

## [1.0.1] - 2026-09-24

### Fixed

- Hardened the calculator, citation handling and API error responses.

## [1.0] - 2026-09-24

### Added

- FastAPI service (`/query`, `/search`, `/health`), Dockerfile, and GitHub Actions CI with
  evaluation regression gates.
- README with the architecture and results.

## [0.3] - 2026-09-24

### Added

- Tool-calling agent with retrieval and calculator tools, and a guardrail that declines
  questions the reports do not support.

## [0.2] - 2026-09-24

### Added

- 25-question evaluation set, hybrid BM25 + vector retrieval and a cross-encoder reranker.

## [0.1] - 2026-09-24

### Added

- RAG pipeline over five annual reports with page-level citations and a CLI.
- Foundation scripts in `stage0/`.

[Unreleased]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v1.4...HEAD
[1.4]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v1.3...v1.4
[1.3]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v1.2...v1.3
[1.2]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v1.1...v1.2
[1.1]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v1.0.1...v1.1
[1.0.1]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v1.0...v1.0.1
[1.0]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v0.3...v1.0
[0.3]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v0.2...v0.3
[0.2]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v0.1...v0.2
[0.1]: https://github.com/Ohm-Kel/consulting-research-copilot/releases/tag/v0.1
