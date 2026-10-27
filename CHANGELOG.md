# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). From v1.5.0 the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); earlier tags have two-part names
such as `v1.4` and are kept as published.

## [Unreleased]

### Added

- A cross-company comparison set (`evals/compare_questions.json`): 7 questions that each need
  evidence from two reports, scored per report. One search on the full question reaches both
  reports in its top 10 for 5 of 7 questions with the default pipeline (vector search: 2 of 7).
  The agent, which searches once per company, answered all 7 with both companies' figures and
  a supporting page from each report. `evals/run_comparison_eval.py --rescore` re-scores saved
  answers after a key correction without calling the API.

### Changed

- The README is a third of its former length: each section summarises and links to the detail.
  The full evaluation write-up is now `docs/evaluation.md`, and the engineering table is
  `docs/engineering.md`.
- The stage 0 learning scripts moved from `stage0/` to `notebooks/stage0/`, where exploration
  work belongs.
- `main` is protected: changes need a pull request with passing `test-and-eval` and `docker`
  checks. A pull-request template asks for what changed, why, testing, sample output and the
  linked issue, and `CONTRIBUTING.md` covers the held-out sets and the release steps.

## [1.6.0] - 2026-10-27

### Added

- A harder held-out set of 20 questions (`evals/hard_questions.json`): table figures, two-fact
  answers and different wording. The default pipeline finds 90% in the top 5 (vector baseline
  80%), level with plain hybrid search but with a lower Hit@1 (0.35 against 0.55). The agent
  answered all 20, cited a supporting page for each and got 6 of 6 calculations right. CI
  reports the set without gating on it.
- A demo page at `/` for trying search and the agent in a browser. It needs no build step, loads
  nothing from other sites, and inserts every server response as text, never as HTML.
- `COPILOT_QUERY_DAILY_LIMIT` caps the total `/query` requests per UTC day, so a public
  deployment cannot spend more than a set amount on the LLM. Over the cap, `/query` returns 429
  with `Retry-After` and `/search` keeps working. `/health` reports the limit.
- The `hybrid_rerank_expanded` retriever mode adds synonyms from `config.SYNONYM_GROUPS` to the
  keyword side of the query. Development Hit@5 0.88 → 0.92 (the one question that shaped the
  list); both held-out sets unchanged. Not the default (ADR 0012).
- `evals/rank_by_density.py` compares reranked and hybrid ranks by how number-dense the
  evidence is. On the development set reranking moves number-dense evidence up on 7 of 9
  questions, so the harder-set hint that it demotes tables does not hold.
- A recorded demo in the README (`docs/images/demo.gif`): the demo page answering a question
  about Nike's gross margin with page citations. There is no hosted copy (ADR 0013).

### Fixed

- Report downloads fall back to byte-identical copies in the `reports-v1` release when an
  original host refuses. stocklight.com began returning 403 to GitHub's CI runners, which
  broke every Docker build. The SHA-256 check still decides which file is accepted.

## [1.5.0] - 2026-10-15

### Added

- Test coverage is measured in CI and the build fails below 93% (94% when the gate was added).
- Pre-commit hooks for Ruff (lint and format) and mypy, plus checks for YAML, TOML, merge
  conflicts, private keys and large files.
- `CONTRIBUTING.md` and this changelog.
- Architecture decision records in `docs/adr/` for eleven design decisions, from page-level
  chunking to the `src/` layout.

### Changed

- Dependencies are managed with uv. `pyproject.toml` lists them, `uv.lock` pins all 160 packages
  for Windows, macOS and Linux, and local setup, CI and the Docker image install from the lock.
  `requirements.txt` and `requirements-dev.txt` are gone.
- The package moved to `src/copilot/`, and the evaluation scripts import the installed package
  instead of adding the repository to `sys.path`.
- The Docker image is built and smoke-tested on every branch and pull request, not only on `main`.
- README follows a standard section order (demo, problem and solution, features, architecture,
  tech stack, getting started with an environment-variable table, usage, evaluation, roadmap,
  contributors), and the stale hard-coded test-count badge is gone.
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

[Unreleased]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v1.6.0...HEAD
[1.6.0]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v1.4...v1.5.0
[1.4]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v1.3...v1.4
[1.3]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v1.2...v1.3
[1.2]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v1.1...v1.2
[1.1]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v1.0.1...v1.1
[1.0.1]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v1.0...v1.0.1
[1.0]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v0.3...v1.0
[0.3]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v0.2...v0.3
[0.2]: https://github.com/Ohm-Kel/consulting-research-copilot/compare/v0.1...v0.2
[0.1]: https://github.com/Ohm-Kel/consulting-research-copilot/releases/tag/v0.1
