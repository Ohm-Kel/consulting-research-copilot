# Contributing

Thanks for taking a look. This guide covers setup, the change workflow and the rules that keep
the evaluation numbers honest.

## Set up

You need [uv](https://docs.astral.sh/uv/getting-started/installation/); it installs Python 3.12
if it is missing.

```powershell
uv sync                                   # .venv with the exact versions in uv.lock
uv run pre-commit install                 # run Ruff and mypy before every commit
uv run python scripts/download_data.py    # the five reports into data/
uv run copilot ingest                     # build the index
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` only if you need the agent or the
LLM-judged evaluations. Everything else runs without a key.

## Make a change

1. Branch off `main` with a prefix that says what the change is: `feat/`, `fix/`, `docs/`,
   `chore/` or `exp/` (an evaluation run or experiment that does not change shipped behaviour).
2. Commit in [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) form, one
   logical change per commit:

   ```
   <type>(<scope>): <imperative summary under 72 characters>
   ```

   Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`,
   `exp`. Add a body when the reason is not obvious, and mark breaking changes with `!` or a
   `BREAKING CHANGE:` footer.
3. Open a pull request; the template asks what changed, why, how you tested it, sample output
   and the linked issue. `main` is protected: merging needs a pull request with passing
   `test-and-eval` and `docker` checks, and nobody can push to `main` directly.

## Checks

The pre-commit hooks run Ruff (lint and format) and mypy on each commit. To run everything by
hand:

```powershell
uv run pre-commit run --all-files    # Ruff lint and format, mypy
uv run pytest --cov                  # tests; fails below 93% coverage
```

CI also builds the index, gates retrieval (development Hit@5 must stay at 0.84 or above) and
the guardrail, reports both held-out sets and the comparison set, and builds and smoke-tests
the Docker image.

## Evaluation rules

- **Never tune on the held-out sets** (`evals/heldout_questions.json`, `evals/hard_questions.json`)
  or the comparison set (`evals/compare_questions.json`). They are scored, not used to choose
  settings, and only development-set misses may shape anything hand-written, such as the
  query-expansion synonyms.
- **Write new held-out questions before measuring the change they will test**, and commit them
  before scoring them.
- **Report development and held-out results together**, with their confidence intervals.
- **A change to the model or pipeline is a `feat:` or `perf:` PR with before-and-after
  metrics** from `evals/run_retrieval_eval.py` (free) and, where answers change, the agent
  evaluation.
- **Check the answer key against the corpus before blaming the retriever or the agent.** Audit
  every question, not only the misses, and report the score before and after a correction.
  `evals/run_comparison_eval.py --rescore` re-scores saved answers without calling the API.
- **Paid evaluations cost real money.** Estimate the cost first: about $0.0014 per agent
  question on `gpt-5.6-luna` and about $0.012 per RAGAS question.
- Changing `RERANKER_MODEL` or `RERANK_WINDOW_WORDS` means recalibrating the relevance floor;
  `evals/run_guardrail_eval.py` reports the score gap.

## Dependencies

Add or upgrade with uv so that `pyproject.toml` and `uv.lock` change together, and commit both:

```powershell
uv add "<package>==<version>"              # runtime dependency
uv add --dev "<package>==<version>"        # development or evaluation tool
uv lock --upgrade-package <package>        # upgrade one locked package
```

Pin exact versions, and re-run the retrieval evaluation after upgrading anything that touches
embeddings, reranking or PDF parsing.

## Releases

1. Check the open [Dependabot alerts](https://github.com/Ohm-Kel/consulting-research-copilot/security/dependabot).
   Upgrade every package that has a fixed release in a `fix/` pull request, and record in
   `pyproject.toml` why each remaining advisory does not apply.
2. In a `chore/release-vX.Y.Z` pull request, move the `Unreleased` entries in `CHANGELOG.md`
   under the new version and date, and bump `__version__` in `src/copilot/__init__.py`.
3. After it merges, wait for CI on `main` to pass, then tag the merge commit with the semantic
   version and publish the release with that changelog section as its notes:

   ```powershell
   git tag -a vX.Y.Z <merge-commit> -m "vX.Y.Z: <summary>"
   git push origin vX.Y.Z
   gh release create vX.Y.Z --verify-tag --notes-file <changelog-section.md>
   ```
