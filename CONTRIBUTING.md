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
3. Open a pull request that says what changed, why, how you tested it, and shows sample output,
   and link the issue it addresses. CI must pass before merging.

## Checks

The pre-commit hooks run Ruff (lint and format) and mypy on each commit. To run everything by
hand:

```powershell
uv run pre-commit run --all-files    # Ruff lint and format, mypy
uv run pytest --cov                  # tests; fails below 93% coverage
```

CI also builds the index, gates retrieval (development Hit@5 must stay at 0.84 or above) and
the guardrail, reports the held-out set, and builds and smoke-tests the Docker image.

## Evaluation rules

- **Never tune on the held-out set** (`evals/heldout_questions.json`). It is scored, not used to
  choose settings.
- **Report development and held-out results together**, with their confidence intervals.
- **A change to the model or pipeline is a `feat:` or `perf:` PR with before-and-after
  metrics** from `evals/run_retrieval_eval.py` (free) and, where answers change, the agent
  evaluation.
- **Check the answer key against the corpus before blaming the retriever.**
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

1. Move the `Unreleased` entries in `CHANGELOG.md` under the new version and date.
2. Bump `__version__` in `src/copilot/__init__.py`.
3. Tag the merge commit on `main` with a semantic version (`v1.5.0`) and push the tag.
