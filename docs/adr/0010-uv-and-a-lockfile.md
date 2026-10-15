# 0010. uv and a lockfile for dependencies

- Status: Accepted
- Date: 2026-10-13

## Context

Dependencies lived in `requirements.txt` and `requirements-dev.txt`. Direct dependencies were
pinned, but about 140 transitive ones (transformers, tokenizers, huggingface-hub and others)
floated, so each CI run could install different versions from the ones the evaluation was
measured with. Linux needed a separate command to install CPU-only torch before the rest, and
local setup took eight commands.

## Options

- Keep pip and add `pip-tools` to compile a fully pinned requirements file.
- Poetry.
- uv: one tool for Python versions, virtual environments, locking and installs, with
  per-platform package sources.

## Decision

Use uv. Dependencies move into `pyproject.toml` (`[project.dependencies]` and a `dev`
dependency group), `uv.lock` pins every package for Windows, macOS and Linux, and
`.python-version` pins Python 3.12. Torch comes from the PyTorch CPU index on Linux through
`[tool.uv.sources]`. Local setup, CI and the Docker image all install from the same lockfile
with `uv sync --locked`.

The first lock was resolved with the packages installed in the environment that produced the
v1.4 results as constraints, so no version changed; the constraints were then removed.

## Consequences

- Setup is `uv sync` plus the data and index commands, five in total including the clone.
- Upgrading is explicit (`uv lock --upgrade-package <name>`) and shows up as a `uv.lock` diff
  in review.
- Contributors need uv installed; it installs Python 3.12 itself if it is missing.
- CI and Docker no longer pick up new transitive releases silently, so security updates need a
  deliberate lock upgrade.
