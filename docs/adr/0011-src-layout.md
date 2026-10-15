# 0011. `src/` package layout

- Status: Accepted
- Date: 2026-10-13

## Context

The `copilot` package sat at the repository root. Python puts the current directory on the
import path, so tests and evaluation scripts imported the source folder directly, whether or
not the package was installed correctly. The evaluation scripts also added the repository
root to `sys.path` by hand.

## Decision

Move the package to `src/copilot/`. uv installs it into the project environment (in editable
mode), and tests, evaluation runners and scripts import that installed package. The
`sys.path` workarounds are removed. `config.ROOT` still points at the repository root, where
`data/`, `chroma_db/` and `evals/` live.

## Consequences

- A packaging mistake, such as a missing module or a broken entry point, now fails the tests
  instead of being hidden by the local folder.
- Paths in the Dockerfile, CI cache keys, mypy settings and the README changed once.
- `config.ROOT` assumes the package runs from a checkout (an editable install, as uv and the
  Docker image use). Running it from a built wheel would need `COPILOT_DATA_DIR` and
  `COPILOT_CHROMA_DIR` to be set.
