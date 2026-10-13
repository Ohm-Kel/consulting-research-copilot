# One-command setup: the image contains the reports, the models and a built index.
#   docker build -t consulting-research-copilot .
#   docker run -p 8000:8000 --env-file .env consulting-research-copilot
FROM python:3.12-slim

# uv installs the exact versions in uv.lock, including CPU-only torch (see pyproject.toml).
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies get their own layer, so a code change does not reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md LICENSE ./
COPY copilot/ copilot/
COPY scripts/ scripts/
RUN uv sync --locked --no-dev

# Download the five reports, the embedding and reranker models, and build the index at build time.
RUN python scripts/download_data.py \
 && python -m copilot.cli ingest \
 && python -c "from copilot.retrieval import get_reranker; get_reranker()"

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "copilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
