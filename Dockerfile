# One-command setup: the image contains the reports, the models and a built index.
#   docker build -t consulting-research-copilot .
#   docker run -p 8000:8000 --env-file .env consulting-research-copilot
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only torch first: the default Linux wheel pulls several GB of CUDA libraries.
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY copilot/ copilot/
COPY scripts/ scripts/

# Download the five reports, the embedding and reranker models, and build the index at build time.
RUN python scripts/download_data.py \
 && python -m copilot.cli ingest \
 && python -c "from copilot.retrieval import get_reranker; get_reranker()"

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "copilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
