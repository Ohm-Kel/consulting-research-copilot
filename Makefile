# uv run uses the project's .venv and keeps it in step with uv.lock.
PYTHON ?= uv run python
IMAGE ?= consulting-research-copilot

.DEFAULT_GOAL := help
.PHONY: help install data index test lint format eval eval-llm serve docker-build docker-run

help: ## Show available targets
	@grep -E "^[a-z-]+:.*## " Makefile | sed -E "s/:.*## /\t/"

install: ## Install the locked runtime and development dependencies
	uv sync

data: ## Download the five annual reports into data/
	$(PYTHON) scripts/download_data.py

index: ## Build the Chroma index from the reports
	$(PYTHON) -m copilot.cli ingest

test: ## Run the test suite with coverage (no API key needed)
	$(PYTHON) -m pytest -q --cov

lint: ## Check formatting, lint rules and types
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m mypy

format: ## Format code and apply safe lint fixes
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

eval: ## Run the free retrieval and guardrail evaluations
	$(PYTHON) evals/run_retrieval_eval.py
	$(PYTHON) evals/run_retrieval_eval.py --set heldout
	$(PYTHON) evals/run_guardrail_eval.py

eval-llm: ## Run the agent and RAGAS evaluations (uses the OpenAI API)
	$(PYTHON) evals/run_agent_eval.py
	$(PYTHON) evals/run_agent_eval.py --set heldout
	$(PYTHON) evals/run_ragas_eval.py --final
	$(PYTHON) evals/run_ragas_eval.py --final --set heldout --modes hybrid_rerank

serve: ## Start the API on http://localhost:8000
	$(PYTHON) -m uvicorn copilot.api:app --port 8000

docker-build: ## Build the Docker image
	docker build -t $(IMAGE) .

docker-run: ## Run the Docker image with your .env
	docker run -p 8000:8000 --env-file .env $(IMAGE)
