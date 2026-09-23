# Consulting Research Copilot

A question-answering assistant over public company annual reports, built in stages from
foundations to a production service. Answers cite the source document and page, and the
assistant declines when the reports do not support an answer.

**Corpus:** athletic apparel & footwear, five companies: Nike (FY2025), Lululemon (FY2024),
Under Armour (FY2025), Columbia Sportswear (FY2024), Deckers Brands (FY2025).

## Status

| Stage | Content | Release |
|---|---|---|
| 0 | Foundations: API call, embeddings, cosine similarity, PDF chunking | done |
| 1 | Basic RAG with citations | in progress |
| 2 | Evaluation, hybrid search, reranking | planned |
| 3 | Tool-calling agent and guardrails | planned |
| 4 | FastAPI, Docker, CI | planned |

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env          # then paste your ANTHROPIC_API_KEY into .env
python scripts/download_data.py # fetches the five annual reports into data/
```

On macOS/Linux use `source .venv/bin/activate` and `cp` instead of `copy`.

## Stage 0: Foundations

| Script | Concept |
|---|---|
| `stage0/01_hello_claude.py` | One Messages API call to Claude Haiku 4.5 |
| `stage0/02_embeddings.py` | Sentences to 384-dim vectors with `all-MiniLM-L6-v2` |
| `stage0/03_similarity.py` | Cosine similarity by hand in NumPy; ranks sentences against a query |
| `stage0/04_chunk_pdf.py` | PDF text extraction and ~500-word chunks with 50-word overlap, tagged with page number |

Run any of them with `python stage0/<script>.py`.
