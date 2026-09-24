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
| 1 | Basic RAG with citations | v0.1 |
| 2 | Evaluation, hybrid search, reranking | in progress |
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

## Stage 1: Basic RAG

```
PDF --pypdf--> page text --300-word chunks, 50 overlap--> bge-small embeddings --> Chroma
Question --embed--> top-5 chunks by cosine --> Claude Haiku 4.5 --> answer with [n] citations
```

- `copilot/ingest.py`: extraction, page-level chunking (each chunk maps to exactly one page), indexing.
  A short header ("Nike annual report, page 38.") is embedded with each chunk so bare
  financial tables still carry the company name.
- `copilot/retrieval.py`: dense vector search over Chroma.
- `copilot/generate.py`: numbered excerpts go to Claude; `[n]` markers in the answer map back to `file, p. N`.
- `copilot/cli.py`: command-line interface.

```powershell
python -m copilot.cli ingest                                    # ~1,230 chunks, a few minutes on CPU
python -m copilot.cli search "What was Nike's gross margin?"    # retrieval only, no API key needed
python -m copilot.cli ask "How did Nike's gross margin change in fiscal 2025?"
python -m pytest
```

Example retrieval (`search`, top 3):

```
1. 0.831  Nike_FY2025_10K.pdf, p. 38
   GROSS MARGIN FISCAL 2025 COMPARED TO FISCAL 2024 For fiscal 2025, our consolidated gross profit
   decreased 14% to $19,790 million compared to $22,887 million for fiscal 2024. Gross margin decreased 190...
2. 0.802  Nike_FY2025_10K.pdf, p. 33
3. 0.792  Nike_FY2025_10K.pdf, p. 42
```

**Why bge-small instead of Stage 0's MiniLM:** `all-MiniLM-L6-v2` truncates input at 256 tokens,
so most of a 300-word financial passage would never be embedded. `bge-small-en-v1.5` has the
same vector size (384) with a 512-token window and stronger retrieval benchmarks.
