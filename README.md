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
| 2 | Evaluation, hybrid search, reranking | v0.2 |
| 3 | Tool-calling agent and guardrails | v0.3 |
| 4 | FastAPI, Docker, CI | in progress |

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

## Stage 2: Evaluation, then Improvement

### Evaluation set

`evals/questions.json` holds 25 questions, five per company: 14 lookups, 7 "why" questions and
4 calculations. Each has a reference answer, the source report, every page that supports the
answer, and a verbatim evidence snippet. `tests/test_eval_set.py` checks that each snippet really
appears on its page, so the answer key cannot drift from the documents.

### Two kinds of scoring

| Script | Needs API key | What it measures |
|---|---|---|
| `evals/run_retrieval_eval.py` | no | Page-level retrieval: is a supporting page among the top-k chunks? Runs in CI on every push. |
| `evals/run_ragas_eval.py` | yes | RAGAS answer quality judged by Claude: faithfulness, answer relevancy, context precision, context recall |

### Improvements

1. **BM25 keyword search** (`rank_bm25`) catches exact names and figures that embeddings blur
   ("HOKA", "demand creation", "$4,689").
2. **Hybrid fusion**: vector and BM25 top-30 lists merged with reciprocal rank fusion (rank-based,
   so the two incompatible score scales never need calibrating).
3. **Cross-encoder reranking**: reads the question and each of the 30 fused candidates together
   and re-orders them.

### Results: retrieval (25 questions, top 5)

| Retriever | Hit@1 | Hit@5 | MRR | Precision@5 | sec/query (CPU) |
|---|---|---|---|---|---|
| Vector only (Stage 1 baseline) | 0.32 | 0.56 | 0.43 | 0.17 | 0.3 |
| BM25 only | 0.24 | 0.60 | 0.39 | 0.18 | 0.01 |
| Hybrid (RRF) | 0.32 | 0.60 | 0.44 | 0.18 | 0.04 |
| **Hybrid + rerank** | **0.40** | **0.76** | **0.54** | **0.24** | 4.4 |

Hybrid + reranking finds a supporting page for 19 of 25 questions versus 14 for the baseline.
Remaining misses are mostly questions whose answer sits in a bullet list of highlights while
many other pages repeat the same terms (e.g. Lululemon revenue growth).

**Reranker choice, measured rather than assumed.** The brief named `bge-reranker-base`. On
this eval set the smaller `ms-marco-MiniLM-L-6-v2` cross-encoder was both better and faster:

| Reranker (candidates) | Hit@5 | MRR | sec/query |
|---|---|---|---|
| bge-reranker-base (20) | 0.72 | 0.49 | ~16 |
| bge-reranker-base (10) | 0.68 | 0.50 | 10.6 |
| ms-marco-MiniLM-L-6-v2 (20) | 0.72 | 0.53 | 3.3 |
| **ms-marco-MiniLM-L-6-v2 (30)** (default) | **0.76** | **0.54** | 4.4 |

Switch back with `COPILOT_RERANKER=BAAI/bge-reranker-base`.

### Results: RAGAS answer quality

Pending the API key. Run:

```powershell
python evals/run_ragas_eval.py --limit 3                              # smoke test, Haiku 4.5
python evals/run_ragas_eval.py --modes vector hybrid_rerank --final   # final table, Sonnet 4.6
```

```powershell
python evals/run_retrieval_eval.py        # reproduce the retrieval table above
python -m copilot.cli search "How fast did HOKA grow?" --retriever hybrid_rerank
```

## Stage 3: Agent and Guardrails

`copilot/agent.py` is a manual tool-use loop over the Claude Messages API (no framework):
Claude decides when to call a tool, the code runs it and returns a `tool_result`, repeating until
Claude answers or the 6-step limit is hit.

| Tool | Purpose |
|---|---|
| `retrieve_documents(query)` | Hybrid + rerank search. Returns passages tagged `[S1]`, `[S2]`... with a relevance score; flags weak results so Claude re-queries with different wording |
| `calculate(expression)` | AST-based evaluator: arithmetic plus `pct_change`, `share`, `cagr`, `round`. No `eval`, no names, no attribute access |

**Fallback guardrail.** The agent returns *"The available reports do not contain this
information."* with `fallback_triggered: true` when any of these holds:

1. **Relevance floor:** no retrieved passage scored above 2.0 (cross-encoder logit). Set from
   data: every answerable eval question scores at least 3.47, and unrelated questions at most 0.56.
2. **Model judgement:** Claude replies `INSUFFICIENT_CONTEXT`. This catches on-topic questions the
   floor cannot, such as another company's figures or a future year.
3. **No citations:** a draft answer that cites no retrieved passage is never returned.

Citations `[S7]` are renumbered `[1]`, `[2]`... and mapped to `file, p. N`.

```powershell
python evals/run_guardrail_eval.py   # relevance floor on real retrieval, no API key
python evals/run_agent_eval.py       # end-to-end agent: answer rate, citations, calc accuracy, declines
python -m copilot.cli agent "By what percentage did Nike's net income fall in fiscal 2025?"
```

Relevance floor results (no LLM):

| Set | Result |
|---|---|
| Answerable questions clearing the floor | 25/25 (lowest score 3.47) |
| Out-of-scope questions declined by the floor alone | 6/8 (World Cup, iPhone, Puma, Skechers...) |
| Left for the model to decline | "Adidas revenue 2024", "Nike revenue fiscal 2030" |
