# Evaluation

How the copilot was measured, every result behind the README's summary, and how to reproduce
them. Each number comes from a file in [`evals/results/`](../evals/results); the ones that need
no LLM are re-measured by CI on every push.

|  | Development set (25) | Held-out set (20) | Harder held-out set (20) |
|---|---|---|---|
| Answer passage in the top 5 search results | **88%** (vector-search baseline 60%) | **90%** (vector-search baseline 95%) | **90%** (vector-search baseline 80%) |
| Questions answered by the agent | **25 / 25** | **20 / 20** | **20 / 20** |
| Answers citing a page that supports them | 24 / 25 | 20 / 20 | 20 / 20 |
| Calculations correct | **4 / 4** | **5 / 5** | **6 / 6** |
| Answers grounded in the retrieved text (RAGAS faithfulness) | 0.91 | 0.93 | not run |
| Out-of-scope questions declined instead of guessed | **8 / 8** | not run | not run |

The development questions were used to choose the design. The two held-out sets were written
afterwards and are only ever scored, never tuned on. Retrieval improved clearly on the harder
development questions and stayed level on the held-out ones; the sections below explain why,
with confidence intervals. Every number above comes from a file in
[`evals/results/`](../evals/results), and the ones that need no LLM are re-measured by CI on every
push.

**Corpus:** athletic apparel and footwear. Nike (FY2025 10-K), Lululemon (FY2024 10-K), Under
Armour (FY2025 10-K), Columbia Sportswear (FY2024 10-K) and Deckers Brands (FY2025 Annual
Report): about 500 pages, split into 1,228 page-level chunks.

Every design decision was measured before it was kept. The evaluation needs no LLM for its
core metric, so it runs in CI, and every result carries a confidence interval because the
question sets are small.

## Question sets and scoring

- **Development set** ([`evals/questions.json`](../evals/questions.json)): 25 questions, five per
  company: 14 lookups, 7 "why" questions and 4 calculations. All settings were chosen on it.
- **Held-out set** ([`evals/heldout_questions.json`](../evals/heldout_questions.json)): 20
  questions, four per company, about facts the development set never asks about. Written after
  the design was fixed and never tuned on. Its answer key was audited once against the whole
  corpus, without looking at any retriever's output, so that other wordings of the same answer
  also count (a table row as well as the sentence that quotes it).
- **Harder held-out set** ([`evals/hard_questions.json`](../evals/hard_questions.json)): 20 more
  questions, four per company, added for v1.6 and aimed at the known weak spots: 7 need a figure
  from a table, 13 need two facts from the same passage, and one avoids the report's wording
  ("Europe, the Middle East and Africa" for "EMEA"). They were written from the raw page text,
  with a plain text search and before any retriever was run on them, and they are never tuned
  on. The vector baseline finds 80%, between the development set (60%) and the first held-out
  set (95%).
- **Comparison set** ([`evals/compare_questions.json`](../evals/compare_questions.json)): 7
  questions that each need evidence from two reports; see
  [Cross-company comparisons](#cross-company-comparisons).
- **Evidence-level scoring.** Each question lists one or more *fact sets*: short strings copied
  from the report that together answer it, such as `46.3 billion` and `51.4 billion`. A
  retrieved chunk counts only if its own text contains a complete fact set, so the metric
  measures whether the model was actually shown the answer. Supporting pages are derived from
  the facts by [`evals/label_pages.py`](../evals/label_pages.py), and a test fails if they drift
  from the documents.
- **Confidence intervals.** 95% bootstrap intervals over the questions. Pipelines are compared
  question by question (paired), which is more sensitive than comparing two averages.

## Retrieval

No LLM involved; CI re-runs this on every push.

Development set (25 questions):

| Retriever | Hit@1 | Hit@5 (95% CI) | MRR | Precision@5 | sec/query (CPU) |
|---|---|---|---|---|---|
| Vector only (baseline) | 0.40 | 0.60 (0.40–0.80) | 0.49 | 0.16 | 0.1 |
| BM25 only | 0.24 | 0.56 (0.36–0.76) | 0.40 | 0.16 | <0.01 |
| Hybrid (RRF) | 0.36 | 0.64 (0.44–0.80) | 0.47 | 0.18 | 0.05 |
| Hybrid + rerank, whole chunks (v1.3) | 0.48 | 0.68 (0.48–0.84) | 0.57 | 0.20 | 3.9 |
| **Hybrid + rerank, best window (default)** | **0.48** | **0.88 (0.76–1.00)** | **0.64** | **0.27** | 7.1 |
| Hybrid + rerank, best window, company-scoped | 0.48 | 0.88 (0.76–1.00) | 0.63 | 0.27 | ≈7 |
| Hybrid + rerank, best window, query expansion (optional) | 0.52 | 0.92 (0.80–1.00) | 0.68 | 0.28 | ≈7 |

Held-out set (20 questions):

| Retriever | Hit@1 | Hit@5 (95% CI) | MRR | Precision@5 |
|---|---|---|---|---|
| Vector only (baseline) | 0.80 | 0.95 (0.85–1.00) | 0.85 | 0.38 |
| BM25 only | 0.35 | 0.60 (0.40–0.80) | 0.47 | 0.27 |
| Hybrid (RRF) | 0.55 | 1.00 | 0.74 | 0.39 |
| Hybrid + rerank, whole chunks (v1.3) | 0.65 | 0.90 (0.75–1.00) | 0.74 | 0.38 |
| **Hybrid + rerank, best window (default)** | **0.75** | **0.90 (0.75–1.00)** | **0.82** | **0.39** |
| Hybrid + rerank, best window, query expansion (optional) | 0.75 | 0.90 (0.75–1.00) | 0.82 | 0.39 |

Harder held-out set (20 questions):

| Retriever | Hit@1 | Hit@5 (95% CI) | MRR | Precision@5 |
|---|---|---|---|---|
| Vector only (baseline) | 0.50 | 0.80 (0.60–0.95) | 0.63 | 0.18 |
| BM25 only | 0.30 | 0.70 (0.50–0.90) | 0.47 | 0.15 |
| Hybrid (RRF) | 0.55 | 0.90 (0.75–1.00) | 0.72 | 0.19 |
| Hybrid + rerank, whole chunks (v1.3) | 0.40 | 0.95 (0.85–1.00) | 0.62 | 0.20 |
| **Hybrid + rerank, best window (default)** | **0.35** | **0.90 (0.75–1.00)** | **0.60** | **0.20** |
| Hybrid + rerank, best window, query expansion (optional) | 0.35 | 0.90 (0.75–1.00) | 0.60 | 0.20 |

*Hit@k: an answer chunk is in the top k. MRR: mean reciprocal rank of the first answer chunk.
Precision@5: share of the top 5 chunks that contain the answer.*

What the three tables say together:

- **On the development set the gain is real.** Question by question, the default pipeline beats
  the vector baseline by +0.28 Hit@5 (95% CI +0.04 to +0.52). The earlier whole-chunk reranker
  gained only +0.08 (−0.12 to +0.28), which is indistinguishable from noise.
- **On the held-out set the pipelines are level.** These questions are easier: most answers sit
  in a single sentence, and vector search alone finds 19 of 20. The default pipeline is within
  noise of the baseline (−0.05 Hit@5, 95% CI −0.20 to +0.10) and ranks answers higher than the
  whole-chunk reranker did (Hit@1 0.65 → 0.75).
- **On the harder held-out set, reranking adds nothing over hybrid search.** Both find 18 of 20
  (+0.10 Hit@5 over the vector baseline, 95% CI −0.10 to +0.30), but the reranker moves the
  answer down a place or two on several questions, so Hit@1 falls from 0.55 to 0.35 and MRR from
  0.72 to 0.60. Most of those are table questions: on the 7 that need a table figure, the
  default pipeline ranks the evidence lower than hybrid search on 4 and higher on 2. The
  cross-encoder was trained on prose, so it might rate number-dense chunks too low. **Checked on
  the development set, the idea does not hold**: there, reranking moves number-dense evidence up
  on 7 of 9 questions and down on 1 ([`evals/rank_by_density.py`](../evals/rank_by_density.py)).
  So the reranker stays as it is, and the harder-set drop in Hit@1 is not explained.
- **Query expansion fixes the one wording gap and changes nothing else.** One synonym group
  ("employ", "employees", "workforce", "teammates" and their forms) is added to the keyword side
  of the query. On the development set it finds the Under Armour headcount question (+0.04
  Hit@5, 95% CI +0.00 to +0.12), which is the question that shaped the list, so the gain is by
  construction. Both held-out sets are unchanged. It is available as the
  `hybrid_rerank_expanded` mode but is not the default
  ([ADR 0012](adr/0012-query-expansion-stays-optional.md)).
- **The honest claim** is that the default pipeline is clearly better on the hard development
  questions and no worse at finding answers elsewhere, but on new questions it does not rank
  them better than plain hybrid search. Part of the development-set gain comes from choosing
  settings on that set. Across all 65 questions it finds 58, against 50 for vector search and
  54 each for hybrid search and whole-chunk reranking.

## Answer quality (RAGAS)

RAGAS scores single-shot answers written from the top 5 passages, without the agent.
`gpt-5.6-terra` writes the answers and `gpt-5.6-luna` judges them: using a different model as
judge matters, because a model tends to rate its own output highly.

| Metric (95% CI) | Vector baseline, dev | Default pipeline, dev | Default pipeline, held-out |
|---|---|---|---|
| Faithfulness | 0.94 (0.90–0.98) | 0.91 (0.85–0.96) | 0.93 (0.86–0.99) |
| Answer relevancy | 0.85 (0.74–0.93) | 0.84 (0.73–0.92) | 0.92 (0.89–0.94) |
| Context precision | 0.61 (0.46–0.75) | 0.76 (0.63–0.87) | 0.80 (0.66–0.92) |
| Context recall | 0.73 (0.57–0.89) | 0.90 (0.78–1.00) | 0.90 (0.75–1.00) |

- **The judge agrees that retrieval improved.** Question by question on the development set,
  context precision rises by +0.15 (95% CI +0.02 to +0.30) and context recall by +0.17 (−0.01 to
  +0.36).
- **Answers stay grounded.** Faithfulness and answer relevancy do not change beyond noise (−0.03
  and −0.01). When the evidence is missing, the answer says so instead of guessing; those
  answers are what score zero on answer relevancy.
- The vector-baseline run crashed after scoring every question but before saving, so its
  per-question scores were recovered from the run log; the results file holds those scores but
  not the answer text.

## Agent and guardrails

`gpt-5.6-luna`, the model the app uses by default:

| Check | Development (25) | Held-out (20) | Harder held-out (20) |
|---|---|---|---|
| Answerable questions answered (not declined) | 25 / 25 | 20 / 20 | 20 / 20 |
| Answers citing a supporting page | 24 / 25 | 20 / 20 | 20 / 20 |
| Calculation questions correct (±0.2 percentage points) | 4 / 4 | 5 / 5 | 6 / 6 |
| Out-of-scope questions declined | 8 / 8 | not run | not run |

- The one answer without an expected citation (Under Armour revenue by region) is correct. It
  cites the detailed regional table on pages 38–39, while the answer key lists only the summary
  on page 33.
- Of the eight out-of-scope questions, the relevance floor stopped two before the model
  answered, and the model's own check declined the other six, including the two on-topic ones
  (Adidas revenue, Nike fiscal 2030).
- The agent writes its own search queries and can search again, so it answers some questions
  whose evidence the retrieval tables count as missed.
- On the harder set the agent also answered the two questions whose evidence the retriever
  ranked below the top 5 (Under Armour's inventory and Deckers' channel growth), by searching
  again; both answers match the key.
- The first 53 questions cost $0.07 in API usage, at about 12 seconds each on a laptop CPU; the
  20 harder ones cost $0.03 (116,110 input and 3,847 output tokens).

The relevance floor needs no LLM, so CI re-measures it on every push with the current retriever:

| Check | Result |
|---|---|
| Development questions clearing the floor | 25 / 25 (lowest score 5.35 vs floor 2.0) |
| Held-out questions clearing the floor | 20 / 20 (lowest score 4.45) |
| Out-of-scope declined by the floor alone | 6 / 8 (World Cup, iPhone, Puma, Skechers…; highest score 1.54) |

## Cross-company comparisons

Seven questions in [`evals/compare_questions.json`](../evals/compare_questions.json) each need
evidence from two reports, for example *Which had the higher gross margin in 2024, Lululemon or
Columbia Sportswear?* Each report is one part of the question, scored with the same evidence
matching as above. The facts reuse the held-out sets (never tuned on) and none from the
development set.

| Retriever | Both reports in one search's top 5 (95% CI) | Top 10 | Reports found in top 10 |
|---|---|---|---|
| Vector only (baseline) | 0.00 | 0.29 | 0.57 |
| Hybrid (RRF) | 0.29 (0.00–0.57) | 0.29 | 0.50 |
| **Hybrid + rerank, best window (default)** | **0.43 (0.14–0.71)** | **0.71** | **0.79** |

The agent (`gpt-5.6-luna`, $0.02 for the run) answered all 7, stated both companies' figures in
all 7, and cited a supporting page from each report in all 7. It searches once per company, so
it covers what a single search on the full question misses.

The first scoring gave the agent 4 of 7 for citations. The three "misses" cited pages that hold
the same facts in the financial tables, such as "Effective tax rate 17.1% 14.9%" on Nike's page
35. Every part of the key was then audited against the whole corpus, not only the misses: pages
holding each part's numbers were found, read, and kept only when they state the same fact (three
were coincidences). `evals/run_comparison_eval.py --rescore` scored the saved answers again
without calling the API.

## What the evaluation taught me

- **My first metric was wrong, and I replaced it.** The original page-level metric counted a hit
  whenever any chunk from a "correct" page was retrieved (over-counting), and relied on a
  hand-made page list that missed valid pages (under-counting). It reported 56% → 76%. When the
  LLM-judged scores did not agree, I traced the gap and rebuilt the metric at the evidence level.
- **A plausible fix that did not help.** Restricting search to the company named in the
  question changed nothing: the remaining misses were the right report, wrong passage.
- **A held-out set changed the story.** On new questions the first reranker did not beat plain
  hybrid search, and the paired interval for its development-set gain included zero. Its
  60% → 68% headline was within noise.
- **The reranker was reading the wrong unit.** The cross-encoder was trained on short
  passages, and a 300-word chunk buries the one sentence that answers the question. Scoring
  each chunk by its best 150-word window lifted development Hit@5 to 0.88 without touching the
  index.
- **My own answer key was too narrow.** The first held-out key accepted one wording per fact
  and scored valid passages as misses. Auditing it against the corpus moved every pipeline up,
  so I now check the key before I blame the retriever.
- **The same mistake, a second time.** The comparison key also listed one wording per fact, and
  the agent's first citation score (4 of 7) was the key's fault, not the agent's. Auditing every
  question, not only the misses, keeps the correction from favouring the model.
- **A model marking its own work is generous.** Faithfulness was 0.97 when the same model wrote
  and judged the answers, and 0.91–0.94 with a separate judge.

## Reproduce the evaluation

```powershell
uv run python evals/run_retrieval_eval.py                 # retrieval, development set (free, about 10 min on CPU)
uv run python evals/run_retrieval_eval.py --set heldout   # retrieval, held-out set (free)
uv run python evals/run_retrieval_eval.py --set hard      # retrieval, harder held-out set (free)
uv run python evals/run_comparison_eval.py                # comparison set, retrieval (free)
uv run python evals/run_comparison_eval.py --agent        # comparison set, agent (about $0.02)
uv run python evals/rank_by_density.py                    # reranked vs hybrid ranks by number density (free)
uv run python evals/run_guardrail_eval.py                 # relevance-floor check (free)
uv run python evals/run_agent_eval.py                     # agent, development set (OpenAI API, about $0.05)
uv run python evals/run_agent_eval.py --set heldout       # agent, held-out set (about $0.03)
uv run python evals/run_agent_eval.py --set hard          # agent, harder held-out set (about $0.03)
uv run python evals/run_ragas_eval.py --final             # RAGAS, development set, both pipelines (about $0.60)
uv run python evals/run_ragas_eval.py --final --set heldout --modes hybrid_rerank   # about $0.22
```
