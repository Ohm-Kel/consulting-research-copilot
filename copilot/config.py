"""Project-wide settings. Every tunable number lives here."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("COPILOT_DATA_DIR", ROOT / "data"))
CHROMA_DIR = Path(os.getenv("COPILOT_CHROMA_DIR", ROOT / "chroma_db"))
COLLECTION_NAME = "annual_reports"

# Models
DEV_MODEL = "claude-haiku-4-5"     # all development calls
EVAL_MODEL = "claude-sonnet-4-6"   # final evaluation runs only
LLM_MODEL = os.getenv("COPILOT_LLM_MODEL", DEV_MODEL)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# bge models are trained to see this prefix on queries (not on passages).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Chunking: page-by-page so each chunk has exactly one page number to cite.
CHUNK_WORDS = 300
OVERLAP_WORDS = 50

# Retrieval
TOP_K = 5
RETRIEVER_MODE = os.getenv("COPILOT_RETRIEVER", "hybrid_rerank")  # vector | bm25 | hybrid | hybrid_rerank
# Measured on the eval set (see README): the MiniLM cross-encoder over 30 candidates
# beat BAAI/bge-reranker-base over 20 on every metric and ran ~4x faster on CPU.
RERANKER_MODEL = os.getenv("COPILOT_RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = int(os.getenv("COPILOT_RERANK_CANDIDATES", "30"))  # fused candidates the cross-encoder re-scores

# Agent and guardrails (Stage 3)
MAX_AGENT_TURNS = 6
# Minimum cross-encoder score (ms-marco logit) for a passage to count as relevant.
# Measured: all 25 eval questions score 3.5 or more; unrelated questions score 0.6 or less.
# Recalibrate if you change RERANKER_MODEL (bge-reranker outputs 0-1 probabilities).
RELEVANCE_THRESHOLD = float(os.getenv("COPILOT_RELEVANCE_THRESHOLD", "2.0"))

# Company name and report label for each file in data/.
DOCUMENTS: dict[str, dict[str, str]] = {
    "Nike_FY2025_10K.pdf": {"company": "Nike", "report": "Nike FY2025 Form 10-K (fiscal year ended May 31, 2025)"},
    "Lululemon_FY2024_10K.pdf": {"company": "Lululemon", "report": "Lululemon FY2024 Form 10-K (fiscal year ended February 2, 2025)"},
    "UnderArmour_FY2025_10K.pdf": {"company": "Under Armour", "report": "Under Armour FY2025 Form 10-K (fiscal year ended March 31, 2025)"},
    "Columbia_FY2024_10K.pdf": {"company": "Columbia Sportswear", "report": "Columbia Sportswear FY2024 Form 10-K (fiscal year ended December 31, 2024)"},
    "Deckers_FY2025_AR.pdf": {"company": "Deckers Brands", "report": "Deckers Brands FY2025 Annual Report (fiscal year ended March 31, 2025)"},
}
COMPANIES = sorted({meta["company"] for meta in DOCUMENTS.values()})
