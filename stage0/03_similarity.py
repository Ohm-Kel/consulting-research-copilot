"""Stage 0.3: rank sentences against a query with hand-written cosine similarity.

Concept: cosine similarity = dot(a, b) / (|a| * |b|). It measures the angle
between two vectors, ignoring their length: 1.0 means same direction (same
meaning), 0 means unrelated. This is the core operation of vector search.

Run: python stage0/03_similarity.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer

SENTENCES = [
    "Revenue grew 10% year over year, driven by strong demand in North America.",
    "Gross margin declined due to higher promotional activity and markdowns.",
    "The company opened 50 new stores in China during the fiscal year.",
    "Inventory levels were reduced by 7% compared with the prior year.",
    "Our running shoes use a new foam that improves energy return.",
]
QUERY = "Why did profitability fall?"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


model = SentenceTransformer("all-MiniLM-L6-v2")
sentence_vectors = model.encode(SENTENCES)
query_vector = model.encode(QUERY)

scores = [cosine_similarity(query_vector, v) for v in sentence_vectors]
ranking = sorted(zip(scores, SENTENCES), reverse=True)

print(f"Query: {QUERY}\n")
for rank, (score, sentence) in enumerate(ranking, start=1):
    print(f"{rank}. {score:.3f}  {sentence}")
