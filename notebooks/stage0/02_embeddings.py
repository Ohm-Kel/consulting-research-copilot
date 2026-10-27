"""Stage 0.2: turn sentences into vectors (embeddings).

Concept: an embedding model maps text to a fixed-length vector so that texts
with similar meaning end up close together. all-MiniLM-L6-v2 outputs 384 numbers
per sentence and runs locally for free.

Run: python stage0/02_embeddings.py
"""

from sentence_transformers import SentenceTransformer

SENTENCES = [
    "Revenue grew 10% year over year, driven by strong demand in North America.",
    "Gross margin declined due to higher promotional activity and markdowns.",
    "The company opened 50 new stores in China during the fiscal year.",
    "Inventory levels were reduced by 7% compared with the prior year.",
    "Our running shoes use a new foam that improves energy return.",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(SENTENCES)

print(f"Embeddings shape: {embeddings.shape}")  # (5 sentences, 384 dimensions)
print(f"First 5 values of sentence 1: {embeddings[0][:5]}")
