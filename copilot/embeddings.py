"""Local sentence-transformers embeddings, loaded once and shared."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from copilot import config


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed documents. Vectors are L2-normalised so dot product = cosine."""
    return get_model().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 100)


def embed_query(query: str) -> np.ndarray:
    """Embed a search query with the bge query instruction."""
    return get_model().encode(config.QUERY_PREFIX + query, normalize_embeddings=True)
