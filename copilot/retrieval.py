"""Retrieval: find the chunks most relevant to a question."""

from dataclasses import dataclass
from pathlib import Path

import chromadb

from copilot import config
from copilot.embeddings import embed_query
from copilot.ingest import Chunk


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk and its relevance score (higher is better)."""

    chunk: Chunk
    score: float


class VectorRetriever:
    """Dense retrieval: cosine similarity between the query and chunk embeddings."""

    def __init__(self, chroma_dir: Path = config.CHROMA_DIR) -> None:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = client.get_collection(config.COLLECTION_NAME)

    def search(self, query: str, k: int = config.TOP_K) -> list[Hit]:
        result = self.collection.query(query_embeddings=[embed_query(query).tolist()], n_results=k)
        hits = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            chunk = Chunk(chunk_id, meta["source"], meta["company"], int(meta["page"]), text)
            hits.append(Hit(chunk, 1.0 - distance))  # Chroma returns cosine distance
        return hits
