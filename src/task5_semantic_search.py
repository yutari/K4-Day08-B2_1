"""Dense retrieval over the local ChromaDB index."""

from __future__ import annotations

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "ecommerce_support_docs"

_MODEL: SentenceTransformer | None = None
_COLLECTION = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def get_collection():
    global _COLLECTION
    if _COLLECTION is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _COLLECTION = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return _COLLECTION


def semantic_search(query: str, top_k: int = 20) -> list[dict]:
    """Return cosine-ranked chunks with a normalised 0–1 dense score."""
    if not query.strip() or top_k < 1:
        return []
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    vector = get_model().encode([query], normalize_embeddings=True)[0].tolist()
    response = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    results: list[dict] = []
    for document, metadata, distance in zip(
        response.get("documents", [[]])[0],
        response.get("metadatas", [[]])[0],
        response.get("distances", [[]])[0],
    ):
        # Cosine distance is 1 - cosine similarity for this collection.
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        results.append(
            {
                "content": document,
                "score": round(score, 4),
                "score_type": "dense_cosine_similarity",
                "metadata": metadata or {},
            }
        )
    return sorted(results, key=lambda item: item["score"], reverse=True)
