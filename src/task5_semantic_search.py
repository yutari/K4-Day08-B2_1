"""
Task 5 — Semantic Search Module.
"""

from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "ecommerce_support_docs"

_MODEL = None
_COLLECTION = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def get_collection():
    global _COLLECTION
    if _COLLECTION is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _COLLECTION = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _COLLECTION


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Tìm kiếm ngữ nghĩa sử dụng vector similarity."""
    model = get_model()
    collection = get_collection()

    count = collection.count()
    if count == 0:
        return []

    query_vector = model.encode([query])[0].tolist()
    n_results = min(top_k, count)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results and results.get("documents") and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            score = max(0.0, 1.0 - dist) if dist is not None else 0.5
            output.append({
                "content": doc,
                "score": round(float(score), 4),
                "metadata": meta or {}
            })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    results = semantic_search("quy định trả hàng hoàn tiền shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

