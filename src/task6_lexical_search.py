"""
Task 6 — Lexical Search Module (BM25).
"""

from rank_bm25 import BM25Okapi
import numpy as np
from src.task4_chunking_indexing import load_documents, chunk_documents

_CORPUS = None
_BM25_INDEX = None


def _get_corpus_and_index():
    global _CORPUS, _BM25_INDEX
    if _CORPUS is None or _BM25_INDEX is None:
        docs = load_documents()
        _CORPUS = chunk_documents(docs)
        if not _CORPUS:
            _CORPUS = [{
                "content": "Không tìm thấy nội dung văn bản chính sách",
                "metadata": {"source": "default.md", "type": "news"}
            }]
        tokenized_corpus = [doc["content"].lower().split() for doc in _CORPUS]
        _BM25_INDEX = BM25Okapi(tokenized_corpus)
    return _CORPUS, _BM25_INDEX


def build_bm25_index(corpus: list[dict]):
    """Xây dựng BM25 index từ corpus."""
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Tìm kiếm từ khóa sử dụng BM25."""
    corpus, bm25 = _get_corpus_and_index()
    if not corpus or not bm25:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score_val = float(scores[idx])
        results.append({
            "content": corpus[idx]["content"],
            "score": round(score_val if score_val > 0 else 0.01, 4),
            "metadata": corpus[idx].get("metadata", {})
        })
    return results[:top_k]


if __name__ == "__main__":
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

