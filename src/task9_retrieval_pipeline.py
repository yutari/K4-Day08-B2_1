"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.
"""

try:
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search
    from src.task7_reranking import rerank, rerank_rrf
    from src.task8_pageindex_vectorless import pageindex_search
except ImportError:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.25
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Retrieval pipeline hoàn chỉnh với fallback logic."""
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    for item in final_results:
        if "source" not in item:
            item["source"] = "hybrid"

    best_score = dense_results[0]["score"] if dense_results else 0.0
    if best_score < score_threshold or not final_results:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            for f in fallback:
                f["source"] = "pageindex"
            return fallback

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "What payment methods does Shopee support?",
        "How do I request a return or refund?",
        "What evidence do I need for a refund request?",
        "xyzabc123nonsense",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            text_preview = r['content'][:80].encode('ascii', errors='ignore').decode('ascii')
            print(f"  {i}. [{r['score']:.3f}] [{r.get('source', 'hybrid')}] {text_preview}...")


