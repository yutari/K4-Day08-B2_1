"""Hybrid retrieval: parallel Dense/BM25, weighted RRF, reranking and fallback."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import fuse_rrf, rerank_cross_encoder
from src.task8_pageindex_vectorless import pageindex_search

DENSE_TOP_N = 20
SPARSE_TOP_N = 20
DEFAULT_TOP_K = 5
SCORE_THRESHOLD = 0.35


def _annotate(results: list[dict], key: str) -> list[dict]:
    return [{**item, key: item["score"]} for item in results]


def _confidence(candidates: list[dict]) -> float:
    """Mean original dense similarity of the top candidates, in the 0–1 range."""
    dense_scores = [float(item["dense_score"]) for item in candidates if "dense_score" in item]
    return round(sum(dense_scores) / len(dense_scores), 4) if dense_scores else 0.0


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    dense_weight: float = 0.5,
) -> list[dict]:
    """Return the best cited chunks for a query.

    Dense and sparse retrieval execute concurrently.  The confidence gate uses
    original dense cosine similarity, never the much smaller RRF score.
    """
    if not query.strip() or top_k < 1:
        return []
    dense_weight = min(1.0, max(0.0, dense_weight))
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(semantic_search, query, DENSE_TOP_N)
        sparse_future = executor.submit(lexical_search, query, SPARSE_TOP_N)
        try:
            dense_results = _annotate(dense_future.result(), "dense_score")
        except Exception:
            dense_results = []
        try:
            sparse_results = _annotate(sparse_future.result(), "sparse_score")
        except Exception:
            sparse_results = []

    fused = fuse_rrf(
        [dense_results, sparse_results],
        weights=[dense_weight, 1.0 - dense_weight],
        top_k=max(DENSE_TOP_N, SPARSE_TOP_N),
    )
    candidates = rerank_cross_encoder(query, fused, top_k=top_k) if use_reranking else fused[:top_k]
    confidence = _confidence(candidates)
    if not candidates or confidence < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback

    for item in candidates:
        item["source"] = "hybrid"
        item["metadata"] = {**item.get("metadata", {}), "confidence": confidence}
    return candidates
