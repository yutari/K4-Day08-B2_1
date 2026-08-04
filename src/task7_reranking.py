"""Rank fusion and Cross-Encoder reranking for hybrid retrieval."""

from __future__ import annotations

from collections.abc import Sequence

_CROSS_ENCODER_MODEL = None


def get_cross_encoder():
    global _CROSS_ENCODER_MODEL
    if _CROSS_ENCODER_MODEL is None:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _CROSS_ENCODER_MODEL


def fuse_rrf(
    ranked_lists: Sequence[list[dict]],
    *,
    weights: Sequence[float] | None = None,
    top_k: int = 20,
    k: int = 60,
) -> list[dict]:
    """Fuse rank lists with weighted Reciprocal Rank Fusion.

    Scores used for fallback remain in ``dense_score``; the RRF score is only
    used to form a stable candidate order across heterogeneous retrievers.
    """
    weights = weights or [1.0] * len(ranked_lists)
    fusion_scores: dict[str, float] = {}
    candidates: dict[str, dict] = {}
    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, item in enumerate(ranked_list, start=1):
            key = f"{item.get('metadata', {}).get('relative_path', item.get('metadata', {}).get('source', ''))}::{item['content']}"
            fusion_scores[key] = fusion_scores.get(key, 0.0) + float(weight) / (k + rank)
            merged = candidates.setdefault(key, {**item})
            if "dense_score" in item:
                merged["dense_score"] = max(float(item["dense_score"]), float(merged.get("dense_score", 0.0)))
            if "sparse_score" in item:
                merged["sparse_score"] = max(float(item["sparse_score"]), float(merged.get("sparse_score", 0.0)))
    output: list[dict] = []
    for key, rrf_score in sorted(fusion_scores.items(), key=lambda pair: pair[1], reverse=True)[:top_k]:
        item = {**candidates[key], "rrf_score": round(rrf_score, 6), "score": round(rrf_score, 6)}
        output.append(item)
    return output


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Rescore fused candidates with a Cross-Encoder, with deterministic fallback."""
    if not candidates:
        return []
    try:
        scores = get_cross_encoder().predict([(query, item["content"]) for item in candidates])
        ranked = []
        for item, score in zip(candidates, scores):
            updated = {**item, "cross_encoder_score": round(float(score), 4), "score": round(float(score), 4)}
            ranked.append(updated)
        return sorted(ranked, key=lambda item: item["score"], reverse=True)[:top_k]
    except Exception:
        # A model download failure must not make the chatbot unavailable.
        return sorted(candidates, key=lambda item: item.get("rrf_score", item.get("score", 0.0)), reverse=True)[:top_k]


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Backward-compatible unweighted RRF helper."""
    return fuse_rrf(ranked_lists, top_k=top_k, k=k)


def rerank(query: str, candidates: list[dict], top_k: int = 5, method: str = "cross_encoder") -> list[dict]:
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    return sorted(candidates, key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]
