"""Score-contract tests for hybrid reranking.

These tests deliberately stub the Cross-Encoder so they never download a model.
The UI-facing ``score`` must always be a comparable 0--1 relevance value;
the model's unbounded logit remains available as diagnostic metadata.
"""

from __future__ import annotations

import math

import pytest

from src import task7_reranking as reranking


class _FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.pairs: list[tuple[str, str]] | None = None

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs = pairs
        return self._scores


def _candidate(content: str, rrf_score: float) -> dict:
    return {
        "content": content,
        "score": rrf_score,
        "rrf_score": rrf_score,
        "score_type": "weighted_rrf_rank",
        "metadata": {"source": f"{content}.md"},
    }


def test_cross_encoder_preserves_raw_logit_but_exposes_normalized_public_score(monkeypatch):
    """A raw logit such as 6.7523 must never leak into public ``score``."""
    raw_logits = [6.7523, 0.0, -2.0]
    fake_model = _FakeCrossEncoder(raw_logits)
    monkeypatch.setattr(reranking, "get_cross_encoder", lambda: fake_model)

    results = reranking.rerank_cross_encoder(
        "payment methods",
        [
            _candidate("best match", 0.03),
            _candidate("ambiguous match", 0.02),
            _candidate("weak match", 0.01),
        ],
        top_k=3,
    )

    assert fake_model.pairs == [
        ("payment methods", "best match"),
        ("payment methods", "ambiguous match"),
        ("payment methods", "weak match"),
    ]
    assert [item["content"] for item in results] == [
        "best match",
        "ambiguous match",
        "weak match",
    ]

    by_content = {item["content"]: item for item in results}
    for content, raw_logit in zip(
        ["best match", "ambiguous match", "weak match"], raw_logits
    ):
        item = by_content[content]
        assert item["cross_encoder_score"] == pytest.approx(raw_logit)
        assert item["cross_encoder_raw_score"] == pytest.approx(raw_logit)
        assert item["score_type"] == "cross_encoder_normalized_relevance"
        assert 0.0 <= item["score"] <= 1.0
        assert item["normalized_score"] == pytest.approx(item["score"])
        assert item["score"] == pytest.approx(
            1.0 / (1.0 + math.exp(-raw_logit)), abs=1e-4
        )

    # This proves the UI score is not the original unbounded logit.
    assert by_content["best match"]["score"] < 1.0
    assert by_content["best match"]["cross_encoder_raw_score"] > 1.0


def test_cross_encoder_failure_keeps_rrf_score_semantics_distinguishable(monkeypatch):
    """When reranking is unavailable, callers can identify RRF-derived scores."""

    def _raise_model_error():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(reranking, "get_cross_encoder", _raise_model_error)
    candidates = [
        _candidate("lower rrf", 0.012),
        _candidate("higher rrf", 0.028),
    ]

    results = reranking.rerank_cross_encoder("payment methods", candidates, top_k=2)

    assert [item["content"] for item in results] == ["higher rrf", "lower rrf"]
    assert [item["rrf_score"] for item in results] == [0.028, 0.012]
    for item in results:
        assert item["score_type"] == "weighted_rrf_rank"
        assert item["score"] == pytest.approx(item["rrf_score"])
        assert 0.0 <= item["score"] <= 1.0
        assert "cross_encoder_raw_score" not in item


def test_fuse_rrf_marks_the_public_score_as_weighted_rrf_rank():
    """Fusion output retains a separately named RRF score before reranking."""
    dense = [_candidate("shared", 0.90), _candidate("dense only", 0.80)]
    sparse = [_candidate("shared", 3.20), _candidate("sparse only", 2.10)]

    results = reranking.fuse_rrf([dense, sparse], top_k=3)

    assert results
    for item in results:
        assert item["score_type"] == "weighted_rrf_rank"
        assert item["score"] == pytest.approx(item["rrf_score"])
        assert 0.0 <= item["score"] <= 1.0
        assert "cross_encoder_raw_score" not in item
