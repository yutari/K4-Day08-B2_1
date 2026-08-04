"""Regression tests for non-RAG small talk and safe extractive evidence."""

from __future__ import annotations

from src import task10_generation as generation


def test_greeting_returns_a_short_non_rag_reply_without_sources(monkeypatch):
    """Greetings must not retrieve policy chunks or display irrelevant sources."""

    def unexpected_retrieval(*_args, **_kwargs):
        raise AssertionError("Small-talk queries must bypass retrieval")

    def unexpected_provider_call(*_args, **_kwargs):
        raise AssertionError("Small-talk queries must bypass LLM generation")

    monkeypatch.setattr(generation, "retrieve", unexpected_retrieval)
    monkeypatch.setattr(generation, "_generate_from_providers", unexpected_provider_call)

    result = generation.generate_with_citation("Xin ch\u00e0o b\u1ea1n")

    assert result["sources"] == []
    assert result["retrieval_source"] == "none"
    assert result["answer"] != generation.UNVERIFIABLE_ANSWER
    assert len(result["answer"]) <= 240
    assert "[Ngu\u1ed3n:" not in result["answer"]


def test_extractive_fallback_removes_front_matter_and_headings_but_keeps_policy_prose(monkeypatch):
    """Raw indexed Markdown must never leak document metadata into an answer."""
    chunk = {
        "content": """---
doc_id: xin-cho-shopee-c-th-gip-g-cho-bn-77246
title: \"Ch\u00ednh s\u00e1ch tr\u1ea3 h\u00e0ng\"
source_url: https://help.shopee.vn/portal/4/article/77246
retrieved_at: 2026-08-03
document_version: \"not-stated\"
customer_role: buyer
---

# CH\u00cdNH S\u00c1CH QUY \u0110\u1ecaNH: TR\u1ea2 H\u00c0NG HO\u00c0N TI\u1ec0N

## Th\u1eddi h\u1ea1n y\u00eau c\u1ea7u

Ng\u01b0\u1eddi mua c\u00f3 th\u1ec3 y\u00eau c\u1ea7u tr\u1ea3 h\u00e0ng/ho\u00e0n ti\u1ec1n trong v\u00f2ng 15 ng\u00e0y k\u1ec3 t\u1eeb khi nh\u1eadn h\u00e0ng.""",
        "score": 0.92,
        "metadata": {"source": "return-policy.md", "section": "Th\u1eddi h\u1ea1n y\u00eau c\u1ea7u"},
        "source": "hybrid",
    }

    monkeypatch.setattr(generation, "retrieve", lambda *_args, **_kwargs: [chunk])
    monkeypatch.setattr(
        generation,
        "_generate_from_providers",
        lambda _message: "Unsupported answer. [Ngu\u1ed3n: invented.md, M\u1ee5c: Unknown]",
    )

    result = generation.generate_with_citation("Th\u1eddi h\u1ea1n tr\u1ea3 h\u00e0ng l\u00e0 bao l\u00e2u?")

    assert result["citation_validation"] == "extractive_fallback"
    assert "Ng\u01b0\u1eddi mua c\u00f3 th\u1ec3 y\u00eau c\u1ea7u tr\u1ea3 h\u00e0ng/ho\u00e0n ti\u1ec1n trong v\u00f2ng 15 ng\u00e0y" in result["answer"]
    assert "[Ngu\u1ed3n: return-policy.md, M\u1ee5c: Th\u1eddi h\u1ea1n y\u00eau c\u1ea7u]" in result["answer"]
    for forbidden in (
        "doc_id:",
        "title:",
        "source_url:",
        "retrieved_at:",
        "document_version:",
        "customer_role:",
        "# CH\u00cdNH S\u00c1CH",
    ):
        assert forbidden not in result["answer"]
