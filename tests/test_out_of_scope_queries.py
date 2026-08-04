"""Regression tests for rejecting queries outside the policy knowledge base."""

from __future__ import annotations

from src import task10_generation as generation
from src import task8_pageindex_vectorless as pageindex


def test_person_lookup_is_rejected_before_retrieval_or_fallback(monkeypatch):
    """An unrelated person lookup must not be answered from a weak text match."""

    def unexpected_retrieval(*_args, **_kwargs):
        raise AssertionError("Out-of-scope queries must be rejected before retrieval")

    def unexpected_generation(*_args, **_kwargs):
        raise AssertionError("Out-of-scope queries must not call an LLM")

    monkeypatch.setattr(generation, "retrieve", unexpected_retrieval)
    monkeypatch.setattr(generation, "_generate_from_providers", unexpected_generation)

    result = generation.generate_with_citation("Nguyễn Nam Hoàng là ai?")

    assert result["answer"] == generation.UNVERIFIABLE_ANSWER
    assert result["sources"] == []
    assert result["retrieval_source"] == "none"
    assert "Dưới đây là các đoạn thông tin liên quan" not in result["answer"]
    assert "[Nguồn:" not in result["answer"]


def test_in_scope_policy_query_is_not_blocked(monkeypatch):
    """The out-of-scope guard must preserve normal policy retrieval/generation."""
    chunks = [
        {
            "content": "Người mua có thể gửi yêu cầu trả hàng trong 15 ngày kể từ khi nhận hàng.",
            "score": 0.92,
            "metadata": {
                "source": "return-policy.md",
                "section": "Thời hạn trả hàng",
            },
            "source": "hybrid",
        }
    ]
    retrieved_queries: list[str] = []

    def retrieve_policy(query, **_kwargs):
        retrieved_queries.append(query)
        return chunks

    monkeypatch.setattr(generation, "retrieve", retrieve_policy)
    monkeypatch.setattr(
        generation,
        "_generate_from_providers",
        lambda _message: (
            "Người mua có thể gửi yêu cầu trả hàng trong 15 ngày kể từ khi nhận hàng. "
            "[Nguồn: return-policy.md, Mục: Thời hạn trả hàng]"
        ),
    )

    query = "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?"
    result = generation.generate_with_citation(query)

    assert retrieved_queries == [query]
    assert result["answer"] != generation.UNVERIFIABLE_ANSWER
    assert result["sources"] == chunks
    assert result["citation_validation"] == "validated"


def test_pageindex_rejects_a_single_weak_keyword_overlap(monkeypatch, tmp_path):
    """A stopword-like one-token overlap must not become fallback evidence."""
    news_dir = tmp_path / "news"
    news_dir.mkdir()
    (news_dir / "policy.md").write_text(
        "# Chính sách giao hàng\n\nNội dung này là hướng dẫn giao hàng của sàn.",
        encoding="utf-8",
    )
    monkeypatch.setattr(pageindex, "STANDARDIZED_DIR", tmp_path)

    results = pageindex.pageindex_search("Nguyễn Nam Hoàng là ai?", top_k=5)

    assert results == []
