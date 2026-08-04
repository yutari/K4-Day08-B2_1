"""Unit tests for the Task 10 citation post-validation guardrail."""

from src import task10_generation as generation


CHUNKS = [
    {
        "content": "Payment methods are documented in this policy.",
        "score": 0.91,
        "metadata": {"source": "payment-policy.md", "section": "Payment methods"},
        "source": "hybrid",
    }
]


def test_valid_citation_is_normalized_to_retrieved_label():
    answer = (
        "The available payment methods are listed in the policy. "
        "[Source: PAYMENT-policy.md, Section: payment methods]"
    )

    validated = generation.validate_and_normalize_citations(answer, CHUNKS)

    assert validated == (
        "The available payment methods are listed in the policy. "
        "[Ngu\u1ed3n: payment-policy.md, M\u1ee5c: Payment methods]"
    )


def test_invented_or_missing_citation_is_rejected():
    invented = "The policy says this. [Ngu\u1ed3n: invented.md, M\u1ee5c: Unknown]"
    uncited = "The policy says this without naming its supporting section."
    partially_cited_bullets = (
        "- This statement is supported. [Ngu\u1ed3n: payment-policy.md, M\u1ee5c: Payment methods]\n"
        "- This separate statement is not cited."
    )

    assert generation.validate_and_normalize_citations(invented, CHUNKS) is None
    assert generation.validate_and_normalize_citations(uncited, CHUNKS) is None
    assert generation.validate_and_normalize_citations(partially_cited_bullets, CHUNKS) is None


def test_generation_uses_extractive_fallback_for_invalid_provider_citation(monkeypatch):
    monkeypatch.setattr(generation, "retrieve", lambda _query, **_kwargs: CHUNKS)
    monkeypatch.setattr(
        generation,
        "_generate_from_providers",
        lambda _message: "Unsupported claim. [Ngu\u1ed3n: invented.md, M\u1ee5c: Unknown]",
    )

    result = generation.generate_with_citation("Which payment methods are available?")

    assert result["citation_validation"] == "extractive_fallback"
    assert "[Ngu\u1ed3n: payment-policy.md, M\u1ee5c: Payment methods]" in result["answer"]
