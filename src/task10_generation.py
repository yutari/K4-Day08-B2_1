"""Grounded answer generation with source-aware citations and provider failover."""

from __future__ import annotations

import os
from collections.abc import Sequence

from dotenv import load_dotenv

from src.task9_retrieval_pipeline import DEFAULT_TOP_K, retrieve

load_dotenv()

TEMPERATURE = 0.2
TOP_P = 0.9
SYSTEM_PROMPT = """Bạn là trợ lý về chính sách thương mại điện tử.
Chỉ trả lời bằng dữ kiện có trong CONTEXT. Mỗi đoạn thông tin phải có trích dẫn
ngay sau câu theo đúng một nhãn được cung cấp: [Nguồn: tên_tệp, Mục: tên_mục].
Nếu CONTEXT không có bằng chứng, trả lời đúng câu: "Tôi không thể xác minh thông
tin này từ nguồn hiện có." Không suy đoán, không bịa đặt và luôn trả lời tiếng Việt."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Place high-ranking evidence at both ends of the context window."""
    if len(chunks) <= 2:
        return list(chunks)
    return chunks[::2] + chunks[1::2][::-1]


def _citation_label(chunk: dict, index: int) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source", f"Source_{index}.md")
    section = metadata.get("section", "Tổng quan")
    return f"[Nguồn: {source}, Mục: {section}]"


def format_context(chunks: list[dict]) -> str:
    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        context_parts.append(
            f"[TÀI LIỆU {index}]\n"
            f"Citation bắt buộc: {_citation_label(chunk, index)}\n"
            f"Nguồn: {metadata.get('source', 'unknown')} | Mục: {metadata.get('section', 'Tổng quan')}\n"
            f"{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(context_parts)


def _call_openai_compatible(*, api_key: str, base_url: str | None, model: str, user_message: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return (response.choices[0].message.content or "").strip()


def _generate_from_providers(user_message: str) -> str:
    """Try configured providers in order; any provider failure safely moves on."""
    providers = [
        (os.getenv("GEMINI_API_KEY"), "https://generativelanguage.googleapis.com/v1beta/openai/", os.getenv("GEMINI_MODEL", "gemini-2.0-flash")),
        (os.getenv("OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1", os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")),
        (os.getenv("OPENAI_API_KEY"), None, os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
    ]
    for api_key, base_url, model in providers:
        if not api_key:
            continue
        try:
            answer = _call_openai_compatible(api_key=api_key, base_url=base_url, model=model, user_message=user_message)
            if answer:
                return answer
        except Exception:
            # This includes rate limiting. The next configured provider is tried.
            continue
    return ""


def _extractive_fallback(chunks: list[dict]) -> str:
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."
    evidence = []
    for index, chunk in enumerate(chunks[:3], start=1):
        text = " ".join(chunk.get("content", "").split())
        evidence.append(f"- {text[:420]} {_citation_label(chunk, index)}")
    return "Dưới đây là các đoạn thông tin liên quan trong tài liệu:\n\n" + "\n".join(evidence)


def generate_with_citation(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    history: Sequence[dict] | None = None,
    score_threshold: float = 0.35,
    use_reranking: bool = True,
    dense_weight: float = 0.5,
) -> dict:
    """Run retrieval then generate an answer without ever fabricating sources."""
    chunks = retrieve(
        query,
        top_k=top_k,
        score_threshold=score_threshold,
        use_reranking=use_reranking,
        dense_weight=dense_weight,
    )
    ordered_chunks = reorder_for_llm(chunks)
    context = format_context(ordered_chunks)
    recent_history = list(history or [])[-6:]
    history_text = "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in recent_history)
    user_message = f"LỊCH SỬ GẦN ĐÂY:\n{history_text or '(không có)'}\n\nCONTEXT:\n{context or '(trống)'}\n\nCÂU HỎI: {query}"
    answer = _generate_from_providers(user_message) if ordered_chunks else ""
    answer = answer or _extractive_fallback(ordered_chunks)
    return {
        "answer": answer,
        "sources": ordered_chunks,
        "retrieval_source": ordered_chunks[0].get("source", "none") if ordered_chunks else "none",
    }
