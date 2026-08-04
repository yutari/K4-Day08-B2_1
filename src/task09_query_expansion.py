"""Query rewriting for conversational RAG.

The module intentionally has no LLM dependency: a deterministic rewrite keeps the
retrieval path usable when the application is running without an API key.
"""

from __future__ import annotations

import re
from collections.abc import Sequence


FOLLOW_UP_MARKERS = (
    "nó", "cái đó", "việc đó", "trường hợp đó", "còn", "vậy",
    "bao lâu", "mất phí", "điều kiện", "bằng chứng", "họ", "này",
)


def _latest_user_question(messages: Sequence[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user" and message.get("content"):
            return str(message["content"]).strip()
    return ""


def needs_rewrite(query: str) -> bool:
    """Return whether *query* looks like a context-dependent follow-up."""
    try:
        from src.task10_generation import is_out_of_scope, is_small_talk
        if is_small_talk(query) or is_out_of_scope(query):
            return False
    except ImportError:
        pass

    normalized = re.sub(r"\s+", " ", query.lower()).strip()
    return any(marker in normalized for marker in FOLLOW_UP_MARKERS)


def rewrite_query(query: str, messages: Sequence[dict] | None = None) -> str:
    """Expand a follow-up question with the latest user topic when necessary.

    The original wording is preserved so exact terms still work well with BM25.
    """
    cleaned_query = re.sub(r"\s+", " ", query).strip()
    if not cleaned_query or not messages or not needs_rewrite(cleaned_query):
        return cleaned_query

    previous_question = _latest_user_question(messages)
    if not previous_question or previous_question.lower() == cleaned_query.lower():
        return cleaned_query

    return f"{previous_question} — Câu hỏi tiếp theo: {cleaned_query}"
