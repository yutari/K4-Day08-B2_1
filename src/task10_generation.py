"""Grounded answer generation with source-aware citations and provider failover."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

from dotenv import load_dotenv

from src.task9_retrieval_pipeline import DEFAULT_TOP_K, retrieve

load_dotenv()

TEMPERATURE = 0.2
TOP_P = 0.9
UNVERIFIABLE_ANSWER = "T\u00f4i kh\u00f4ng th\u1ec3 x\u00e1c minh th\u00f4ng tin n\u00e0y t\u1eeb ngu\u1ed3n hi\u1ec7n c\u00f3."

# The generator is told to use the Vietnamese label below.  The parser also
# accepts its unaccented and English equivalents so a provider formatting
# variation can be normalised to the exact label supplied in the context.
_CITATION_RE = re.compile(
    r"\[\s*(?:ngu\u1ed3n|nguon|source)\s*:\s*"
    r"(?P<source>[^,\]\n]+?)\s*,\s*"
    r"(?:m\u1ee5c|muc|section)\s*:\s*(?P<section>[^\]\n]+?)\s*\]",
    flags=re.IGNORECASE,
)
_CITATION_LIKE_RE = re.compile(
    r"\[\s*(?:ngu\u1ed3n|nguon|source)\s*:[^\]\n]*\]",
    flags=re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
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
    metadata = chunk.get("metadata") or {}
    source = str(metadata.get("source") or f"Source_{index}.md").strip()
    section = str(metadata.get("section") or "Tổng quan").strip()
    return f"[Nguồn: {source}, Mục: {section}]"


def _normalise_citation_part(value: str) -> str:
    """Compare citation parts case-insensitively and without spacing noise."""
    return " ".join(value.split()).casefold()


def _citation_catalog(chunks: Sequence[dict]) -> dict[tuple[str, str], str]:
    """Map every retrieved source/section pair to its canonical display label."""
    catalog: dict[tuple[str, str], str] = {}
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        source = str(metadata.get("source") or f"Source_{index}.md")
        section = str(metadata.get("section") or "Tổng quan")
        key = (_normalise_citation_part(source), _normalise_citation_part(section))
        catalog.setdefault(key, _citation_label(chunk, index))
    return catalog


def _contains_uncited_substantive_block(answer: str) -> bool:
    """Require an inline citation for each answer paragraph or bullet block.

    Markdown headings and citation-only lines are deliberately ignored.  When
    this check is too strict for a provider's answer format we choose the
    extractive fallback instead of attaching a source to an unsupported claim.
    """
    answer_units: list[str] = []
    for raw_block in re.split(r"\n\s*\n", answer):
        prose_lines: list[str] = []
        for raw_line in raw_block.splitlines():
            line = raw_line.strip()
            if line.startswith("#"):
                continue
            if _BULLET_RE.match(line):
                if prose_lines:
                    answer_units.append("\n".join(prose_lines))
                    prose_lines = []
                answer_units.append(raw_line)
            else:
                prose_lines.append(raw_line)
        if prose_lines:
            answer_units.append("\n".join(prose_lines))

    for unit in answer_units:
        without_citations = _CITATION_RE.sub("", unit)
        # Ignore bullets, markdown decoration, and short connective text.  A
        # meaningful statement has at least twelve word characters remaining.
        meaningful_characters = re.findall(r"\w", without_citations, flags=re.UNICODE)
        if len(meaningful_characters) >= 12 and not _CITATION_RE.search(unit):
            return True
    return False


def validate_and_normalize_citations(answer: str, chunks: Sequence[dict]) -> str | None:
    """Return a citation-safe answer, or ``None`` when it cannot be trusted.

    A model may invent a filename, section, or citation syntax even when the
    prompt lists allowed labels.  This post-check accepts only citations whose
    ``source`` *and* ``section`` match a retrieved chunk, converts formatting
    variations to the canonical label, and rejects answers with missing or
    malformed citation blocks.  Callers should use an extractive fallback when
    ``None`` is returned.
    """
    if not answer or not answer.strip() or not chunks:
        return None

    catalog = _citation_catalog(chunks)
    matches = list(_CITATION_RE.finditer(answer))
    if not catalog or not matches:
        return None
    # A citation by itself is not an answer.  Require at least a small amount
    # of non-citation text before allowing provider output through.
    if len(re.findall(r"\w", _CITATION_RE.sub("", answer), flags=re.UNICODE)) < 3:
        return None

    # A partial label such as ``[Nguồn: made-up.pdf]`` must not be allowed to
    # sit beside otherwise valid citations.
    valid_spans = {(match.start(), match.end()) for match in matches}
    for citation_like in _CITATION_LIKE_RE.finditer(answer):
        if (citation_like.start(), citation_like.end()) not in valid_spans:
            return None

    def canonicalise(match: re.Match[str]) -> str:
        key = (
            _normalise_citation_part(match.group("source")),
            _normalise_citation_part(match.group("section")),
        )
        return catalog.get(key, "")

    # Never silently delete an invalid label: rejecting the entire generated
    # answer prevents a hallucinated claim from surviving without a citation.
    if any(not canonicalise(match) for match in matches):
        return None
    normalised = _CITATION_RE.sub(canonicalise, answer).strip()
    if _contains_uncited_substantive_block(normalised):
        return None
    return normalised


def format_context(chunks: list[dict]) -> str:
    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
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
        return UNVERIFIABLE_ANSWER
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
    generated_answer = _generate_from_providers(user_message) if ordered_chunks else ""
    validated_answer = validate_and_normalize_citations(generated_answer, ordered_chunks)
    if validated_answer:
        answer = validated_answer
        citation_validation = "validated"
    elif ordered_chunks:
        # Do not repair a model's invented source by guessing.  The fallback is
        # made directly from retrieved evidence and carries canonical labels.
        answer = _extractive_fallback(ordered_chunks)
        citation_validation = "extractive_fallback"
    else:
        answer = _extractive_fallback([])
        citation_validation = "no_evidence"
    return {
        "answer": answer,
        "sources": ordered_chunks,
        "retrieval_source": ordered_chunks[0].get("source", "none") if ordered_chunks else "none",
        "citation_validation": citation_validation,
    }
