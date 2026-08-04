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
_METADATA_LINE_RE = re.compile(
    r"^\s*(?:doc_id|title|source_url|url|retrieved_at|date_crawled|"
    r"document_version|version|customer_role|category|language|doc_type|"
    r"relative_path)\s*:\s*.*$",
    flags=re.IGNORECASE,
)
_MARKDOWN_SOURCE_LINE_RE = re.compile(r"^\s*\*\*(?:source|nguồn)\*\*\s*:\s*.*$", flags=re.IGNORECASE)
_SMALL_TALK_QUERIES = frozenset(
    {
        "xin chào",
        "xin chào bạn",
        "chào",
        "chào bạn",
        "hello",
        "hi",
        "hey",
        "alo",
        "alô",
        "cảm ơn",
        "cảm ơn bạn",
        "thank you",
        "thanks",
        "cảm ơn nhiều",
        "tạm biệt",
        "bye",
        "goodbye",
        "ok",
        "oke",
        "dạ",
        "ừ",
        "uh",
        "hihi",
        "haha",
        "chán",
        "chán quá",
        "chán thế",
        "chán vãi",
        "chán ghê",
        "nản",
        "nản quá",
        "buồn",
        "buồn quá",
        "mệt",
        "mệt quá",
        "mệt mỏi",
        "haiz",
        "haizz",
        "chat chán quá",
        "nói chuyện chán quá",
        "tâm sự",
        "tâm sự tí",
        "vui quá",
        "tốt quá",
        "tuyệt",
        "bạn là ai",
        "bạn tên gì",
        "bạn là gì",
        "bạn làm được gì",
        "bot là ai",
        "who are you",
    }
)
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
    # A model occasionally echoes document front matter along with a valid
    # citation.  That is provenance, not a user-facing answer, so force the
    # already-sanitised extractive fallback instead.
    if any(_METADATA_LINE_RE.match(line) or _MARKDOWN_SOURCE_LINE_RE.match(line) for line in answer.splitlines()):
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


_DOMAIN_ROLES = {"người mua", "người bán", "shopee", "bên vận chuyển", "đơn vị vận chuyển", "chính sách"}


def is_out_of_scope(query: str) -> bool:
    """Return whether a query is clearly outside the e-commerce policy domain."""
    norm = query.lower().strip()
    if re.search(r"\b(là ai|ai là|ai thế)\b", norm):
        if not any(role in norm for role in _DOMAIN_ROLES):
            return True
    return False


def is_small_talk(query: str) -> bool:
    """Return whether a query is a standalone greeting, courtesy message, or casual chat.

    Such messages are not policy questions, so running retrieval for them can
    surface an arbitrary, unrelated document.
    """
    normalised = " ".join(re.sub(r"[^\w\s]", " ", query.casefold()).split())
    if normalised in _SMALL_TALK_QUERIES:
        return True
    casual_patterns = (
        r"^(chán|nản|buồn|mệt|haiz|haha|hihi)(\s+.*)?$",
        r"^.*(chán quá|nói chuyện chán|chat chán).*$",
        r"^(bạn|bot) (là ai|tên gì|làm được gì|có thể làm gì)",
    )
    return any(re.search(pat, normalised) for pat in casual_patterns)


def _clean_evidence_text(content: str) -> str:
    """Remove Markdown structure and document metadata from fallback evidence.

    Older indexed documents may contain a YAML-like block after a heading.
    It is useful provenance for indexing but is never an answer for a user.
    Line-based filtering also protects users before they rebuild an old index.
    """
    prose_lines: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line == "---" or line.startswith("#"):
            continue
        if _METADATA_LINE_RE.match(line) or _MARKDOWN_SOURCE_LINE_RE.match(line):
            continue
        line = _BULLET_RE.sub("", line)
        # Preserve content while removing only common inline Markdown wrappers.
        line = re.sub(r"`([^`]*)`", r"\1", line)
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        if line:
            prose_lines.append(line)
    return " ".join(prose_lines)


def _extractive_fallback(chunks: list[dict]) -> str:
    if not chunks:
        return UNVERIFIABLE_ANSWER
    evidence = []
    for index, chunk in enumerate(chunks[:3], start=1):
        text = _clean_evidence_text(str(chunk.get("content", "")))
        if text:
            evidence.append(f"- {text[:420]} {_citation_label(chunk, index)}")
    if not evidence:
        return UNVERIFIABLE_ANSWER
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
    if is_out_of_scope(query):
        return {
            "answer": UNVERIFIABLE_ANSWER,
            "sources": [],
            "retrieval_source": "none",
            "citation_validation": "no_evidence",
        }

    if is_small_talk(query):
        norm = query.lower()
        if any(w in norm for w in ("chán", "nản", "buồn", "mệt", "haiz")):
            answer = (
                "Mình rất tiếc nếu trải nghiệm chưa như bạn kỳ vọng! "
                "Mình là Trợ lý AI chuyên hỗ trợ giải đáp các quy định và chính sách Thương mại Điện tử (đổi trả, hoàn tiền, giao hàng, thanh toán...). "
                "Nếu bạn có câu hỏi nào liên quan đến các quy định này, hãy cho mình biết nhé!"
            )
        elif any(w in norm for w in ("bạn là ai", "bạn làm được gì", "bạn tên gì", "bot là ai")):
            answer = (
                "Mình là Trợ lý AI hỗ trợ giải đáp các quy định và chính sách Thương mại Điện tử "
                "(như Shopee: đổi trả hàng, hoàn tiền, thời gian giao hàng, bằng chứng khiếu nại, quy định người bán...). "
                "Bạn cần mình hỗ trợ thông tin gì không?"
            )
        else:
            answer = (
                "Xin chào! Mình có thể hỗ trợ bạn về thanh toán, trả hàng/hoàn tiền, "
                "giao hàng và quy định người bán. Bạn muốn hỏi nội dung nào?"
            )
        return {
            "answer": answer,
            "sources": [],
            "retrieval_source": "none",
            "citation_validation": "not_required",
        }

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
