"""Vectorless fallback based on the Markdown document hierarchy.

The local implementation is always available and does not send documents to a
third party.  It provides the same safe fallback role when PageIndex credentials
are not configured.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.task6_lexical_search import tokenize

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _sections(content: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"^(#{1,6})\s+(.+)$", content, flags=re.MULTILINE))
    if not matches:
        return [("Tổng quan", content)]
    output: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        output.append((match.group(2).strip(), content[match.start():end].strip()))
    return output


_STOPWORDS = {
    "là", "ai", "và", "hoặc", "của", "trong", "cho", "các", "những", "có", "bị", "được", "khi", "này", "với", "gì", "thế", "nào"
}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Search document sections structurally using exact lexical coverage.

    This is deliberately independent of Chroma/BM25, so it remains a useful
    fallback for an empty or low-confidence dense index.
    """
    terms = set(tokenize(query))
    content_terms = terms - _STOPWORDS
    if not terms or not STANDARDIZED_DIR.exists() or top_k < 1:
        return []
    candidates: list[dict] = []
    for path in STANDARDIZED_DIR.rglob("*.md"):
        content = path.read_text(encoding="utf-8")
        for section, section_content in _sections(content):
            haystack = set(tokenize(f"{section} {section_content}"))
            overlap = len(terms & haystack)
            if content_terms and not (content_terms & haystack):
                continue
            if not overlap:
                continue
            score = overlap / len(terms)
            candidates.append(
                {
                    "content": section_content[:1800],
                    "score": round(score, 4),
                    "score_type": "structural_keyword_coverage",
                    "metadata": {
                        "source": path.name,
                        "relative_path": str(path.relative_to(STANDARDIZED_DIR)).replace("\\", "/"),
                        "type": "legal" if "legal" in path.parts else "news",
                        "section": section,
                    },
                    "source": "pageindex",
                }
            )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:top_k]


def upload_documents() -> None:
    """Document the deliberate local-first fallback.

    PageIndex cloud ingestion is optional and provider-specific; this project
    remains operational without exposing the private policy corpus externally.
    """
    print("[INFO] Local structural PageIndex fallback is ready; no upload is required.")
