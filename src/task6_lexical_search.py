"""BM25 lexical retrieval using a Vietnamese-friendly tokenizer."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.task4_chunking_indexing import chunk_documents, load_documents

_CORPUS: list[dict] | None = None
_BM25_INDEX: BM25Okapi | None = None


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def invalidate_bm25_cache() -> None:
    global _CORPUS, _BM25_INDEX
    _CORPUS, _BM25_INDEX = None, None


def _get_corpus_and_index() -> tuple[list[dict], BM25Okapi | None]:
    global _CORPUS, _BM25_INDEX
    if _CORPUS is None or _BM25_INDEX is None:
        _CORPUS = chunk_documents(load_documents())
        _BM25_INDEX = BM25Okapi([tokenize(item["content"]) for item in _CORPUS]) if _CORPUS else None
    return _CORPUS, _BM25_INDEX


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    return BM25Okapi([tokenize(item["content"]) for item in corpus])


def lexical_search(query: str, top_k: int = 20) -> list[dict]:
    """Return only positively matching BM25 results, sorted high to low."""
    if not query.strip() or top_k < 1:
        return []
    corpus, index = _get_corpus_and_index()
    if not corpus or index is None:
        return []
    scores = index.get_scores(tokenize(query))
    positive_indices = [int(i) for i in np.argsort(scores)[::-1] if scores[i] > 0]
    return [
        {
            "content": corpus[i]["content"],
            "score": round(float(scores[i]), 4),
            "metadata": corpus[i].get("metadata", {}),
        }
        for i in positive_indices[:top_k]
    ]
