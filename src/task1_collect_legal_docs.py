"""Validate the raw legal-policy documents used by the RAG knowledge base.

The repository intentionally keeps the downloaded source PDFs in ``data/landing``.
This task never fabricates policy PDFs: doing so would make a production answer
look cited while being based on invented source material.
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
VALID_EXTENSIONS = {".pdf", ".doc", ".docx"}


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def validate_legal_documents(minimum: int = 3) -> list[Path]:
    """Return real source documents or explain exactly what is missing."""
    setup_directory()
    documents = sorted(path for path in DATA_DIR.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS)
    invalid = [path.name for path in documents if path.stat().st_size <= 1024]
    if len(documents) < minimum:
        raise RuntimeError(f"Cần ít nhất {minimum} PDF/DOCX trong {DATA_DIR}; hiện có {len(documents)}.")
    if invalid:
        raise RuntimeError(f"Tài liệu nguồn quá nhỏ hoặc hỏng: {', '.join(invalid)}")
    return documents


if __name__ == "__main__":
    for document in validate_legal_documents():
        print(f"[OK] {document.name}")
