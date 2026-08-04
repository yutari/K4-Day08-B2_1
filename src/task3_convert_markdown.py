"""Standardise landing documents to Markdown while retaining provenance metadata."""

from __future__ import annotations

import json
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _metadata_block(metadata: dict[str, str]) -> str:
    lines = ["---"]
    for key, value in metadata.items():
        if value:
            lines.append(f"{key}: {str(value).replace(chr(10), ' ')}")
    return "\n".join(lines) + "\n---\n\n"


def _convert_with_markitdown(path: Path) -> str:
    """Convert a binary office document using the required MarkItDown package."""
    from markitdown import MarkItDown

    result = MarkItDown().convert(str(path))
    return result.text_content.strip()


def convert_legal_docs() -> None:
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not legal_dir.exists():
        return

    for path in legal_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            continue
        try:
            content = _convert_with_markitdown(path)
        except Exception as exc:
            raise RuntimeError(f"Không thể chuyển đổi {path.name} bằng MarkItDown: {exc}") from exc

        metadata = {
            "title": path.stem.replace("_", " ").replace("-", " ").title(),
            "source_file": path.name,
            "doc_type": "legal",
            "category": "ecommerce-policy",
            "version": "unknown",
        }
        (output_dir / f"{path.stem}.md").write_text(
            _metadata_block(metadata) + f"# {metadata['title']}\n\n{content}\n",
            encoding="utf-8",
        )


def convert_news_articles() -> None:
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not news_dir.exists():
        return

    for path in news_dir.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        output_path = output_dir / f"{path.stem}.md"
        if path.suffix.lower() == ".json":
            article = json.loads(path.read_text(encoding="utf-8"))
            title = article.get("title") or path.stem
            metadata = {
                "title": title,
                "url": article.get("url", ""),
                "date_crawled": article.get("date_crawled", ""),
                "source_file": path.name,
                "doc_type": "news",
                "category": "customer-support",
            }
            content = article.get("content_markdown", "").strip()
            if not content.startswith("#"):
                content = f"# {title}\n\n{content}"
        elif path.suffix.lower() in {".md", ".txt", ".html"}:
            metadata = {"source_file": path.name, "doc_type": "news", "category": "customer-support"}
            content = path.read_text(encoding="utf-8").strip()
            if not content.startswith("#"):
                content = f"# {path.stem.replace('_', ' ').title()}\n\n{content}"
        else:
            continue
        output_path.write_text(_metadata_block(metadata) + content + "\n", encoding="utf-8")


def convert_all() -> None:
    convert_legal_docs()
    convert_news_articles()


if __name__ == "__main__":
    convert_all()
