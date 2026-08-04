"""Load, chunk and persist the standardised knowledge base in ChromaDB."""

from __future__ import annotations

import re
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive (heading -> paragraph -> sentence)"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
COLLECTION_NAME = "ecommerce_support_docs"


def _parse_metadata_lines(block: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    return metadata


def _parse_front_matter(content: str) -> tuple[dict[str, str], str]:
    """Read a conventional front-matter block at the beginning of a document."""
    if not content.startswith("---\n"):
        return {}, content
    end = content.find("\n---", 4)
    if end < 0:
        return {}, content
    metadata = _parse_metadata_lines(content[4:end])
    return (metadata, content[end + 4 :].lstrip()) if metadata else ({}, content)


def _extract_embedded_metadata(content: str) -> tuple[dict[str, str], str]:
    """Remove YAML-like metadata blocks placed after an initial Markdown heading.

    Some legacy documents use a horizontal rule followed by a second `---`
    delimiter before their metadata.  That block must be treated as metadata,
    not as retrievable answer text.
    """
    delimiters = list(re.finditer(r"^---\s*$", content, flags=re.MULTILINE))
    for start, end in zip(delimiters, delimiters[1:]):
        metadata = _parse_metadata_lines(content[start.end() : end.start()])
        if metadata:
            cleaned_content = content[: start.start()] + content[end.end() :]
            return metadata, cleaned_content.lstrip()
    return {}, content


def _infer_customer_role(text: str) -> str:
    lowered = text.lower()
    has_buyer = "người mua" in lowered
    has_seller = "người bán" in lowered
    if has_buyer and has_seller:
        return "both"
    if has_seller:
        return "seller"
    return "buyer" if has_buyer else "both"


def _heading_for_offset(text: str, offset: int) -> str:
    headings = list(re.finditer(r"^#{1,6}\s+(.+)$", text, flags=re.MULTILINE))
    prior = [match.group(1).strip() for match in headings if match.start() <= offset]
    return prior[-1] if prior else "Tổng quan"


def load_documents() -> list[dict]:
    """Read all Markdown files and expose consistent, indexable provenance metadata."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        raw_content = md_file.read_text(encoding="utf-8")
        front_matter, content = _parse_front_matter(raw_content)
        embedded_metadata, content = _extract_embedded_metadata(content)
        document_metadata = {**embedded_metadata, **front_matter}
        doc_type = document_metadata.get("doc_type") or ("legal" if "legal" in md_file.parts else "news")
        title_match = re.search(r"^#\s+(.+)$", content, flags=re.MULTILINE)
        source_url_match = re.search(r"^\*\*Source:\*\*\s*(https?://\S+)", content, flags=re.MULTILINE)
        legacy_source_url_match = re.search(
            r"^(?:source_url|url):\s*(https?://\S+)",
            content,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        metadata = {
            "source": md_file.name,
            "relative_path": str(md_file.relative_to(STANDARDIZED_DIR)).replace("\\", "/"),
            "type": doc_type,
            "title": title_match.group(1).strip() if title_match else document_metadata.get("title", md_file.stem),
            "url": (
                document_metadata.get("url")
                or document_metadata.get("source_url")
                or (source_url_match.group(1) if source_url_match else "")
                or (legacy_source_url_match.group(1) if legacy_source_url_match else "")
            ),
            "category": document_metadata.get("category") or "ecommerce-policy",
            "version": document_metadata.get("version") or document_metadata.get("document_version") or "unknown",
            "date": document_metadata.get("date_crawled") or document_metadata.get("retrieved_at") or "",
            "customer_role": document_metadata.get("customer_role") or _infer_customer_role(content),
        }
        documents.append({"content": content, "metadata": metadata})
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents recursively, preserving the nearest Markdown section in metadata."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    for document in documents:
        content = document["content"]
        for index, chunk_text in enumerate(splitter.split_text(content)):
            if not chunk_text.strip():
                continue
            offset = content.find(chunk_text)
            metadata = {
                **document["metadata"],
                "chunk_index": index,
                "section": _heading_for_offset(content, max(offset, 0)),
            }
            chunks.append({"content": chunk_text, "metadata": metadata})
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed chunks locally with normalised sentence-transformer vectors."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode([chunk["content"] for chunk in chunks], normalize_embeddings=True, show_progress_bar=False)
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict], rebuild: bool = False) -> int:
    """Upsert chunks into the persistent Chroma collection and return indexed count."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    if not chunks:
        return 0
    ids = [f"{chunk['metadata']['relative_path']}::{chunk['metadata']['chunk_index']}" for chunk in chunks]
    collection.upsert(
        ids=ids,
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    return len(chunks)


def run_pipeline(rebuild: bool = True) -> int:
    """Build a clean dense index from the current standardised corpus."""
    chunks = embed_chunks(chunk_documents(load_documents()))
    indexed = index_to_vectorstore(chunks, rebuild=rebuild)
    print(f"[OK] Indexed {indexed} chunks into {CHROMA_DIR}")
    return indexed


if __name__ == "__main__":
    run_pipeline()
