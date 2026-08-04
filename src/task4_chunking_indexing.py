"""
Task 4 — Chunking & Indexing vào Vector Store.
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "ecommerce_support_docs"


def load_documents() -> list[dict]:
    """Đọc toàn bộ markdown files từ data/standardized/."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chunk documents theo strategy RecursiveCharacterTextSplitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            if chunk_text.strip():
                chunks.append({
                    "content": chunk_text,
                    "metadata": {**doc["metadata"], "chunk_index": i}
                })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed toàn bộ chunks bằng SentenceTransformer."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks vào vector store ChromaDB."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{c['metadata']['source']}_chunk_{c['metadata'].get('chunk_index', idx)}" for idx, c in enumerate(chunks)]
    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("[OK] Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()

