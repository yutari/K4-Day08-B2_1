"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""


_model = None
_collection = None

def _init_components():
    global _model, _collection
    if _model is not None and _collection is not None:
        return
        
    try:
        from .task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL
    except ImportError:
        from pathlib import Path
        CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
        COLLECTION_NAME = "ecommerce_support_docs"
        EMBEDDING_MODEL = "BAAI/bge-m3"
        
    import chromadb
    from sentence_transformers import SentenceTransformer
    
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(name=COLLECTION_NAME)
        
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    _init_components()
    global _model, _collection
    
    query_vector = _model.encode(query).tolist()
    
    results = _collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    
    output = []
    if results["documents"] and len(results["documents"]) > 0:
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            score = max(0.0, 1.0 - dist)  # cosine distance → similarity
            output.append({"content": doc, "score": round(score, 4), "metadata": meta})
            
        output.sort(key=lambda x: x["score"], reverse=True)
        
    return output[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("quy định trả hàng hoàn tiền shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
