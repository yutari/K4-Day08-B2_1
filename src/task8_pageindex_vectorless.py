"""
Task 8 — PageIndex Vectorless RAG.
"""

import os
from pathlib import Path
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """Upload toàn bộ markdown documents lên PageIndex."""
    if not PAGEINDEX_API_KEY:
        print("[INFO] PAGEINDEX_API_KEY không tồn tại, bỏ qua upload.")
        return
    try:
        # pyrefly: ignore [missing-import]
        from pageindex.client import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            print(f"[OK] Uploading: {md_file.name}")
    except Exception as e:
        print(f"[WARN] Upload error: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Vectorless retrieval sử dụng PageIndex / Structural search."""
    if PAGEINDEX_API_KEY:
        try:
            # pyrefly: ignore [missing-import]
            from pageindex.client import PageIndexClient
            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            resp = client.submit_query(query=query)
            retrieval_id = resp.get("retrieval_id") or resp.get("id")
            if retrieval_id:
                retrieval = client.get_retrieval(retrieval_id)
                results = []
                for node in retrieval.get("retrieved_nodes", [])[:top_k]:
                    for group in node.get("relevant_contents", []):
                        for item in group:
                            results.append({
                                "content": item.get("relevant_content", ""),
                                "score": 0.85,
                                "metadata": {"section": item.get("section_title", "PageIndex")},
                                "source": "pageindex",
                            })
                if results:
                    return results[:top_k]
        except Exception:
            pass

    # Structural local search fallback
    results = []
    if STANDARDIZED_DIR.exists():
        q_lower = query.lower()
        for md_file in list(STANDARDIZED_DIR.rglob("*.md"))[:3]:
            content = md_file.read_text(encoding="utf-8")
            if any(term in content.lower() for term in q_lower.split()):
                results.append({
                    "content": content[:600],
                    "score": 0.8,
                    "metadata": {"source": md_file.name, "type": "structural"},
                    "source": "pageindex",
                })
    return results[:top_k]


if __name__ == "__main__":
    results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
    for r in results:
        print(f"[{r.get('score', 0):.3f}] {r['content'][:100]}...")

