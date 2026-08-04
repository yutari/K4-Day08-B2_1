"""
Task 10 — Generation Có Citation.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-2.0-flash-exp:free")

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về chính sách thương mại điện tử và hỗ trợ
khách hàng (thanh toán, đổi trả, giao hàng, quyền riêng tư, quy định người bán).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Nguồn: tên_file.md]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Sắp xếp chunks để tránh 'lost in the middle' effect."""
    if len(chunks) <= 2:
        return chunks
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Format chunks thành context string cho prompt."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", f"Source_{i}.md")
        doc_type = meta.get("type", "policy")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk.get('content', '')}\n"
        )
    return "\n---\n".join(context_parts)


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """End-to-end RAG generation có citation."""
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
    answer = ""

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=LLM_MODEL if os.getenv("OPENROUTER_API_KEY") else "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            answer = f"Dựa trên các tài liệu trích dẫn:\n\n" + "\n\n".join([f"- {c['content'][:150]}... [Nguồn: {c.get('metadata',{}).get('source','Chính sách')}]" for c in chunks[:3]])
    else:
        answer = f"Dựa trên các văn bản quy định:\n\n" + "\n\n".join([f"- {c['content'][:150]}... [Nguồn: {c.get('metadata',{}).get('source','Chính sách')}]" for c in chunks[:3]])

    retrieval_src = chunks[0].get("source", "hybrid") if chunks else "none"

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_src
    }


if __name__ == "__main__":
    test_queries = [
        "Shopee ho tro nhung phuong thuc thanh toan nao?",
        "Lam sao de yeu cau doi tra hay hoan tien?",
        "Can chuan bi bang chung gi khi yeu cau hoan tien?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        ans = result['answer'].encode('ascii', errors='ignore').decode('ascii')
        print(f"\nA: {ans[:200]}...")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")


