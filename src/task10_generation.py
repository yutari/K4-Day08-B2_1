"""
Task 10 — Generation Có Citation (Hỗ trợ OpenAI, Google Gemini 1.5 Flash, OpenRouter).
"""

import os
from dotenv import load_dotenv

load_dotenv()

try:
    from src.task9_retrieval_pipeline import retrieve
except ImportError:
    from .task9_retrieval_pipeline import retrieve


TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")

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
    """End-to-end RAG generation có citation với hỗ trợ Gemini 1.5 Flash & OpenAI."""
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
    answer = ""

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    # Provider 1: Direct Google Gemini API (via OpenAI-compatible endpoint)
    if gemini_key:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=gemini_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content
        except Exception:
            pass

    # Provider 2: OpenRouter API
    if not answer and openrouter_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
            model_name = os.getenv("LLM_MODEL", "google/gemini-1.5-flash")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content
        except Exception:
            pass

    # Provider 3: Direct OpenAI API
    if not answer and openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            model_name = os.getenv("LLM_MODEL", "gpt-4o")
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content
        except Exception:
            pass

    # Fallback template
    if not answer:
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
