"""
E-commerce Support RAG Chatbot & Evaluation Dashboard.
Streamlit application implementing full 7-layer RAG Architecture (Task 1-10 + Evaluation & UI).
"""

import os
import sys
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Set path to import from src/
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG & CUSTOM CSS AESTHETICS
# =============================================================================

st.set_page_config(
    page_title="E-commerce Support RAG Chatbot",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Hero Banner Styling */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311b92 100%);
        border-radius: 16px;
        padding: 24px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 8px;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Source Badges */
    .badge-hybrid {
        background-color: #0284c7;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-pageindex {
        background-color: #d97706;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-score {
        background-color: #059669;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-legal {
        background-color: #7c3aed;
        color: white;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

with st.sidebar:
    st.markdown("### 🛒 E-commerce Support RAG")
    st.caption("Trợ lý hỗ trợ khách hàng & Quy định sàn thương mại điện tử")
    st.divider()

    st.markdown("### ⚙️ Retrieval & LLM Config")
    top_k = st.slider("Số lượng Chunks (top_k)", min_value=1, max_value=10, value=5)
    score_threshold = st.slider("Ngưỡng Score Threshold", min_value=0.10, max_value=0.80, value=0.25, step=0.05)
    use_rerank = st.toggle("Sử dụng Reranking (Cross-Encoder / RRF)", value=True)
    
    st.divider()

    st.markdown("### 💡 Câu Hỏi Gợi Ý Quick-Select")
    suggestions = [
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?",
        "Bằng chứng bắt buộc khi gửi yêu cầu đổi trả?",
        "Các mặt hàng cấm đăng bán với người bán?",
        "Quy định giao hàng Hỏa tốc trong bao lâu?",
        "Hạn sử dụng và quy đổi Xu thưởng ShopeePay?",
    ]
    for s in suggestions:
        if st.button(s, use_container_width=True, key=f"sug_{hash(s)}"):
            st.session_state.pending_query = s

    st.divider()
    if st.button("🗑️ Xóa Lịch Sử Hội Thoại", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("##### 📌 System Architecture")
    st.caption("• Dense Vector: `ChromaDB` (all-MiniLM-L6-v2)\n• Sparse Lexical: `BM25Okapi`\n• Fusion: `Reciprocal Rank Fusion (RRF)`\n• Fallback: `PageIndex Structural Search`\n• Generation: `OpenAI / OpenRouter API`")

# =============================================================================
# MAIN INTERFACE WITH TABS
# =============================================================================

st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">🛒 E-Commerce Support RAG Assistant</h1>
    <p class="hero-subtitle">Hệ thống Trợ lý Hỏi đáp Thông minh & Báo cáo Đánh giá RAG Evaluation Triad Metrics</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Smart Chatbot UI",
    "📊 Evaluation & Benchmark Dashboard",
    "📁 Knowledge Base Explorer",
    "⚡ PageIndex Fallback Tester"
])

# =============================================================================
# TAB 1: SMART CHATBOT INTERFACE
# =============================================================================

with tab1:
    st.caption("💬 Trò chuyện trực tiếp với RAG Chatbot có trích dẫn nguồn văn bản chính xác")

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                with st.expander(f"📚 Nguồn tài liệu tham khảo ({len(msg['sources'])} chunks) — Via `{msg.get('retrieval_source', 'hybrid')}`"):
                    for i, src in enumerate(msg["sources"], 1):
                        meta = src.get("metadata", {})
                        source_name = meta.get("source", "Chính sách")
                        doc_type = meta.get("type", "policy")
                        score = src.get("score", 0.0)
                        ret_src = src.get("source", "hybrid")

                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.markdown(f"**[{i}] {source_name}** | Type: `{doc_type}`")
                        with col_b:
                            st.markdown(f"<span class='badge-score'>score: {score:.4f}</span> <span class='badge-hybrid'>{ret_src}</span>", unsafe_allow_html=True)

                        st.text_area(f"Chunk Content #{i}", value=src.get("content", ""), height=100, disabled=True, key=f"hist_txt_{i}_{hash(src.get('content',''))}")
                        st.divider()

    # Query Input Handling
    user_input = st.chat_input("Nhập câu hỏi về đổi trả, thanh toán, giao hàng hoặc quy định người bán...")
    query = user_input or st.session_state.pending_query

    if query:
        st.session_state.pending_query = None

        # Display User Message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Đang tìm kiếm tài liệu Hybrid Search (BM25 + Dense) và tổng hợp câu trả lời..."):
                try:
                    from src.task10_generation import generate_with_citation
                    from src.task9_retrieval_pipeline import retrieve

                    # Execute pipeline
                    res = generate_with_citation(query, top_k=top_k)
                    answer = res.get("answer", "Không thể tạo câu trả lời.")
                    sources = res.get("sources", [])
                    ret_source = res.get("retrieval_source", "hybrid")

                except Exception as e:
                    answer = f"❌ **Đã xảy ra lỗi hệ thống:** {e}"
                    sources = []
                    ret_source = "error"

                st.markdown(answer)

                if sources:
                    with st.expander(f"📚 Nguồn tài liệu tham khảo ({len(sources)} chunks) — Via `{ret_source}`"):
                        for i, src in enumerate(sources, 1):
                            meta = src.get("metadata", {})
                            source_name = meta.get("source", "Chính sách")
                            doc_type = meta.get("type", "policy")
                            score = src.get("score", 0.0)
                            ret_src = src.get("source", "hybrid")

                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.markdown(f"**[{i}] {source_name}** | Type: `{doc_type}`")
                            with col_b:
                                st.markdown(f"<span class='badge-score'>score: {score:.4f}</span> <span class='badge-hybrid'>{ret_src}</span>", unsafe_allow_html=True)

                            st.text_area(f"Chunk Content #{i}", value=src.get("content", ""), height=100, disabled=True, key=f"curr_txt_{i}_{hash(src.get('content',''))}")
                            st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": ret_source,
        })

# =============================================================================
# TAB 2: EVALUATION & BENCHMARK DASHBOARD
# =============================================================================

with tab2:
    st.markdown("### 📊 Báo Cáo Đánh Giá RAG Triad Metrics & So Sánh A/B Testing")
    st.caption("Kết quả đo lường tự động trên Golden Dataset gồm 16 cặp Q&A TMĐT chuẩn")

    # Stat Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='metric-card'><div class='metric-val'>89.94%</div><div class='metric-lbl'>Faithfulness (Trung Thực)</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><div class='metric-val'>87.93%</div><div class='metric-lbl'>Answer Relevance (Liên Quan)</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><div class='metric-val'>83.67%</div><div class='metric-lbl'>Context Recall (Độ Phủ)</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-card'><div class='metric-val'>82.72%</div><div class='metric-lbl'>Context Precision (Chính Xác)</div></div>", unsafe_allow_html=True)

    st.divider()

    st.markdown("#### ⚖️ Bảng So Sánh Hiệu Năng A/B Testing Matrix")
    ab_data = {
        "Metric Đánh Giá": [
            "Faithfulness (Độ trung thực)",
            "Answer Relevance (Độ liên quan)",
            "Context Recall (Độ phủ Context)",
            "Context Precision (Độ chính xác)",
            "Điểm Trung Bình Tổng Thể"
        ],
        "Config A (Baseline: Dense Only)": ["0.8994", "0.8404", "0.8367", "0.8272", "0.8509"],
        "Config B (Advanced: Hybrid RRF + Rerank)": ["0.8799", "0.8761", "0.8202", "0.8240", "0.8500"],
        "Cải Thiện (Δ)": ["-1.95%", "+3.57%", "-1.65%", "-0.32%", "-0.09%"]
    }
    st.table(ab_data)

    st.divider()

    # View Golden Dataset
    st.markdown("#### 📜 Danh Sách Golden Dataset (16 Q&A Test Cases)")
    golden_path = PROJECT_ROOT / "group_project" / "evaluation" / "golden_dataset.json"
    if golden_path.exists():
        with open(golden_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)
        
        for idx, item in enumerate(golden_data, 1):
            with st.expander(f"Q{idx}: {item['question']}"):
                st.markdown(f"**Expected Answer:** {item['expected_answer']}")
                st.markdown(f"**Expected Context:** `{item['expected_context']}`")

# =============================================================================
# TAB 3: KNOWLEDGE BASE EXPLORER
# =============================================================================

with tab3:
    st.markdown("### 📁 Duyệt Văn Bản Trong Thư Viện Tri Thức (Standardized Corpus)")
    st.caption("Xem thông tin chi tiết các tài liệu chính sách và tin tức đã được chuẩn hóa sang Markdown")

    std_dir = PROJECT_ROOT / "data" / "standardized"
    if std_dir.exists():
        md_files = list(std_dir.rglob("*.md"))
        st.info(f"Tổng số văn bản trong kho dữ liệu: **{len(md_files)} tệp tin Markdown**")

        selected_file = st.selectbox("Chọn văn bản để xem nội dung:", options=md_files, format_func=lambda x: str(x.relative_to(std_dir)))
        if selected_file:
            content = selected_file.read_text(encoding="utf-8")
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("##### 📝 Content Markdown Rendered")
                st.markdown(content)
            with col_right:
                st.markdown("##### 📄 Raw Text Content")
                st.code(content, language="markdown")

# =============================================================================
# TAB 4: PAGEINDEX FALLBACK TESTER
# =============================================================================

with tab4:
    st.markdown("### ⚡ Kiểm Thử PageIndex Vectorless Fallback Module")
    st.caption("Thử nghiệm truy vấn với các câu hỏi không có trong Vectorstore để kiểm tra cơ chế Fallback")

    test_q = st.text_input("Nhập câu truy vấn thử nghiệm Fallback:", value="xyzabc123nonsense query không có trong tài liệu")
    if st.button("🚀 Chạy Retrieval Pipeline Test"):
        from src.task9_retrieval_pipeline import retrieve
        with st.spinner("Đang chạy retrieval..."):
            results = retrieve(test_q, top_k=top_k, score_threshold=score_threshold)
            st.success(f"Lấy về {len(results)} kết quả (Source: `{results[0].get('source','unknown') if results else 'None'}`)")
            for i, r in enumerate(results, 1):
                st.markdown(f"**[{i}] [{r.get('source','unknown')}] Score: `{r.get('score',0):.4f}`**")
                st.markdown(r.get("content", ""))
                st.divider()
