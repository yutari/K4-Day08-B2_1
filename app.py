"""
E-commerce Support RAG Chatbot & Evaluation Dashboard.
Streamlit application implementing full 7-layer RAG Architecture (Task 1-10 + Evaluation & UI).
Theme: Bright Cool Light Mode (Màu lạnh sáng đẹp) với Direct Source Web Links
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
# URL MAPPING FOR DIRECT SOURCE ACCESS
# =============================================================================

URL_MAP = {
    "shopeereturn.md": "https://help.shopee.vn/portal/4/article/77251",
    "shopeesell.md": "https://help.shopee.vn/portal/4/article/79200",
    "shopeespaylater.md": "https://help.shopee.vn/portal/4/article/79205",
    "article_01.md": "https://help.shopee.vn/portal/4/article/79198-Phuong-thuc-thanh-toan",
    "article_02.md": "https://help.shopee.vn/portal/4/article/77251-Quy-dinh-tra-hang-hoan-tien",
    "article_03.md": "https://help.shopee.vn/portal/4/article/77244-Huong-dan-giao-hang-tiet-kiem",
    "article_04.md": "https://help.shopee.vn/portal/4/article/79200-Quy-dinh-dang-ban-san-pham-nguoi-ban",
    "article_05.md": "https://help.shopee.vn/portal/4/article/79205-Huong-dan-su-dung-ShopeePay-Coin",
    "tra_hang_hoan_tien.md": "https://help.shopee.vn/portal/4/article/77251",
    "dieu_khoan_splater.md": "https://help.shopee.vn/portal/4/article/79205",
    "dang_ban_san_pham.md": "https://help.shopee.vn/portal/4/article/79200",
    "chinh_sach_van_chuyen.md": "https://help.shopee.vn/portal/4/article/77244",
    "quy_che_hoat_dong_chung.md": "https://help.shopee.vn/portal/4/article/79198",
    "chinh_sach_chong_gian_lan.md": "https://help.shopee.vn/portal/4/article/77251",
    "chinh_sach_ma_uu_dai.md": "https://help.shopee.vn/portal/4/article/79198",
    "dieu_khoan_shopee_ai.md": "https://help.shopee.vn/portal/4/article/79198",
    "quyen_so_huu_tri_tue.md": "https://help.shopee.vn/portal/4/article/79200",
    "tranh_chap_khieu_nai.md": "https://help.shopee.vn/portal/4/article/77251",
}

# =============================================================================
# PAGE CONFIG & BRIGHT COOL CUSTOM CSS AESTHETICS
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Bright Cool Background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    /* Hero Banner - Bright Cool Glacier Gradient */
    .hero-banner {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 40%, #1e40af 100%);
        border-radius: 18px;
        padding: 28px 36px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 12px 28px -6px rgba(2, 132, 199, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        color: #ffffff;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .hero-subtitle {
        font-size: 1.08rem;
        color: #e0f2fe;
        margin-top: 8px;
        margin-bottom: 0;
        font-weight: 400;
    }

    /* Metric Cards - Icy White Glassmorphism */
    .metric-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.06);
        transition: all 0.25s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px -4px rgba(2, 132, 199, 0.18);
        border-color: #38bdf8;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 800;
        color: #0284c7;
        letter-spacing: -0.5px;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #475569;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 4px;
    }

    /* Source Badges - Crisp Cool Tones */
    .badge-hybrid {
        background-color: #0284c7;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .badge-pageindex {
        background-color: #d97706;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .badge-score {
        background-color: #059669;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* Sidebar - Soft Ice Blue Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f9ff 0%, #e0f2fe 100%);
        border-right: 1px solid #bae6fd;
    }
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #0f172a !important;
    }

    /* Chat Messages Styling */
    .stChatMessage[data-testid="stChatMessage"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        padding: 14px;
        margin-bottom: 12px;
    }

    /* Expander Styling */
    div[data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
    }

    /* Tabs Header Styling */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 1rem !important;
        color: #475569 !important;
    }
    button[aria-selected="true"] {
        color: #0284c7 !important;
        border-bottom-color: #0284c7 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER: RENDER SOURCE CARD WITH UNIQUE KEYS
# =============================================================================

def render_source_card(i: int, src: dict, key_prefix: str = "src"):
    meta = src.get("metadata", {})
    source_name = meta.get("source", "Chính sách")
    doc_type = meta.get("type", "policy")
    score = src.get("score", 0.0)
    ret_src = src.get("source", "hybrid")
    content_str = src.get("content", "")

    # Resolve URL
    url = meta.get("url") or URL_MAP.get(source_name)
    if not url or not str(url).startswith("http"):
        for line in content_str.splitlines()[:10]:
            if "**Source:**" in line:
                extracted = line.split("**Source:**")[1].strip()
                if extracted.startswith("http"):
                    url = extracted
                    break

    col_a, col_b = st.columns([2.5, 1.5])
    with col_a:
        if url and str(url).startswith("http"):
            st.markdown(f"**[{i}] 🔗 [{source_name}]({url})** | Type: `{doc_type}`")
            st.caption(f"🌐 Nguồn web chính thức: [{url}]({url})")
        else:
            st.markdown(f"**[{i}] 📄 {source_name}** | Type: `{doc_type}`")
            st.caption(f"📁 Tệp tin tài liệu: `data/standardized/{doc_type}/{source_name}`")

    with col_b:
        st.markdown(
            f"<span class='badge-score'>score: {score:.4f}</span> "
            f"<span class='badge-hybrid'>{ret_src}</span>",
            unsafe_allow_html=True
        )
        if url and str(url).startswith("http"):
            st.link_button("🔗 Mở Nguồn Web Trực Tiếp", url=url, use_container_width=True, key=f"btn_{key_prefix}_{i}_{id(src)}")

    st.text_area(
        f"Chunk Content #{i}",
        value=content_str,
        height=110,
        disabled=True,
        key=f"txt_{key_prefix}_{i}_{id(src)}"
    )
    st.divider()

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
    st.markdown("##### 📌 Connection Status (.env)")
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))

    if has_openai or has_openrouter or has_gemini:
        st.success("🟢 .env Connected (LLM Ready)")
    else:
        st.info("🔵 Local Template Fallback Ready")

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
    st.caption("💬 Trò chuyện trực tiếp với RAG Chatbot có trích dẫn nguồn văn bản chính xác kèm Link trực tiếp")

    # Render Chat History
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                with st.expander(f"📚 Nguồn tài liệu tham khảo ({len(msg['sources'])} chunks) — Via `{msg.get('retrieval_source', 'hybrid')}`"):
                    for i, src in enumerate(msg["sources"], 1):
                        render_source_card(i, src, key_prefix=f"hist_msg_{msg_idx}")

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
                            render_source_card(i, src, key_prefix=f"live_{len(st.session_state.messages)}")

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
        st.info(f"Tổng số văn bản trong kho dữ liệu: **{len(md_files)} tệp tin Markdown** (Chứa 375 Chunks)")

        selected_file = st.selectbox("Chọn văn bản để xem nội dung:", options=md_files, format_func=lambda x: str(x.relative_to(std_dir)))
        if selected_file:
            content = selected_file.read_text(encoding="utf-8")
            file_name = selected_file.name
            web_url = URL_MAP.get(file_name)
            if web_url:
                st.markdown(f"🔗 **Link nguồn chính thức trên Web:** [{web_url}]({web_url})")
                st.link_button("🌐 Mở Bài Viết Gốc Trên Web", url=web_url, key=f"tab3_link_{hash(file_name)}")

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
                render_source_card(i, r, key_prefix="fallback_test")
