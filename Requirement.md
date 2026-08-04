# BẢN MÔ TẢ YÊU CẦU DỰ ÁN (REQUIREMENT SPECIFICATION)
## Dự Án Nhóm: E-Commerce Support RAG Chatbot & Evaluation Pipeline

---

## 1. TỔNG QUAN DỰ ÁN (PROJECT OVERVIEW)

Dự án nhóm tập trung vào việc tích hợp toàn bộ Pipeline Retrieval-Augmented Generation (RAG) từ các Task 1-10 cá nhân để hoàn thiện 2 sản phẩm chính:
1. **RAG Chatbot Hỗ Trợ Thương Mại Điện Tử:** Ứng dụng chatbot tương tác trả lời thắc mắc về chính sách sàn TMĐT (đổi trả, thanh toán, quy định người bán, vận chuyển, bảo hành).
2. **Hệ Thống Đánh Giá RAG (RAG Evaluation Pipeline):** Pipeline tự động hóa đánh giá chất lượng RAG dựa trên bộ **Golden Dataset (≥15 Q&A pairs)** với các khung đo lường tiêu chuẩn (RAGAS / DeepEval) và thực hiện thử nghiệm **A/B Testing** giữa các cấu hình Pipeline.

---

## 2. CHI TIẾT YÊU CẦU (REQUIREMENTS BREAKDOWN)

### YÊU CẦU 1: RAG CHATBOT SẢN PHẨM NHÓM (E-COMMERCE RAG CHATBOT)

#### 1.1. Giao Diện Tương Tác (User Interface - Streamlit)
- [x] **Giao diện Chat chuẩn UI/UX:** Sử dụng Streamlit (`st.chat_message`, `st.chat_input`) tạo trải nghiệm nhắn tin hiện đại.
- [x] **Bộ nhớ hội thoại (Conversation Memory):** Lưu giữ lịch sử chat (`st.session_state.messages`) cho phép người dùng đặt câu hỏi nối tiếp (follow-up questions).
- [x] **Trích dẫn minh bạch (Citations & Sources):** 
  - Mỗi câu trả lời của Bot phải có phần trích dẫn nguồn cụ thể (tên văn bản/trang, mục chính sách).
  - Tích hợp khung hiển thị danh sách **Source Documents** đã dùng (với điểm số tương đồng/similarity score, đoạn trích context, đường dẫn file).
- [x] **Điều chỉnh Cấu hình Pipeline linh hoạt (Sidebar Config):**
  - Cho phép người dùng bật/tắt **Reranking**.
  - Tùy chỉnh tham số **Alpha** cho Hybrid Search (Dense vs BM25).
  - Lựa chọn Provider LLM / Embedding (Sentence-Transformers local, OpenAI, OpenRouter).

#### 1.2. Tích Hợp Pipeline RAG Nền Tảng (Task 1 - 10 Integration)
- [x] **Task 1 - 3 (Data Ingestion & Standardizing):** Chuẩn hóa tài liệu chính sách (.pdf, .docx, .html, .json) từ `data/landing/` thành Markdown tại `data/standardized/`.
- [x] **Task 4 - 6 (Chunking, Hybrid Search & Indexing):** 
  - Phân đoạn tài liệu (Text Splitter: Chunk size 500-1000, Overlap 100).
  - Lưu trữ Vector Index trong ChromaDB và tạo BM25 Index cho Keyword Search.
  - Tích hợp **Hybrid Search** kết hợp Sparse (BM25) và Dense Retrieval với thuật toán Reciprocal Rank Fusion (RRF) hoặc Convex Combination.
- [x] **Task 7 - 8 (Reranking & PageIndex Fallback):**
  - Áp dụng Cross-Encoder Reranking (hoặc Jina Reranker) để tinh chỉnh thứ tự tài liệu.
  - Tích hợp cơ chế fallback sang **PageIndex (Vectorless RAG)** khi độ tin cậy tìm kiếm thấp.
- [x] **Task 9 - 10 (Query Expansion & Generation with Citation):**
  - Mở rộng câu hỏi (Query Expansion / Rewrite) dựa trên lịch sử hội thoại.
  - Sinh câu trả lời bám sát Context với trích dẫn rõ ràng, chống hallucination.

---

### YÊU CẦU 2: RAG EVALUATION PIPELINE & A/B TESTING

#### 2.1. Bộ Dữ Liệu Chuẩn (Golden Dataset)
- [x] **File lưu trữ:** `group_project/evaluation/golden_dataset.json`
- [x] **Quy mô:** Tối thiểu **15+ cặp Q&A** đa dạng covering các chủ đề TMĐT (Đổi trả hoàn tiền, Phương thức thanh toán, Sản phẩm cấm bán, Vận chuyển, Xử lý khiếu nại, Khuyến mãi).
- [x] **Cấu trúc mỗi Test Case:**
  ```json
  {
    "question": "Câu hỏi từ người dùng...",
    "expected_answer": "Câu trả lời chuẩn mực (Ground Truth)...",
    "expected_context": "Văn bản/Đoạn chính sách làm căn cứ..."
  }
  ```

#### 2.2. Khung Đánh Giá (Evaluation Framework - RAGAS / DeepEval)
- [x] **File thực thi:** `group_project/evaluation/eval_pipeline.py`
- [x] **Đo lường 4 chỉ số cốt lõi (Core RAG Triad Metrics):**
  1. **Faithfulness (Độ trung thực):** Kiểm tra câu trả lời sinh ra có hoàn toàn bám sát vào Context thu thập được hay không (chống nói suông/chế tạo thông tin).
  2. **Answer Relevance (Độ liên quan câu trả lời):** Đánh giá câu trả lời có giải quyết trực tiếp và đúng trọng tâm câu hỏi người dùng hay không.
  3. **Context Recall (Độ phủ của Context):** Đánh giá Retriever có lấy về đầy đủ các đoạn thông tin cần thiết chứa trong Ground Truth hay không.
  4. **Context Precision (Độ chính xác của Context):** Đánh giá tỉ lệ các đoạn context thu thập về thực sự có ích đối với câu hỏi (được xếp hạng ở vị trí cao).

#### 2.3. Thử Nghiệm So Sánh A/B (A/B Testing Matrix)
- [x] Chạy Evaluation trên **ít nhất 2 cấu hình khác nhau** của RAG Pipeline để so sánh hiệu năng:
  - **Config A (Baseline - Dense Only):** Tìm kiếm ngữ nghĩa thuần túy với ChromaDB Vector Search (không Reranking, không Hybrid).
  - **Config B (Advanced RAG - Hybrid + Reranking):** Kết hợp Sparse BM25 + Dense Vector (RRF) và lọc lại qua Cross-Encoder Reranker.
  - *(Tùy chọn) Config C (Full Pipeline with Query Expansion & PageIndex Fallback).*

#### 2.4. Báo Cáo Đánh Giá (Evaluation Report)
- [x] **File báo cáo:** `group_project/evaluation/results.md`
- [x] **Nội dung bắt buộc:**
  - Bảng tổng hợp điểm số chi tiết từng Metric cho mỗi Config (Định dạng Markdown Table).
  - Bảng so sánh trực quan A/B Testing.
  - Phân tích các trường hợp tệ nhất (**Worst Performers Analysis**) — chỉ ra nguyên nhân thất bại (Retriever miss, LLM Hallucination, Chunking quá ngắn/dài).
  - Đề xuất giải pháp cải tiến kiến trúc cho các bài toán TMĐT thực tế.

---

### YÊU CẦU CHUNG & THƯ MỤC NỘP BÀI (DELIVERABLES)

```
K4-Day08-RAG-Pipeline/
├── Requirement.md                          # [NEW] File mô tả yêu cầu dự án nhóm
├── app.py                                  # Giao diện Chatbot Streamlit tích hợp
├── group_project/
│   ├── README.md                           # Mô tả kiến trúc, hướng dẫn chạy & phân công
│   └── evaluation/
│       ├── golden_dataset.json             # Bộ 15+ cặp Q&A mẫu chuẩn
│       ├── eval_pipeline.py                # Script chạy tự động RAGAS/DeepEval
│       └── results.md                      # Báo cáo kết quả chi tiết & A/B testing
└── src/                                    # Các module RAG Task 1-10
```

---

## 3. PHÂN CÔNG VÀ KẾ HOẠCH THỰC HIỆN (TASK ASSIGNMENT MATRIX)

| STT | Phân Hệ / Hạng Mục | Mô Tả Chi Tiết | Người Thực Hiện | Thời Hạn | Trạng Thái |
|:---:|:---|:---|:---:|:---:|:---:|
| 1 | **Data & Golden Dataset** | Xây dựng bộ 15+ cặp Q&A TMĐT chuẩn tại `golden_dataset.json` | Nhóm Data | Day 1 | ⏳ Pending |
| 2 | **Chatbot UI & Memory** | Hoàn thiện Streamlit UI, hiển thị Citations & Sources, lưu Session state | Developer UI | Day 1-2 | ⏳ Pending |
| 3 | **RAG Pipeline Core** | Kết nối Retriever (Hybrid + Reranking) và LLM Generator (Task 9-10) vào `app.py` | Core Engineer | Day 2 | ⏳ Pending |
| 4 | **Eval Pipeline Engine** | Lập trình `eval_pipeline.py` sử dụng RAGAS/DeepEval, chạy A/B Testing | Eval Specialist | Day 2-3 | ⏳ Pending |
| 5 | **Report & Documentation** | Phân tích Worst Performers, viết `results.md` và hoàn thiện `group_project/README.md` | Tech Lead / All | Day 3 | ⏳ Pending |

---

## 4. TIÊU CHÍ BÁO CÁO & DEMO (DEMO CHECKLIST)

1. **Khả năng phản hồi Chatbot:** Trả lời chính xác câu hỏi chính sách, trích dẫn rõ tên tài liệu, ghi nhớ câu hỏi nối tiếp mượt mà.
2. **Khả năng chạy Evaluation tự động:** Chạy `python group_project/evaluation/eval_pipeline.py` không phát sinh lỗi và xuất ra kết quả đo lường rõ ràng.
3. **Báo cáo kết quả A/B Testing:** Chứng minh được sự cải thiện chỉ số (Faithfulness, Context Precision/Recall) khi bật Reranking / Hybrid Search so với Dense-only baseline.
