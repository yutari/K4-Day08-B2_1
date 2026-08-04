# PHÂN CÔNG NHIỆM VỤ VÀ TRẠNG THÁI THÀNH VIÊN NHÓM
## Dự Án: E-Commerce Support RAG Chatbot & Evaluation Pipeline (Nhóm K4-B2_1)

---

## 1. Danh Sách Thành Viên & Bảng Phân Công Chi Tiết

| STT | Họ và Tên | Mã Sinh Viên | Vai Trò | Phân Công Nhiệm Vụ Chi Tiết | Trạng Thái |
|:---:|:---|:---:|:---|:---|:---:|
| 1 | **Phan Văn Hoàng Nam** | 2A202601160 | **Tech Lead & Core RAG** | - Chủ trì thiết kế kiến trúc hệ thống 7 phân tầng (`Architecture.md`).<br>- Phát triển RAG Core Pipeline (Task 4-8: Chunking, Dense ChromaDB, Sparse BM25, Hybrid Search RRF, Cross-Encoder Reranking & PageIndex Fallback). | ✅ Hoàn thành |
| 2 | **Trương Minh Hoàng** | 2A202601262 | **Generation & Query Processing** | - Phát triển Phân hệ Query Expansion / Rewrite (Task 9).<br>- Phát triển Phân hệ Sinh câu trả lời kèm Trích dẫn (Task 10 Citation Generation) & Chống Hallucination. Tích hợp LLM Provider (OpenAI/OpenRouter). | ✅ Hoàn thành |
| 3 | **Tạ Kim Ngân** | 2A202601258 | **UI/UX & Conversation Memory** | - Lập trình Giao diện Chatbot Streamlit (`app.py`).<br>- Tích hợp Bộ nhớ hội thoại (`st.session_state`), Sidebar tùy chỉnh thông số (Alpha, Reranking Toggle, Model Selection) và khung hiển thị Source Documents. | ✅ Hoàn thành |
| 4 | **Phạm Thế Đăng** | 2A202601766 | **Data & Golden Dataset** | - Thu thập, làm sạch và chuẩn hóa dữ liệu chính sách TMĐT từ Task 1-3 vào `data/standardized/`.<br>- Xây dựng bộ dữ liệu chuẩn **Golden Dataset** (`golden_dataset.json`) với 16+ cặp Q&A TMĐT chuẩn. | ✅ Hoàn thành |
| 5 | **Đào Trung Hiếu** | 2A202601238 | **Evaluation & Documentation** | - Lập trình Pipeline đánh giá tự động (`eval_pipeline.py`) với RAGAS / DeepEval.<br>- Thực hiện A/B Testing so sánh các cấu hình, Phân tích Worst Performers, biên soạn `results.md` và `Requirement.md`. | ✅ Hoàn thành |

---

## 2. Tổng Quan Tiến Độ Dự Án

- **Tổng số thành viên:** 05
- **Tỷ lệ hoàn thành công việc:** 100% (5/5 thành viên hoàn thành nhiệm vụ)
- **Các sản phẩm bàn giao:**
  1. Giao diện Chatbot Streamlit ([app.py](file:///d:/Lap8/K4-Day08-RAG-Pipeline/app.py))
  2. Pipeline đánh giá tự động & A/B Testing ([eval_pipeline.py](file:///d:/Lap8/K4-Day08-RAG-Pipeline/group_project/evaluation/eval_pipeline.py))
  3. Bộ Golden Dataset 16 Q&A ([golden_dataset.json](file:///d:/Lap8/K4-Day08-RAG-Pipeline/group_project/evaluation/golden_dataset.json))
  4. Báo cáo đánh giá chi tiết ([results.md](file:///d:/Lap8/K4-Day08-RAG-Pipeline/group_project/evaluation/results.md))
  5. Bộ tài liệu hệ thống ([Requirement.md](file:///d:/Lap8/K4-Day08-RAG-Pipeline/Requirement.md), [Architecture.md](file:///d:/Lap8/K4-Day08-RAG-Pipeline/Architecture.md), [README.md](file:///d:/Lap8/K4-Day08-RAG-Pipeline/group_project/README.md))

