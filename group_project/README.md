# Bài Tập Nhóm — E-commerce Support RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về chính sách thương mại điện tử và hỗ trợ khách hàng liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

Xem code mẫu (DeepEval/RAGAS/TruLens) chi tiết trong `README.md` gốc mục "Yêu cầu 2".

# Bài Tập Nhóm — E-commerce Support RAG Chatbot & Evaluation Pipeline

## Mục Tiêu

Dự án nhóm hoàn thiện **Hệ thống RAG Chatbot hỗ trợ Thương mại Điện tử** và **Hệ thống Đánh giá Tự động (RAG Evaluation Pipeline)** dựa trên kiến trúc 7 phân tầng tiêu chuẩn.

---

## Deliverables & Trạng Thái Hoàn Thành

- [x] File `Requirement.md` — Mô tả chi tiết yêu cầu dự án
- [x] File `Architecture.md` — Bản thiết kế kiến trúc hệ thống 7 phân tầng
- [x] File `group_project/evaluation/golden_dataset.json` — 16 cặp Q&A TMĐT mẫu chuẩn
- [x] File `group_project/evaluation/eval_pipeline.py` — Script tự động đánh giá RAGAS/DeepEval
- [x] File `group_project/evaluation/results.md` — Bảng điểm đánh giá + A/B Testing Matrix + Phân tích Worst Performers
- [x] Giao diện Chatbot Streamlit (`app.py`) tích hợp Citations & Memory

---

## Kiến Trúc Hệ Thống

Chi tiết xem tại tài liệu kiến trúc: **[Architecture.md](file:///d:/Lap8/K4-Day08-RAG-Pipeline/Architecture.md)**

```
[User Query] ──> [Streamlit UI / Memory]
                       │
                       ▼
       ┌───────────────────────────────┐
       │   Hybrid Search Retrieval     │
       │ (Dense ChromaDB + Sparse BM25)│
       └───────────────┬───────────────┘
                       │ Reciprocal Rank Fusion (RRF)
                       ▼
       ┌───────────────────────────────┐
       │    Cross-Encoder Reranking    │
       └───────────────┬───────────────┘
                       │ (Fallback PageIndex if score < threshold)
                       ▼
       ┌───────────────────────────────┐
       │  Generation with Citations    │
       │   (OpenAI / OpenRouter LLM)   │
       └───────────────────────────────┘
```

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Nhóm Data | K4-B2_1 | Thu thập văn bản pháp lý, tin tức & Xây dựng Golden Dataset | [OK] Completed |
| Core Dev | K4-B2_1 | Lập trình Task 1 - 10 (Chunking, Hybrid Search, Rerank, Citation) | [OK] Completed |
| UI Lead | K4-B2_1 | Hoàn thiện Streamlit Chatbot UI & Source Document Expanders | [OK] Completed |
| Eval Specialist | K4-B2_1 | Lập trình `eval_pipeline.py`, chạy A/B Testing & viết `results.md` | [OK] Completed |

---

## Hướng Dẫn Chạy Dự Án

### 1. Kích hoạt môi trường và Cài đặt
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Chạy Kiểm Thử Bài Cá Nhân (Pytest)
```powershell
pytest tests/test_individual.py -v
```

### 3. Chạy Pipeline Đánh Giá RAG (Evaluation & A/B Testing)
```powershell
python group_project/evaluation/eval_pipeline.py
```
*(Kết quả đánh giá sẽ tự động được ghi vào `group_project/evaluation/results.md`)*

### 4. Khởi Chạy Ứng Dụng Chatbot Streamlit
```powershell
streamlit run app.py
```

