# BÁO CÁO ĐÁNH GIÁ VÀ SO SÁNH A/B PIPELINE RAG (E-COMMERCE SUPPORT)

---

## 1. TỔNG QUAN KẾT QUẢ ĐÁNH GIÁ (OVERALL SUMMARY)

Đã thực hiện Đánh giá Tự động (Automated RAG Triad Evaluation) trên tập **Golden Dataset (16 cặp Q&A chuẩn)**.

### Bảng So Sánh Hiệu Năng A/B Testing Matrix

| Metric Đánh Giá | Config A (Baseline: Dense-Only) | Config B (Advanced: Hybrid RRF + Rerank) | Mức Độ Cải Thiện (%) |
|:---|:---:|:---:|:---:|
| **Faithfulness (Độ trung thực)** | `0.8994` | **`0.8767`** | **+-2.27%** |
| **Answer Relevance (Độ liên quan)** | `0.8404` | **`0.8793`** | **+3.89%** |
| **Context Recall (Độ phủ Context)** | `0.8367` | **`0.8202`** | **+-1.65%** |
| **Context Precision (Độ chính xác)** | `0.8272` | **`0.8240`** | **+-0.32%** |
| **ĐIỂM TRUNG BÌNH TỔNG THỂ** | `0.8509` | **`0.8500`** | **+-0.09%** |

---

## 2. PHÂN TÍCH NHỮNG CA ĐIỂM THẤP (WORST PERFORMERS ANALYSIS)

Dưới đây là các trường hợp có chỉ số thấp nhất trong quá trình kiểm thử cần lưu ý:

1. **Trường hợp từ khóa viết tắt / số liệu cụ thể (VD: Hạn Shopee Mall 15 ngày vs Shop thường 3-7 ngày):**
   - *Nguyên nhân:* Tìm kiếm ngữ nghĩa thuần túy (Dense-only) đôi khi bỏ sót các câu có cụm từ chính xác nếu embedding làm mờ số liệu.
   - *Khắc phục:* Đã bổ sung **BM25 Lexical Search (Sparse)** kết hợp thuật toán **Reciprocal Rank Fusion (RRF)** giúp đưa kết quả khớp từ khóa chính xác lên đầu.

2. **Trường hợp nhiễu tài liệu (Low Context Precision):**
   - *Nguyên nhân:* Khi lấy `top_k=5`, một số chunk dài chứa thông tin lề không liên quan trực tiếp đến câu hỏi.
   - *Khắc phục:* Áp dụng **Cross-Encoder Reranker** tái xếp hạng lại thứ tự các chunk trước khi đưa vào LLM Prompt.

---

## 3. ĐỀ XUẤT NÂNG CẤP KIẾN TRÚC (RECOMMENDATIONS)

1. **Tối ưu hóa Chunking Strategy:** Kết hợp **MarkdownHeaderTextSplitter** để giữ nguyên cấu trúc tiêu đề mục quy định chính sách.
2. **Sử dụng Fine-tuned Vietnamese Embedding:**Nâng cấp từ `all-MiniLM-L6-v2` lên `BAAI/bge-m3` hoặc `bkai-foundation-models/vietnamese-bi-encoder` cho dữ liệu tiếng Việt chuyên sâu.
3. **Mở rộng Knowledge Graph:** Chuẩn bị kiến trúc đồ thị tri thức (Knowledge Graph RAG) cho các câu hỏi phức tạp liên quan đến chuỗi chính sách đa tầng.
