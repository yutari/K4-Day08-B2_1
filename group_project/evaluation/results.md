# Báo cáo A/B RAG

## Thông tin lần chạy

- Thời điểm: **2026-08-04 10:14 UTC**
- Golden Dataset: **16** câu hỏi
- Backend đánh giá: **RAGAS 0.4.3**
- Cấu hình evaluator: LLM `gpt-4o-mini`; embeddings `text-embedding-3-small`; đánh giá LLM-based qua RAGAS.
- Config A: **Dense-only retrieval + cùng policy generation**
- Config B: **Dense + BM25 + Weighted RRF + Cross-Encoder + cùng policy generation**

## So sánh 4 RAG Triad Metrics

| Metric | A: Dense-only | B: Hybrid + Cross-Encoder | Δ (điểm %) |
|---|---:|---:|---:|
| Faithfulness (độ trung thực) | 0.9792 | 0.8819 | -9.73 pp |
| Answer Relevancy (độ liên quan) | 0.3645 | 0.4082 | +4.37 pp |
| Context Recall (độ phủ) | 0.3750 | 0.6042 | +22.92 pp |
| Context Precision (độ chính xác) | 0.2941 | 0.6438 | +34.97 pp |
| **Tổng thể** | **0.5032** | **0.6345** | **+13.13 pp** |

## Phân tích A/B

- Faithfulness (độ trung thực) giảm **9.73 điểm %** ở Config B.
- Answer Relevancy (độ liên quan) tăng **4.37 điểm %** ở Config B.
- Context Recall (độ phủ) tăng **22.92 điểm %** ở Config B.
- Context Precision (độ chính xác) tăng **34.97 điểm %** ở Config B.

## Worst Performers — Config B (Hybrid + Cross-Encoder)

Các ca dưới đây được xếp theo điểm trung bình của 4 metrics, từ thấp đến cao. Nguyên nhân và đề xuất được suy ra từ các metric và trạng thái retrieval của từng ca, để nhóm có checklist cải thiện cụ thể.

| Hạng | Câu hỏi | Overall | 4 metrics | Nguyên nhân khả dĩ | Đề xuất cải thiện |
|---:|---|---:|---|---|---|
| 1 | Q15: Cách kích hoạt giảm giá Xu thưởng tại trang thanh toán như thế nào? | 0.1945 | faithfulness=0.7778, answer_relevancy=0.0000, context_recall=0.0000, context_precision=0.0000 | Evidence bị bỏ sót hoặc chưa đủ; Context có nhiều đoạn nhiễu; Câu trả lời chưa trực tiếp giải đáp câu hỏi | Tăng top_k, cải thiện chunking hoặc Query Expansion; Tinh chỉnh α BM25/Dense và Cross-Encoder reranking; Cải thiện Query Rewrite và prompt trả l… |
| 2 | Q14: Đơn vị vận chuyển hợp tác cùng sàn TMĐT gồm những đơn vị nào? | 0.2500 | faithfulness=1.0000, answer_relevancy=0.0000, context_recall=0.0000, context_precision=0.0000 | Evidence bị bỏ sót hoặc chưa đủ; Context có nhiều đoạn nhiễu; Câu trả lời chưa trực tiếp giải đáp câu hỏi | Tăng top_k, cải thiện chunking hoặc Query Expansion; Tinh chỉnh α BM25/Dense và Cross-Encoder reranking; Cải thiện Query Rewrite và prompt trả l… |
| 3 | Q5: Các phương thức vận chuyển đơn hàng được hỗ trợ bao gồm những gì? | 0.2903 | faithfulness=0.8333, answer_relevancy=0.3277, context_recall=0.0000, context_precision=0.0000 | Evidence bị bỏ sót hoặc chưa đủ; Context có nhiều đoạn nhiễu; Câu trả lời chưa trực tiếp giải đáp câu hỏi | Tăng top_k, cải thiện chunking hoặc Query Expansion; Tinh chỉnh α BM25/Dense và Cross-Encoder reranking; Cải thiện Query Rewrite và prompt trả l… |
| 4 | Q3: Người bán không được đăng bán những sản phẩm nào? | 0.3347 | faithfulness=0.7500, answer_relevancy=0.0000, context_recall=0.0000, context_precision=0.5889 | Evidence bị bỏ sót hoặc chưa đủ; Câu trả lời chưa trực tiếp giải đáp câu hỏi | Tăng top_k, cải thiện chunking hoặc Query Expansion; Cải thiện Query Rewrite và prompt trả lời ngắn gọn |
| 5 | Q13: Những lý do hợp lệ nào cho phép người mua gửi yêu cầu hoàn tiền? | 0.3633 | faithfulness=0.9167, answer_relevancy=0.1697, context_recall=0.0000, context_precision=0.3667 | Evidence bị bỏ sót hoặc chưa đủ; Context có nhiều đoạn nhiễu; Câu trả lời chưa trực tiếp giải đáp câu hỏi | Tăng top_k, cải thiện chunking hoặc Query Expansion; Tinh chỉnh α BM25/Dense và Cross-Encoder reranking; Cải thiện Query Rewrite và prompt trả l… |

## Cách tái lập RAGAS thật

1. Cài đúng dependencies đã pin trong `requirements.txt`.
2. Cấu hình `RAGAS_EVALUATOR_API_KEY` (hoặc `OPENAI_API_KEY`) và, nếu dùng endpoint tương thích OpenAI, đặt thêm `RAGAS_EVALUATOR_BASE_URL`.
3. Chạy `python group_project/evaluation/eval_pipeline.py --require-ragas`.

`--require-ragas` sẽ fail rõ ràng thay vì thay bằng proxy nếu evaluator không sẵn sàng. Chạy không có flag sẽ tự fallback để chatbot/demo không bị chặn; report luôn ghi rõ backend thực tế đã dùng.
