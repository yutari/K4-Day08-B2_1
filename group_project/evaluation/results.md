# Báo cáo A/B RAG

Đánh giá offline trên **16** câu hỏi Golden Dataset. Các metric dùng cosine similarity của embedding local, vì vậy có thể chạy lặp lại không cần API evaluator; đây không phải kết quả RAGAS/DeepEval.

| Metric | A: Dense-only | B: Hybrid + Cross-Encoder | Chênh lệch |
|---|---:|---:|---:|
| Faithfulness (độ trung thực) | 0.9724 | 0.7342 | -23.82% |
| Answer Relevance (độ liên quan) | 0.7054 | 0.7190 | +1.36% |
| Context Recall (độ phủ) | 0.6307 | 0.6193 | -1.14% |
| Context Precision (độ chính xác) | 0.6335 | 0.6296 | -0.39% |
| **Tổng thể** | **0.7355** | **0.6755** | **-6.00%** |

## Lưu ý

- Config A chỉ dùng Chroma dense retrieval.
- Config B dùng Dense + BM25 + weighted RRF + Cross-Encoder; PageIndex local được dùng khi confidence thấp.
- Muốn đo Faithfulness theo RAGAS/DeepEval cần cấu hình evaluator LLM riêng; không nên gắn nhãn các metric offline này là RAGAS.
