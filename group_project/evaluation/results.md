# Báo cáo A/B RAG

Chưa có kết quả hợp lệ cho phiên bản pipeline hiện tại. Chạy lệnh dưới đây sau khi cài dependencies và index lại knowledge base:

```powershell
python group_project/evaluation/eval_pipeline.py
```

Script sẽ ghi lại bảng so sánh Dense-only với Hybrid + Cross-Encoder bằng các metric offline, có thể tái lập bằng embedding local.
