"""Repeatable offline A/B evaluation for the RAG pipeline.

It deliberately uses local embeddings, so a benchmark can run without spending
tokens.  RAGAS can be layered on top later when an evaluator LLM is configured.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import generate_with_citation

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
_MODEL = None


def get_eval_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


def cosine_similarity(left: str, right: str) -> float:
    vectors = get_eval_model().encode([left or " ", right or " "], normalize_embeddings=True)
    return max(0.0, min(1.0, float(np.dot(vectors[0], vectors[1]))))


def load_golden_dataset() -> list[dict]:
    return json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))


def calculate_metrics(question: str, answer: str, contexts: list[str], ground_truth: str, expected_context: str) -> dict:
    combined_context = " ".join(contexts)
    faithfulness = cosine_similarity(answer, combined_context) if contexts else 0.0
    answer_relevance = cosine_similarity(question, answer)
    context_recall = cosine_similarity(ground_truth, combined_context) if contexts else 0.0
    precision_values = [cosine_similarity(ground_truth, context) for context in contexts[:3]]
    context_precision = float(np.mean(precision_values)) if precision_values else 0.0
    return {
        "faithfulness": round(faithfulness, 4),
        "answer_relevance": round(answer_relevance, 4),
        "context_recall": round(context_recall, 4),
        "context_precision": round(context_precision, 4),
        "expected_context": expected_context,
    }


def run_evaluation_config(golden_dataset: list[dict], config_name: str, use_hybrid_and_rerank: bool) -> list[dict]:
    records = []
    print(f"Running {config_name} on {len(golden_dataset)} questions")
    for index, item in enumerate(golden_dataset, start=1):
        question = item["question"]
        if use_hybrid_and_rerank:
            response = generate_with_citation(question, top_k=5, use_reranking=True)
            answer, chunks = response["answer"], response["sources"]
        else:
            chunks = semantic_search(question, top_k=5)
            answer = "\n".join(chunk["content"][:350] for chunk in chunks[:2])
        contexts = [chunk.get("content", "") for chunk in chunks]
        metrics = calculate_metrics(question, answer, contexts, item["expected_answer"], item["expected_context"])
        records.append({"id": index, "question": question, "metrics": metrics})
        print(f"[{index}/{len(golden_dataset)}] {metrics}")
    return records


def _average(records: list[dict], key: str) -> float:
    return round(sum(item["metrics"][key] for item in records) / len(records), 4) if records else 0.0


def export_results(config_a: list[dict], config_b: list[dict]) -> None:
    keys = ("faithfulness", "answer_relevance", "context_recall", "context_precision")
    a_values = {key: _average(config_a, key) for key in keys}
    b_values = {key: _average(config_b, key) for key in keys}
    a_overall = round(sum(a_values.values()) / len(keys), 4)
    b_overall = round(sum(b_values.values()) / len(keys), 4)
    labels = {
        "faithfulness": "Faithfulness (độ trung thực)",
        "answer_relevance": "Answer Relevance (độ liên quan)",
        "context_recall": "Context Recall (độ phủ)",
        "context_precision": "Context Precision (độ chính xác)",
    }
    rows = []
    for key in keys:
        delta = (b_values[key] - a_values[key]) * 100
        rows.append(f"| {labels[key]} | {a_values[key]:.4f} | {b_values[key]:.4f} | {delta:+.2f}% |")
    rows.append(f"| **Tổng thể** | **{a_overall:.4f}** | **{b_overall:.4f}** | **{(b_overall-a_overall)*100:+.2f}%** |")
    report = f"""# Báo cáo A/B RAG

Đánh giá offline trên **{len(config_b)}** câu hỏi Golden Dataset. Các metric dùng cosine similarity của embedding local, vì vậy có thể chạy lặp lại không cần API evaluator; đây không phải kết quả RAGAS/DeepEval.

| Metric | A: Dense-only | B: Hybrid + Cross-Encoder | Chênh lệch |
|---|---:|---:|---:|
{chr(10).join(rows)}

## Lưu ý

- Config A chỉ dùng Chroma dense retrieval.
- Config B dùng Dense + BM25 + weighted RRF + Cross-Encoder; PageIndex local được dùng khi confidence thấp.
- Muốn đo Faithfulness theo RAGAS/DeepEval cần cấu hình evaluator LLM riêng; không nên gắn nhãn các metric offline này là RAGAS.
"""
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(f"[OK] Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    dataset = load_golden_dataset()
    dense_only = run_evaluation_config(dataset, "Config A (Dense-only)", use_hybrid_and_rerank=False)
    hybrid = run_evaluation_config(dataset, "Config B (Hybrid + Cross-Encoder)", use_hybrid_and_rerank=True)
    export_results(dense_only, hybrid)
