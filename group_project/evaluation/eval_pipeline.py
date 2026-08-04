"""Run a reproducible A/B evaluation for the E-commerce RAG pipeline.

The preferred backend is RAGAS 0.1.x.  It evaluates both configurations with
four LLM-based RAG metrics: faithfulness, answer relevancy, context recall and
context precision.  The script deliberately falls back to a clearly-labelled
local cosine proxy only if RAGAS, its evaluator configuration, or the evaluator
service is unavailable.  That fallback keeps demos/CI usable, but is never
reported as a RAGAS result.

Required for a real RAGAS run:
    RAGAS_EVALUATOR_API_KEY (or OPENAI_API_KEY)

Optional OpenAI-compatible settings:
    RAGAS_EVALUATOR_MODEL, RAGAS_EVALUATOR_EMBEDDING_MODEL,
    RAGAS_EVALUATOR_BASE_URL
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.task5_semantic_search import semantic_search
from src.task9_retrieval_pipeline import retrieve

load_dotenv(PROJECT_ROOT / ".env")

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
METRIC_KEYS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")
METRIC_LABELS = {
    "faithfulness": "Faithfulness (độ trung thực)",
    "answer_relevancy": "Answer Relevancy (độ liên quan)",
    "context_recall": "Context Recall (độ phủ)",
    "context_precision": "Context Precision (độ chính xác)",
}
RAGAS_COLUMN_ALIASES = {
    "faithfulness": ("faithfulness",),
    "answer_relevancy": ("answer_relevancy", "answer_relevance"),
    "context_recall": ("context_recall",),
    "context_precision": ("context_precision",),
}
_EMBEDDING_MODEL: SentenceTransformer | None = None


@dataclass(frozen=True)
class RagasRuntime:
    """The RAGAS objects needed to evaluate a complete configuration."""

    dataset_class: Any
    evaluate_function: Any
    metrics: list[Any]
    llm: Any
    embeddings: Any
    ragas_version: str
    model_name: str
    embedding_model_name: str


def get_eval_model() -> SentenceTransformer:
    """Load the local model only for the explicitly-labelled offline fallback."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


def cosine_similarity(left: str, right: str) -> float:
    vectors = get_eval_model().encode([left or " ", right or " "], normalize_embeddings=True)
    return max(0.0, min(1.0, float(np.dot(vectors[0], vectors[1]))))


def load_golden_dataset() -> list[dict[str, str]]:
    """Read and validate the Golden Dataset before an expensive evaluator run."""
    data = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("golden_dataset.json must be a non-empty JSON list.")

    required_fields = {"question", "expected_answer", "expected_context"}
    for index, item in enumerate(data, start=1):
        missing = required_fields - item.keys() if isinstance(item, dict) else required_fields
        if not isinstance(item, dict) or missing:
            raise ValueError(f"Golden Dataset item {index} is missing fields: {sorted(missing)}")
    return data


def _answer_from_chunks(question: str, chunks: list[dict]) -> str:
    """Use the production generator for either retrieval configuration.

    ``generate_with_citation`` always invokes the hybrid retriever, so the
    dense-only baseline cannot call it directly.  Reusing Task 10's context
    formatting/provider failover keeps the answer-generation policy identical
    for both arms of the A/B test.  If no generator key is configured, the same
    extractive fallback is used for both arms.
    """
    from src.task10_generation import (
        _extractive_fallback,
        _generate_from_providers,
        format_context,
        reorder_for_llm,
        validate_and_normalize_citations,
    )

    ordered_chunks = reorder_for_llm(chunks)
    if not ordered_chunks:
        return _extractive_fallback(ordered_chunks)

    user_message = (
        "CONTEXT:\n"
        f"{format_context(ordered_chunks)}\n\n"
        f"CÂU HỎI: {question}"
    )
    try:
        generated_answer = _generate_from_providers(user_message)
    except Exception:
        # Generation must not prevent evaluation of retrieval quality.
        generated_answer = ""

    # Apply exactly the same post-generation citation safety gate as Task 10.
    # An invalid model citation must not contaminate answer-quality evaluation.
    validated_answer = validate_and_normalize_citations(generated_answer, ordered_chunks)
    return validated_answer or _extractive_fallback(ordered_chunks)


def _source_labels(chunks: list[dict]) -> list[str]:
    labels: list[str] = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {}) or {}
        labels.append(str(metadata.get("source") or metadata.get("relative_path") or "unknown"))
    return labels


def run_evaluation_config(
    golden_dataset: list[dict[str, str]],
    config_name: str,
    use_hybrid_and_rerank: bool,
) -> list[dict[str, Any]]:
    """Collect answers and contexts for one full pipeline configuration."""
    records: list[dict[str, Any]] = []
    print(f"Running {config_name} on {len(golden_dataset)} questions")

    for index, item in enumerate(golden_dataset, start=1):
        question = item["question"]
        pipeline_error = ""
        try:
            if use_hybrid_and_rerank:
                chunks = retrieve(question, top_k=5, use_reranking=True)
                retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"
            else:
                chunks = semantic_search(question, top_k=5)
                retrieval_source = "dense" if chunks else "none"
        except Exception as exc:
            chunks = []
            retrieval_source = "error"
            pipeline_error = f"Retrieval failed: {type(exc).__name__}: {exc}"

        contexts = [str(chunk.get("content", "")) for chunk in chunks if chunk.get("content")]
        try:
            answer = _answer_from_chunks(question, chunks)
        except Exception as exc:
            answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."
            generation_error = f"Generation failed: {type(exc).__name__}: {exc}"
            pipeline_error = "; ".join(part for part in (pipeline_error, generation_error) if part)
        records.append(
            {
                "id": index,
                "question": question,
                "expected_answer": item["expected_answer"],
                "expected_context": item["expected_context"],
                "answer": answer,
                "contexts": contexts,
                "source_labels": _source_labels(chunks),
                "retrieval_source": retrieval_source,
                "pipeline_error": pipeline_error,
                "metrics": {},
            }
        )
        print(f"[{index}/{len(golden_dataset)}] retrieved={len(contexts)} source={retrieval_source}")
    return records


def _make_langchain_clients(api_key: str, model_name: str, embedding_model_name: str, base_url: str | None) -> tuple[Any, Any]:
    """Support both current and older langchain-openai constructor aliases."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    common_chat: dict[str, Any] = {"model": model_name, "api_key": api_key, "temperature": 0}
    common_embeddings: dict[str, Any] = {"model": embedding_model_name, "api_key": api_key}
    if base_url:
        common_chat["base_url"] = base_url
        common_embeddings["base_url"] = base_url

    try:
        return ChatOpenAI(**common_chat), OpenAIEmbeddings(**common_embeddings)
    except TypeError:
        # langchain-openai 0.1.x accepts these pre-alias argument names.
        legacy_chat: dict[str, Any] = {
            "model_name": model_name,
            "openai_api_key": api_key,
            "temperature": 0,
        }
        legacy_embeddings: dict[str, Any] = {
            "model": embedding_model_name,
            "openai_api_key": api_key,
        }
        if base_url:
            legacy_chat["openai_api_base"] = base_url
            legacy_embeddings["openai_api_base"] = base_url
        return ChatOpenAI(**legacy_chat), OpenAIEmbeddings(**legacy_embeddings)


def get_ragas_runtime() -> tuple[RagasRuntime | None, str]:
    """Return a ready RAGAS runtime, or a human-readable unavailable reason."""
    api_key = (os.getenv("RAGAS_EVALUATOR_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None, "Thiếu RAGAS_EVALUATOR_API_KEY hoặc OPENAI_API_KEY."

    model_name = os.getenv("RAGAS_EVALUATOR_MODEL", "gpt-4o-mini").strip()
    embedding_model_name = os.getenv("RAGAS_EVALUATOR_EMBEDDING_MODEL", "text-embedding-3-small").strip()
    base_url = (os.getenv("RAGAS_EVALUATOR_BASE_URL") or "").strip() or None
    if not model_name or not embedding_model_name:
        return None, "RAGAS evaluator model hoặc embedding model đang để trống."

    try:
        import ragas
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        llm, embeddings = _make_langchain_clients(api_key, model_name, embedding_model_name, base_url)
    except Exception as exc:
        return None, f"Không khởi tạo được RAGAS: {type(exc).__name__}: {exc}"

    return (
        RagasRuntime(
            dataset_class=Dataset,
            evaluate_function=evaluate,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=llm,
            embeddings=embeddings,
            ragas_version=getattr(ragas, "__version__", "unknown"),
            model_name=model_name,
            embedding_model_name=embedding_model_name,
        ),
        "",
    )


def _ragas_contexts(record: dict[str, Any]) -> list[str]:
    """RAGAS requires a non-empty list even when retrieval had no candidates."""
    return record["contexts"] or ["Không có ngữ cảnh nào được truy xuất cho câu hỏi này."]


def _normalise_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return round(max(0.0, min(1.0, score)), 4)


def _result_rows(result: Any) -> list[dict[str, Any]]:
    """Extract per-row RAGAS scores across the supported 0.1.x result shapes."""
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        return frame.to_dict(orient="records")

    scores = getattr(result, "scores", None)
    if scores is not None and hasattr(scores, "to_pandas"):
        return scores.to_pandas().to_dict(orient="records")
    if isinstance(result, dict):
        row_count = len(next(iter(result.values()), []))
        return [{key: values[index] for key, values in result.items()} for index in range(row_count)]
    raise TypeError(f"Unsupported RAGAS result type: {type(result).__name__}")


def evaluate_with_ragas(records: list[dict[str, Any]], runtime: RagasRuntime) -> None:
    """Mutate records with genuine RAGAS scores for the four required metrics."""
    dataset = runtime.dataset_class.from_dict(
        {
            "question": [record["question"] for record in records],
            "answer": [record["answer"] for record in records],
            "contexts": [_ragas_contexts(record) for record in records],
            "ground_truth": [record["expected_answer"] for record in records],
        }
    )
    try:
        result = runtime.evaluate_function(
            dataset,
            metrics=runtime.metrics,
            llm=runtime.llm,
            embeddings=runtime.embeddings,
            raise_exceptions=False,
        )
    except TypeError as exc:
        # Some patch versions do not expose raise_exceptions; retry their API.
        if "raise_exceptions" not in str(exc):
            raise
        result = runtime.evaluate_function(
            dataset,
            metrics=runtime.metrics,
            llm=runtime.llm,
            embeddings=runtime.embeddings,
        )

    rows = _result_rows(result)
    if len(rows) != len(records):
        raise RuntimeError(f"RAGAS returned {len(rows)} rows for {len(records)} records.")

    for record, row in zip(records, rows):
        metrics: dict[str, float | None] = {}
        for key, aliases in RAGAS_COLUMN_ALIASES.items():
            metrics[key] = next((_normalise_score(row.get(alias)) for alias in aliases if alias in row), None)
        record["metrics"] = metrics

    missing_scores = [
        f"Q{record['id']}:{key}"
        for record in records
        for key in METRIC_KEYS
        if record["metrics"].get(key) is None
    ]
    if missing_scores:
        preview = ", ".join(missing_scores[:8])
        raise RuntimeError(f"RAGAS did not return all four numeric metrics ({preview}).")


def calculate_offline_proxy_metrics(record: dict[str, Any]) -> dict[str, float]:
    """Local, deterministic proxy used only when a real RAGAS run is unavailable."""
    contexts = record["contexts"]
    combined_context = " ".join(contexts)
    precision_values = [cosine_similarity(record["expected_answer"], context) for context in contexts[:3]]
    return {
        "faithfulness": round(cosine_similarity(record["answer"], combined_context), 4) if contexts else 0.0,
        "answer_relevancy": round(cosine_similarity(record["question"], record["answer"]), 4),
        "context_recall": round(cosine_similarity(record["expected_answer"], combined_context), 4) if contexts else 0.0,
        "context_precision": round(float(np.mean(precision_values)), 4) if precision_values else 0.0,
    }


def evaluate_with_offline_proxy(records: list[dict[str, Any]]) -> None:
    for record in records:
        record["metrics"] = calculate_offline_proxy_metrics(record)


def _average(records: list[dict[str, Any]], key: str) -> float | None:
    values = [record["metrics"].get(key) for record in records]
    valid_values = [float(value) for value in values if value is not None]
    return round(sum(valid_values) / len(valid_values), 4) if valid_values else None


def _overall(metrics: dict[str, float | None]) -> float | None:
    values = [metrics.get(key) for key in METRIC_KEYS]
    valid_values = [float(value) for value in values if value is not None]
    return round(sum(valid_values) / len(valid_values), 4) if valid_values else None


def _score_cell(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "N/A"


def _delta_cell(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "N/A"
    return f"{(after - before) * 100:+.2f} pp"


def _markdown_cell(value: str, limit: int = 160) -> str:
    compact = " ".join(value.split())
    if len(compact) > limit:
        compact = f"{compact[:limit - 1]}…"
    return compact.replace("|", "\\|")


def _failure_analysis(record: dict[str, Any]) -> tuple[str, str]:
    """Return deterministic, actionable analysis for one low-scoring case."""
    metrics = record["metrics"]
    causes: list[str] = []
    recommendations: list[str] = []

    if record["pipeline_error"]:
        causes.append("Lỗi khi truy xuất dữ liệu")
        recommendations.append("Kiểm tra ChromaDB/index và log lỗi retrieval")
    if not record["contexts"]:
        causes.append("Không lấy được context")
        recommendations.append("Re-index corpus và kiểm tra truy vấn/chroma collection")
    if metrics.get("context_recall") is not None and metrics["context_recall"] < 0.55:
        causes.append("Evidence bị bỏ sót hoặc chưa đủ")
        recommendations.append("Tăng top_k, cải thiện chunking hoặc Query Expansion")
    if metrics.get("context_precision") is not None and metrics["context_precision"] < 0.55:
        causes.append("Context có nhiều đoạn nhiễu")
        recommendations.append("Tinh chỉnh α BM25/Dense và Cross-Encoder reranking")
    if metrics.get("faithfulness") is not None and metrics["faithfulness"] < 0.55:
        causes.append("Câu trả lời chưa bám sát evidence")
        recommendations.append("Siết prompt grounded answer và kiểm tra citation")
    if metrics.get("answer_relevancy") is not None and metrics["answer_relevancy"] < 0.55:
        causes.append("Câu trả lời chưa trực tiếp giải đáp câu hỏi")
        recommendations.append("Cải thiện Query Rewrite và prompt trả lời ngắn gọn")
    if record["retrieval_source"] == "pageindex":
        causes.append("Đã kích hoạt fallback PageIndex")
        recommendations.append("Đánh giá lại confidence threshold hoặc chất lượng index chính")

    if not causes:
        causes.append("Điểm tổng hợp thấp tương đối so với các câu còn lại")
    if not recommendations:
        recommendations.append("Review thủ công context và expected answer để tinh chỉnh corpus")
    return "; ".join(causes), "; ".join(recommendations)


def _worst_performers(records: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    ranked = sorted(records, key=lambda record: _overall(record["metrics"]) if _overall(record["metrics"]) is not None else -1.0)
    output: list[dict[str, Any]] = []
    for rank, record in enumerate(ranked[:limit], start=1):
        cause, recommendation = _failure_analysis(record)
        output.append(
            {
                "rank": rank,
                "id": record["id"],
                "question": record["question"],
                "overall": _overall(record["metrics"]),
                "causes": cause,
                "recommendations": recommendation,
                "sources": ", ".join(record["source_labels"][:3]) or "Không có",
                "metrics": record["metrics"],
            }
        )
    return output


def export_results(
    config_a: list[dict[str, Any]],
    config_b: list[dict[str, Any]],
    *,
    backend_name: str,
    backend_detail: str,
    fallback_reason: str = "",
) -> None:
    """Write a self-contained, numeric A/B report and a ranked failure analysis."""
    a_values = {key: _average(config_a, key) for key in METRIC_KEYS}
    b_values = {key: _average(config_b, key) for key in METRIC_KEYS}
    a_overall = _overall(a_values)
    b_overall = _overall(b_values)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    metric_rows = [
        f"| {METRIC_LABELS[key]} | {_score_cell(a_values[key])} | {_score_cell(b_values[key])} | {_delta_cell(a_values[key], b_values[key])} |"
        for key in METRIC_KEYS
    ]
    metric_rows.append(
        f"| **Tổng thể** | **{_score_cell(a_overall)}** | **{_score_cell(b_overall)}** | **{_delta_cell(a_overall, b_overall)}** |"
    )

    insight_lines: list[str] = []
    for key in METRIC_KEYS:
        before, after = a_values[key], b_values[key]
        if before is not None and after is not None:
            direction = "tăng" if after >= before else "giảm"
            insight_lines.append(f"- {METRIC_LABELS[key]} {direction} **{abs(after - before) * 100:.2f} điểm %** ở Config B.")

    worst_rows = []
    for item in _worst_performers(config_b):
        metric_summary = ", ".join(f"{key}={_score_cell(item['metrics'].get(key))}" for key in METRIC_KEYS)
        worst_rows.append(
            "| {rank} | Q{id}: {question} | {overall} | {metrics} | {causes} | {recommendations} |".format(
                rank=item["rank"],
                id=item["id"],
                question=_markdown_cell(item["question"], 105),
                overall=_score_cell(item["overall"]),
                metrics=_markdown_cell(metric_summary, 120),
                causes=_markdown_cell(item["causes"], 140),
                recommendations=_markdown_cell(item["recommendations"], 145),
            )
        )

    fallback_note = ""
    if fallback_reason:
        fallback_note = (
            "\n> **Lưu ý quan trọng:** Lần chạy này dùng proxy embedding local, **không phải điểm RAGAS**. "
            f"Lý do fallback: {_markdown_cell(fallback_reason, 260)}\n"
        )

    report = f"""# Báo cáo A/B RAG

## Thông tin lần chạy

- Thời điểm: **{timestamp}**
- Golden Dataset: **{len(config_b)}** câu hỏi
- Backend đánh giá: **{backend_name}**
- Cấu hình evaluator: {backend_detail}
- Config A: **Dense-only retrieval + cùng policy generation**
- Config B: **Dense + BM25 + Weighted RRF + Cross-Encoder + cùng policy generation**
{fallback_note}
## So sánh 4 RAG Triad Metrics

| Metric | A: Dense-only | B: Hybrid + Cross-Encoder | Δ (điểm %) |
|---|---:|---:|---:|
{chr(10).join(metric_rows)}

## Phân tích A/B

{chr(10).join(insight_lines) or '- Không đủ metric hợp lệ để tính chênh lệch.'}

## Worst Performers — Config B (Hybrid + Cross-Encoder)

Các ca dưới đây được xếp theo điểm trung bình của 4 metrics, từ thấp đến cao. Nguyên nhân và đề xuất được suy ra từ các metric và trạng thái retrieval của từng ca, để nhóm có checklist cải thiện cụ thể.

| Hạng | Câu hỏi | Overall | 4 metrics | Nguyên nhân khả dĩ | Đề xuất cải thiện |
|---:|---|---:|---|---|---|
{chr(10).join(worst_rows) or '| — | Không có dữ liệu | N/A | N/A | N/A | N/A |'}

## Cách tái lập RAGAS thật

1. Cài đúng dependencies đã pin trong `requirements.txt`.
2. Cấu hình `RAGAS_EVALUATOR_API_KEY` (hoặc `OPENAI_API_KEY`) và, nếu dùng endpoint tương thích OpenAI, đặt thêm `RAGAS_EVALUATOR_BASE_URL`.
3. Chạy `python group_project/evaluation/eval_pipeline.py --require-ragas`.

`--require-ragas` sẽ fail rõ ràng thay vì thay bằng proxy nếu evaluator không sẵn sàng. Chạy không có flag sẽ tự fallback để chatbot/demo không bị chặn; report luôn ghi rõ backend thực tế đã dùng.
"""
    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(f"[OK] Wrote {RESULTS_PATH}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Dense-only vs Hybrid RAG with RAGAS when configured.")
    parser.add_argument(
        "--require-ragas",
        action="store_true",
        help="Fail instead of using the offline proxy when RAGAS/evaluator is unavailable.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force the clearly-labelled local cosine proxy (useful for offline demos only).",
    )
    args = parser.parse_args(argv)
    if args.require_ragas and args.offline:
        parser.error("--require-ragas and --offline cannot be used together.")

    fallback_reason = ""
    backend_name = "RAGAS"
    backend_detail = ""
    if args.offline:
        fallback_reason = "Được ép bằng flag --offline."
        runtime = None
    else:
        runtime, unavailable_reason = get_ragas_runtime()
        fallback_reason = unavailable_reason
    if args.require_ragas and runtime is None:
        raise RuntimeError(f"Không thể chạy RAGAS: {fallback_reason}")

    dataset = load_golden_dataset()
    config_a = run_evaluation_config(dataset, "Config A (Dense-only)", use_hybrid_and_rerank=False)
    config_b = run_evaluation_config(dataset, "Config B (Hybrid + Cross-Encoder)", use_hybrid_and_rerank=True)

    if runtime is not None:
        try:
            evaluate_with_ragas(config_a, runtime)
            evaluate_with_ragas(config_b, runtime)
            backend_name = f"RAGAS {runtime.ragas_version}"
            backend_detail = (
                f"LLM `{runtime.model_name}`; embeddings `{runtime.embedding_model_name}`; "
                "đánh giá LLM-based qua RAGAS."
            )
            fallback_reason = ""
        except Exception as exc:
            fallback_reason = f"RAGAS evaluator không khả dụng: {type(exc).__name__}: {exc}"
            if args.require_ragas:
                raise RuntimeError(fallback_reason) from exc

    if fallback_reason:
        if args.require_ragas:
            raise RuntimeError(f"Không thể chạy RAGAS: {fallback_reason}")
        evaluate_with_offline_proxy(config_a)
        evaluate_with_offline_proxy(config_b)
        backend_name = "Offline cosine proxy (fallback, không phải RAGAS)"
        backend_detail = "SentenceTransformer all-MiniLM-L6-v2; chỉ dùng để giữ demo/CI chạy được."
        print(f"[WARN] {fallback_reason} Chuyển sang offline proxy.")

    export_results(
        config_a,
        config_b,
        backend_name=backend_name,
        backend_detail=backend_detail,
        fallback_reason=fallback_reason,
    )


if __name__ == "__main__":
    main()
