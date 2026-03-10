from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from backend.agent.agent_flow import AgenticRAG, normalize_key_from_question
from backend.evaluation.rag_eval import (
    answer_relevance,
    context_precision,
    faithfulness,
    precision_recall_at_k,
    retrieval_relevance_delta,
    safe_mean,
)
from backend.observability import mlflow_tracker
from backend.rag.rag_store import RAGStore


async def run_eval(
    store: RAGStore,
    dataset_path: str,
    namespace: str,
    k: int = 5,
    agent: Optional[AgenticRAG] = None,
) -> Dict[str, Any]:
    dataset = json.load(open(dataset_path, encoding="utf-8"))
    rows: List[Dict[str, Any]] = []

    with mlflow_tracker.start_run(
        run_name="rag-evaluation",
        tags={"kind": "evaluation", "namespace": namespace, "dataset_path": dataset_path},
    ):
        mlflow_tracker.log_params({"dataset_path": dataset_path, "namespace": namespace, "k": k})

        for idx, item in enumerate(dataset, start=1):
            q = item["question"]
            relevant_patterns = item.get("relevant_chunks", [])
            answer_patterns = item.get("answer_patterns", []) or relevant_patterns

            semantic_chunks = await store.query_semantic(namespace, q, top_k=max(k, 8))
            semantic_texts = [c.text for c in semantic_chunks]
            semantic_scores = precision_recall_at_k(semantic_texts, relevant_patterns, k)
            semantic_scores["semantic_context_precision"] = context_precision(semantic_texts, relevant_patterns, k)

            structured_scores: Dict[str, Any] = {
                "structured_precision@k": 0.0,
                "structured_recall@k": 0.0,
                "structured_relevant_retrieved": 0,
                "structured_total_relevant": 0,
            }
            key = normalize_key_from_question(q)
            if key:
                facts = await store.query_structured(namespace, key=key, limit=max(k, 8))
                structured_texts = [
                    " ".join([
                        str(f.get("entity", "")),
                        str(f.get("key", "")),
                        str(f.get("value", "")),
                        str(f.get("evidence_text", "")),
                    ]).strip()
                    for f in facts
                ]
                raw_structured = precision_recall_at_k(structured_texts, relevant_patterns, k)
                structured_scores = {
                    "structured_precision@k": raw_structured["precision@k"],
                    "structured_recall@k": raw_structured["recall@k"],
                    "structured_relevant_retrieved": raw_structured["relevant_retrieved"],
                    "structured_total_relevant": raw_structured["total_relevant"],
                }

            answer_metrics: Dict[str, Any] = {
                "answer_relevance": 0.0,
                "faithfulness": 0.0,
                "answer_route": "",
            }
            if agent is not None:
                answer_out = await agent.answer(q, top_k=max(k, 8))
                answer_text = answer_out.get("answer", "")
                answer_metrics = {
                    "answer_relevance": answer_relevance(answer_text, q, answer_patterns),
                    "faithfulness": faithfulness(answer_text, semantic_texts),
                    "answer_route": answer_out.get("route", ""),
                }

            row = {
                "question": q,
                "namespace": namespace,
                **semantic_scores,
                **structured_scores,
                **answer_metrics,
                "retrieval_delta": retrieval_relevance_delta(
                    semantic_scores["precision@k"],
                    structured_scores["structured_precision@k"],
                ),
            }
            rows.append(row)
            mlflow_tracker.log_metrics(
                {
                    "semantic_precision@k": row["precision@k"],
                    "semantic_recall@k": row["recall@k"],
                    "semantic_context_precision": row["semantic_context_precision"],
                    "structured_precision@k": row["structured_precision@k"],
                    "structured_recall@k": row["structured_recall@k"],
                    "answer_relevance": row["answer_relevance"],
                    "faithfulness": row["faithfulness"],
                    "retrieval_delta": row["retrieval_delta"],
                },
                step=idx,
            )

        summary = {
            "mean_semantic_precision@k": safe_mean([r["precision@k"] for r in rows]),
            "mean_semantic_recall@k": safe_mean([r["recall@k"] for r in rows]),
            "mean_context_precision": safe_mean([r["semantic_context_precision"] for r in rows]),
            "mean_structured_precision@k": safe_mean([r["structured_precision@k"] for r in rows]),
            "mean_structured_recall@k": safe_mean([r["structured_recall@k"] for r in rows]),
            "mean_answer_relevance": safe_mean([r["answer_relevance"] for r in rows]),
            "mean_faithfulness": safe_mean([r["faithfulness"] for r in rows]),
            "mean_retrieval_delta": safe_mean([r["retrieval_delta"] for r in rows]),
            "rows": len(rows),
        }

        mlflow_tracker.log_metrics(summary)
        mlflow_tracker.log_dict({"summary": summary, "rows": rows}, "evaluation/results.json")
        return {"summary": summary, "rows": rows}
