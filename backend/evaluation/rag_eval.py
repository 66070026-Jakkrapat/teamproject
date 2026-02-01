# backend/evaluation/rag_eval.py
from __future__ import annotations

from typing import Dict, List

def is_relevant(chunk_text: str, patterns: List[str]) -> bool:
    text = (chunk_text or "").lower()
    return any((p or "").lower() in text for p in patterns)

def precision_recall_at_k(retrieved_chunks: List[str], relevant_patterns: List[str], k: int) -> Dict[str, float]:
    top_k = retrieved_chunks[:k]
    relevant_retrieved = sum(1 for c in top_k if is_relevant(c, relevant_patterns))
    total_relevant = sum(1 for c in retrieved_chunks if is_relevant(c, relevant_patterns))

    precision = relevant_retrieved / k if k else 0.0
    recall = relevant_retrieved / total_relevant if total_relevant else 0.0
    return {
        "precision@k": round(precision, 4),
        "recall@k": round(recall, 4),
        "relevant_retrieved": relevant_retrieved,
        "total_relevant": total_relevant
    }
