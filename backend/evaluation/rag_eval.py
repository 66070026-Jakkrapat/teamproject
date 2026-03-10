from __future__ import annotations

import math
import re
from typing import Dict, List


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[\wก-๙]{2,}", normalize_text(text))


def is_relevant(chunk_text: str, patterns: List[str]) -> bool:
    text = normalize_text(chunk_text)
    return any(normalize_text(p) in text for p in patterns or [])


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
        "total_relevant": total_relevant,
    }


def context_precision(retrieved_chunks: List[str], relevant_patterns: List[str], k: int) -> float:
    top_k = retrieved_chunks[:k]
    if not top_k:
        return 0.0
    return round(sum(1 for c in top_k if is_relevant(c, relevant_patterns)) / len(top_k), 4)


def answer_relevance(answer: str, question: str, answer_patterns: List[str]) -> float:
    answer_tokens = set(tokenize(answer))
    question_tokens = set(tokenize(question))
    expected_tokens = set()
    for pattern in answer_patterns or []:
        expected_tokens.update(tokenize(pattern))

    target_tokens = expected_tokens or question_tokens
    if not target_tokens:
        return 0.0
    overlap = len(answer_tokens & target_tokens)
    return round(overlap / max(1, len(target_tokens)), 4)


def faithfulness(answer: str, retrieved_chunks: List[str]) -> float:
    answer_tokens = set(tokenize(answer))
    context_tokens = set()
    for chunk in retrieved_chunks:
        context_tokens.update(tokenize(chunk))
    if not answer_tokens:
        return 0.0
    overlap = len(answer_tokens & context_tokens)
    return round(overlap / max(1, len(answer_tokens)), 4)


def retrieval_relevance_delta(semantic_precision: float, structured_precision: float) -> float:
    return round(semantic_precision - structured_precision, 4)


def safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
