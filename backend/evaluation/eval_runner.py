# backend/evaluation/eval_runner.py
from __future__ import annotations

import json
from typing import Any, Dict, List

from backend.rag.rag_store import RAGStore
from backend.evaluation.rag_eval import precision_recall_at_k

async def run_eval(store: RAGStore, dataset_path: str, namespace: str, k: int = 5) -> List[Dict[str, Any]]:
    dataset = json.load(open(dataset_path, encoding="utf-8"))
    results = []

    for item in dataset:
        q = item["question"]
        patterns = item["relevant_chunks"]
        chunks = await store.query_semantic(namespace, q, top_k=max(k, 8))
        texts = [c.text for c in chunks]
        score = precision_recall_at_k(texts, patterns, k)
        score["question"] = q
        score["namespace"] = namespace
        results.append(score)

    return results
