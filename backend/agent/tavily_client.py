# backend/agent/tavily_client.py
from __future__ import annotations

from typing import Any, Dict, List
import requests

from backend.settings import settings

def tavily_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    if not settings.TAVILY_API_KEY:
        return {"ok": False, "error": "TAVILY_API_KEY missing", "results": []}

    payload = {
        "api_key": settings.TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
        "search_depth": "advanced",
    }
    try:
        r = requests.post("https://api.tavily.com/search", json=payload, timeout=30)
        if r.status_code != 200:
            return {"ok": False, "error": f"{r.status_code}: {r.text[:200]}", "results": []}
        data = r.json()
        return {"ok": True, "results": data.get("results", [])}
    except Exception as e:
        return {"ok": False, "error": str(e), "results": []}
