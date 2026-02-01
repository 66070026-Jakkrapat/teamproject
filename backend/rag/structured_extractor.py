# backend/rag/structured_extractor.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import httpx

from backend.settings import settings
from backend.utils.text import normalize_text


def _safe_json_parse(s: str) -> Dict[str, Any]:
    s = (s or "").strip()
    if not s:
        return {}
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    if not (s.startswith("{") and s.endswith("}")):
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            s = m.group(0)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


class OllamaLLM:
    def __init__(self, host: str, model: str, timeout_s: int = 120):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    async def generate(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        timeout = httpx.Timeout(self.timeout_s)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{self.host}/api/generate", json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama generate failed: {r.status_code} {r.text[:200]}")
        data = r.json()
        return (data.get("response") or "").strip()


EXTRACT_PROMPT = """You are an information extraction system.
Extract structured business facts from the given text.

Return STRICT JSON with this schema:
{{
  "entity_hint": "<string>",
  "facts": [
    {{
      "entity": "<string>",
      "key": "<string>",
      "value": "<string>",
      "unit": "<string>",
      "year": <int>,
      "evidence_text": "<string up to 250 chars>"
    }}
  ]
}}

Rules:
- Only output JSON. No markdown.
- If no facts found, return {{"entity_hint":"...","facts":[]}}
- Prefer Thai keys? NO: use canonical English keys.
- evidence_text must be a direct excerpt from the input (short).

ENTITY HINT:
{entity_hint}

TEXT:
{text}
"""



async def extract_facts_llm(text: str, entity_hint: str = "unknown") -> List[Dict[str, Any]]:
    """
    ✅ async + กัน hallucination:
    - evidence_text ต้องเป็น substring ของ input จริง (ไม่ใช่ทิ้ง fact)
    """
    t = normalize_text(text)
    if len(t) < 50:
        return []

    llm = OllamaLLM(settings.OLLAMA_HOST, settings.OLLAMA_LLM_MODEL)
    prompt = EXTRACT_PROMPT.format(entity_hint=entity_hint, text=t[:6000])

    try:
        out = await llm.generate(prompt)
        obj = _safe_json_parse(out)
        facts = obj.get("facts") if isinstance(obj, dict) else None
        if not isinstance(facts, list):
            return []

        cleaned: List[Dict[str, Any]] = []
        for f in facts:
            if not isinstance(f, dict):
                continue

            key = str(f.get("key", "")).strip()
            val = str(f.get("value", "")).strip()
            if not key or not val:
                continue

            ev = str(f.get("evidence_text") or "")[:250]
            # ✅ evidence ต้องอยู่ใน input จริง
            if ev and ev not in t:
                continue

            cleaned.append({
                "entity": str(f.get("entity") or entity_hint),
                "key": key,
                "value": val,
                "unit": str(f.get("unit") or ""),
                "year": int(f.get("year") or 0),
                "evidence_text": ev,
            })
        return cleaned
    except Exception:
        return []
