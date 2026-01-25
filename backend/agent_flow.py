"""
agent_flow.py

Agentic RAG (ReAct-like) แบบครบ:
- Router Agent: vectorstore vs web_search
- Retrieval Node: query RAGStore
- Grader Agent: yes/no ต่อ chunk
- Web Search Agent: Tavily (ถ้ามี key) fallback
- Synthesizer Agent: สรุป + citation + dashboard JSON

LLM Backend:
- ใช้ Ollama /api/generate (default llama3.1:8b)
"""

from __future__ import annotations

import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from rag_store import RAGStore, RetrievedChunk
from dashboard_builder import build_dashboard_json


# ----------------------------
# Ollama LLM Client
# ----------------------------

class OllamaLLM:
    def __init__(self, host: str, model: str, timeout_s: int = 90):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def generate(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        r = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout_s)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama generate failed: {r.status_code} {r.text[:200]}")
        data = r.json()
        return (data.get("response") or "").strip()


# ----------------------------
# Prompts (Router/Grader/Synthesizer)
# ----------------------------

ROUTER_PROMPT = """You are an expert router agent.
Route the user question to one of two destinations:
1) vectorstore: internal documents, specific reports, scraped folders, uploaded files
2) web_search: current events, public news, real-time facts

Return ONLY one token: vectorstore OR web_search

Question:
{question}
"""

GRADER_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question.

Retrieved document:
{document}

User question:
{question}

Return ONLY JSON:
{{"score":"yes"}} or {{"score":"no"}}
"""

SYNTH_PROMPT = """You are an intelligent assistant for question-answering tasks.
Use ONLY the provided context to answer the question.
If context is insufficient, say you don't know.

Rules:
1) Answer comprehensively based ONLY on context.
2) Conflict handling: if conflicting info exists, explicitly describe conflicts and cite both.
3) Always provide citations as [source: ...] per bullet/paragraph.
4) Output in Thai.

Question:
{question}

Context blocks (each has SOURCE + TEXT):
{context}

Now produce:
1) Answer (Thai)
2) Key points (bullet)
3) Sources list (dedup)

Return as plain text (not JSON).
"""

DASHBOARD_PROMPT = """Extract business metrics and insights from the given context for dashboarding.

Return ONLY JSON with schema:
{{
  "title": "string",
  "highlights": ["..."],
  "metrics": [
    {{"name":"Revenue","unit":"THB","points":[{{"x":"YYYY","y":123}}]}},
    ...
  ],
  "tables": [
    {{"name":"Key Figures","columns":["..."],"rows":[["..."],["..."]]}}
  ]
}}

Constraints:
- If you cannot find numeric values, return empty metrics/tables but keep highlights.
- Use Thai labels when possible.

Question:
{question}

Context:
{context}
"""


# ----------------------------
# Web Search (Tavily optional)
# ----------------------------

def web_search_tavily(query: str, k: int = 5) -> List[Dict[str, Any]]:
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        return []

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=key)
        res = client.search(query=query, search_depth="advanced", max_results=k)
        # Normalize
        out = []
        for item in res.get("results", [])[:k]:
            out.append({
                "title": item.get("title") or "",
                "url": item.get("url") or "",
                "content": item.get("content") or "",
            })
        return out
    except Exception:
        return []


# ----------------------------
# Utilities
# ----------------------------

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

def format_context(chunks: List[RetrievedChunk], limit_chars: int = 7000) -> str:
    """
    ทำ context string แบบมี SOURCE + TEXT เพื่อให้ synth cite ได้
    """
    parts: List[str] = []
    total = 0
    for c in chunks:
        m = c.meta or {}
        src = m.get("source_url") or m.get("source_path") or m.get("source_type") or "unknown"
        page = m.get("page")
        if page:
            src = f"{src} (page {page})"
        block = f"SOURCE: {src}\nTEXT:\n{c.text}".strip()
        if total + len(block) > limit_chars:
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n---\n\n".join(parts)


# ----------------------------
# Main Agent Flow
# ----------------------------

class AgenticRAG:
    def __init__(self, store: RAGStore, llm: OllamaLLM):
        self.store = store
        self.llm = llm

    def route(self, question: str) -> str:
        out = self.llm.generate(ROUTER_PROMPT.format(question=question))
        out = out.strip().lower()
        return "web_search" if "web" in out else "vectorstore"

    def grade_chunks(self, question: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        kept: List[RetrievedChunk] = []
        for ch in chunks:
            prompt = GRADER_PROMPT.format(document=ch.text[:1800], question=question)
            resp = self.llm.generate(prompt)
            obj = _safe_json_parse(resp)
            if (obj.get("score") or "").strip().lower() == "yes":
                kept.append(ch)
        return kept

    def synthesize(self, question: str, chunks: List[RetrievedChunk]) -> str:
        ctx = format_context(chunks)
        return self.llm.generate(SYNTH_PROMPT.format(question=question, context=ctx))

    def dashboard(self, question: str, chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        # ถ้าอยากให้ครบแบบแน่น ใช้ LLM ทำ schema
        ctx = format_context(chunks, limit_chars=8000)
        raw = self.llm.generate(DASHBOARD_PROMPT.format(question=question, context=ctx))
        obj = _safe_json_parse(raw)
        if obj:
            return obj
        # fallback rule-based (กัน model ออกนอก schema)
        return build_dashboard_json(question=question, context=ctx)

    def answer(self, question: str, top_k: int = 10) -> Dict[str, Any]:
        route = self.route(question)

        if route == "vectorstore":
            chunks = self.store.query(question, top_k=top_k)
            graded = self.grade_chunks(question, chunks)
            if graded:
                ans = self.synthesize(question, graded)
                dash = self.dashboard(question, graded)
                return {"route": "vectorstore", "answer": ans, "dashboard": dash, "chunks": [c.meta for c in graded]}
            # fallback -> web_search
            route = "web_search"

        # web search
        results = web_search_tavily(question, k=5)
        if not results:
            # ถ้าไม่มี Tavily key ให้ตอบตาม RAG ที่มี (แม้ graded ไม่ผ่าน) หรือบอกไม่มีข้อมูล
            chunks = self.store.query(question, top_k=top_k)
            ans = self.synthesize(question, chunks) if chunks else "ไม่พบข้อมูลเพียงพอจากทั้ง RAG และ Web Search (ยังไม่ได้ตั้งค่า Tavily)"
            dash = self.dashboard(question, chunks) if chunks else build_dashboard_json(question, "")
            return {"route": "web_search", "answer": ans, "dashboard": dash, "chunks": [c.meta for c in chunks]}

        # convert web results to pseudo chunks
        pseudo: List[RetrievedChunk] = []
        for r in results:
            text = (r.get("content") or "").strip()
            if not text:
                continue
            pseudo.append(RetrievedChunk(
                text=text,
                score=1.0,
                meta={"source_type": "web_search", "source_url": r.get("url") or "", "title": r.get("title") or ""},
            ))

        graded = self.grade_chunks(question, pseudo) or pseudo
        ans = self.synthesize(question, graded)
        dash = self.dashboard(question, graded)
        return {"route": "web_search", "answer": ans, "dashboard": dash, "chunks": [c.meta for c in graded]}
