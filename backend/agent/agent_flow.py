# backend/agent/agent_flow.py
from __future__ import annotations

from typing import Any, Dict, List

import httpx

from backend.settings import settings
from backend.rag.rag_store import RAGStore, RetrievedChunk
from backend.agent.prompts import ROUTER_PROMPT, SYNTH_PROMPT, TAVILY_PROMPT
from backend.agent.tavily_client import tavily_search


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


def heuristic_route(question: str) -> str:
    q = (question or "").lower()
    structured_signals = [
        "จำนวนพนักงาน", "พนักงาน", "employees", "employee",
        "รายได้", "revenue", "กำไร", "profit",
        "สินทรัพย์", "assets", "หนี้สิน", "liabilities",
        "เท่าไหร่", "กี่คน", "ตัวเลข", "เปอร์เซ็นต์", "%"
    ]
    if any(s in q for s in structured_signals):
        return "structured_rag"
    return "semantic_rag"


def normalize_key_from_question(question: str) -> str:
    q = (question or "").lower()
    if any(k in q for k in ["พนักงาน", "employees", "employee"]):
        return "employees"
    if any(k in q for k in ["รายได้", "revenue"]):
        return "revenue"
    if any(k in q for k in ["กำไร", "profit", "net income", "net_income"]):
        return "profit"
    if any(k in q for k in ["สินทรัพย์", "assets"]):
        return "assets"
    if any(k in q for k in ["หนี้สิน", "liabilities"]):
        return "liabilities"
    return ""


def format_context(chunks: List[RetrievedChunk], limit_chars: int = 8000) -> str:
    parts: List[str] = []
    total = 0
    for c in chunks:
        m = c.meta or {}
        src = m.get("source_url") or m.get("source_path") or m.get("source_type") or "unknown"
        ns = m.get("namespace", "")
        page = m.get("page")
        if page:
            src = f"{src} (page {page})"
        block = f"SOURCE: [{ns}] {src}\nTEXT:\n{c.text}".strip()
        if total + len(block) > limit_chars:
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n---\n\n".join(parts)


def snippets_to_text(results: list[dict], max_chars: int = 7000) -> str:
    out = []
    total = 0
    for r in results[:10]:
        url = r.get("url", "")
        title = r.get("title", "")
        content = (r.get("content", "") or "").strip()
        block = f"- title: {title}\n  url: {url}\n  content: {content}".strip()
        if total + len(block) > max_chars:
            break
        out.append(block)
        total += len(block) + 1
    return "\n\n".join(out)


class AgenticRAG:
    def __init__(self, store: RAGStore):
        self.store = store
        self.llm = OllamaLLM(settings.OLLAMA_HOST, settings.OLLAMA_LLM_MODEL)

    async def route(self, question: str) -> str:
        base = heuristic_route(question)
        try:
            out = (await self.llm.generate(ROUTER_PROMPT.format(question=question))).strip().lower()
            if "structured" in out:
                return "structured_rag"
            if "semantic" in out:
                return "semantic_rag"
            return base
        except Exception:
            return base

    async def answer(self, question: str, top_k: int = 8) -> Dict[str, Any]:
        route = await self.route(question)
        namespaces_order = ["internal", "external"] if settings.ROUTE_PREFER_INTERNAL else ["external", "internal"]

        if route == "structured_rag":
            key = normalize_key_from_question(question)
            if key:
                facts = []
                for ns in namespaces_order:
                    facts.extend(await self.store.query_structured(ns, key=key, limit=30))
                if facts:
                    lines = ["คำตอบจาก Structured RAG (ข้อเท็จจริง):\n"]
                    for f in facts[:25]:
                        src = f"{f.get('source_path','')} (page {f.get('page',0)})" if f.get("page") else f.get("source_path", "")
                        lines.append(
                            f"- [{f['namespace']}] {f['entity']} | {f['key']} = {f['value']} {f.get('unit','')} (year={f.get('year',0)}) [source: {src}]"
                        )
                    return {"route": "structured_rag", "answer": "\n".join(lines), "chunks": [], "tavily_used": False}
            route = "semantic_rag"

        all_chunks: List[RetrievedChunk] = []
        for ns in namespaces_order:
            chunks = await self.store.query_semantic(ns, question, top_k=top_k)
            all_chunks.extend(chunks)

        all_chunks = sorted(all_chunks, key=lambda x: x.score, reverse=True)[:top_k]
        ctx = format_context(all_chunks, limit_chars=8500)

        if ctx.strip():
            try:
                ans = await self.llm.generate(SYNTH_PROMPT.format(question=question, context=ctx))
                low = (len(all_chunks) == 0) or (all_chunks[0].score < 0.12)
                if low:
                    return await self._tavily_fallback(question, all_chunks)
                return {"route": "semantic_rag", "answer": ans, "chunks": [c.meta for c in all_chunks], "tavily_used": False}
            except Exception:
                return await self._tavily_fallback(question, all_chunks)

        return await self._tavily_fallback(question, all_chunks)

    async def _tavily_fallback(self, question: str, chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        t = tavily_search(question, max_results=settings.TAVILY_MAX_RESULTS)
        if not t.get("ok"):
            return {
                "route": "tavily_fallback",
                "answer": f"ไม่พบข้อมูลเพียงพอใน RAG และ Tavily ใช้งานไม่ได้: {t.get('error')}",
                "chunks": [c.meta for c in chunks],
                "tavily_used": True
            }

        results = t.get("results", [])
        snippets = snippets_to_text(results)

        try:
            ans = await self.llm.generate(TAVILY_PROMPT.format(question=question, snippets=snippets))
        except Exception:
            ans = "สรุปไม่ได้ด้วย LLM ในตอนนี้ แต่พบผลค้นหา Tavily:\n\n" + snippets

        return {
            "route": "tavily_fallback",
            "answer": ans,
            "chunks": [c.meta for c in chunks],
            "tavily_used": True,
            "tavily_results": results[: settings.TAVILY_MAX_RESULTS]
        }
