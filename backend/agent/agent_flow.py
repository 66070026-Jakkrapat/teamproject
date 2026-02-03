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

def infer_prefer_domain(question: str) -> str:
    """
    บางคำถามควร prefer แหล่งเฉพาะ เพื่อไม่ให้เว็บอื่น (เช่น KTC 12 trends)
    มาบดทับบทความ ttb 5 trends
    """
    q = (question or "").lower()
    th = question or ""
    if ("5" in q or "ห้า" in th) and ("เทรนด์" in th or "trend" in q) and ("sme" in q or "เอสเอ็มอี" in th):
        return "ttbbank.com"
    if "ttb" in q or "finbiz" in q:
        return "ttbbank.com"
    return ""


def is_context_sufficient(chunks: List[RetrievedChunk], ctx: str) -> bool:
    """
    ตัดสินว่าควรตอบจาก RAG ได้แล้วหรือยัง
    - อย่าใช้ score อย่างเดียว เพราะ L2 score มักต่ำ (0.07-0.10)
    """
    if not ctx or not ctx.strip():
        return False
    if not chunks:
        return False

    # heuristic: ถ้ามีข้อความรวมพอสมควร ก็พอตอบสรุปได้
    if len(ctx) >= 900:
        return True

    # ถ้าสั้น แต่มีหลาย chunk ก็ยังพอได้
    if len(chunks) >= 3:
        return True

    # ถ้ามี chunk เดียว ต้องให้ score พอประมาณ (ลด threshold ลง)
    top_score = float(chunks[0].score or 0.0)
    return top_score >= 0.06


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

        # -------- semantic rag --------
        all_chunks: List[RetrievedChunk] = []
        for ns in namespaces_order:
            chunks = await self.store.query_semantic(ns, question, top_k=max(top_k, 12))
            all_chunks.extend(chunks)

        # sort by score desc
        all_chunks = sorted(all_chunks, key=lambda x: float(x.score or 0.0), reverse=True)

        # ✅ prefer domain (กัน KTC/เว็บอื่นกลบ ttb)
        prefer_domain = infer_prefer_domain(question)
        if prefer_domain:
            filtered = [
                c for c in all_chunks
                if prefer_domain in ((c.meta or {}).get("source_url", "") + " " + (c.meta or {}).get("source_path", ""))
            ]
            if filtered:
                all_chunks = filtered

        all_chunks = all_chunks[:top_k]
        ctx = format_context(all_chunks, limit_chars=8500)

        # ถ้ามี context -> พยายามตอบจาก RAG ก่อนเสมอ
        if ctx.strip():
            try:
                ans = await self.llm.generate(SYNTH_PROMPT.format(question=question, context=ctx))

                # ✅ ถ้าพอแล้ว ให้ตอบเลย (ไม่สนใจ score threshold แข็งๆ)
                if is_context_sufficient(all_chunks, ctx):
                    return {
                        "route": "semantic_rag",
                        "answer": ans,
                        "chunks": [c.meta for c in all_chunks],
                        "tavily_used": False
                    }

                # ✅ ถ้ายังไม่พอจริงๆ ค่อย fallback (แต่ต้อง handle Tavily missing แบบไม่ล้ม)
                return await self._tavily_fallback(question, all_chunks)

            except Exception:
                # ถ้า LLM พัง ค่อย fallback
                return await self._tavily_fallback(question, all_chunks)

        # ไม่มี context เลย -> fallback
        return await self._tavily_fallback(question, all_chunks)


    async def _tavily_fallback(self, question: str, chunks: List[RetrievedChunk]) -> Dict[str, Any]:
        # ✅ ถ้า Tavily ใช้ไม่ได้ แต่เรายังมี RAG chunks -> สรุปจาก RAG แทน (ไม่ล้ม)
        t = tavily_search(question, max_results=settings.TAVILY_MAX_RESULTS)
        if not t.get("ok"):
            # ถ้ามี chunks ให้ลอง synth จาก RAG อีกครั้งแบบ "best effort"
            ctx = format_context(chunks, limit_chars=8500)
            if ctx.strip():
                try:
                    ans = await self.llm.generate(SYNTH_PROMPT.format(question=question, context=ctx))
                    # ใส่หมายเหตุให้รู้ว่า Tavily ใช้ไม่ได้ แต่เราตอบจาก RAG
                    ans = ans.strip() + f"\n\nหมายเหตุ: Tavily ใช้งานไม่ได้ ({t.get('error')}) จึงตอบจาก RAG เท่าที่มี"
                    return {
                        "route": "semantic_rag",
                        "answer": ans,
                        "chunks": [c.meta for c in chunks],
                        "tavily_used": False
                    }
                except Exception:
                    pass

            return {
                "route": "tavily_fallback",
                "answer": f"ไม่พบข้อมูลเพียงพอใน RAG และ Tavily ใช้งานไม่ได้: {t.get('error')}",
                "chunks": [c.meta for c in chunks],
                "tavily_used": False
            }

        # ✅ Tavily ใช้งานได้ตามปกติ
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

