# backend/agent/agent_flow.py
from __future__ import annotations

import re
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
        "เท่าไหร่", "กี่คน", "ตัวเลข", "เปอร์เซ็นต์", "%",
        "มีอะไรบ้าง", "อะไรบ้าง", "what are the", "which are the", "list"
    ]
    if any(s in q for s in structured_signals):
        return "structured_rag"
    return "semantic_rag"


def normalize_key_from_question(question: str) -> str:
    q = (question or "").lower()
    if ("มีอะไรบ้าง" in question or "อะไรบ้าง" in question) and ("เทรนด์" in question or "trend" in q):
        return "numbered_section_title"
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
    if ("8" in q or "แปด" in th) and ("เทรนด์" in th or "trend" in q):
        return "marketingoops.com"
    return ""


def _normalize_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def extract_query_phrases(question: str) -> List[str]:
    raw = question or ""
    th = raw
    for noise in [
        "\u0e43\u0e19\u0e1a\u0e17\u0e04\u0e27\u0e32\u0e21",
        "\u0e1a\u0e17\u0e04\u0e27\u0e32\u0e21\u0e19\u0e35\u0e49",
        "\u0e02\u0e48\u0e32\u0e27\u0e19\u0e35\u0e49",
        "\u0e2b\u0e19\u0e49\u0e32\u0e19\u0e35\u0e49",
        "\u0e04\u0e37\u0e2d\u0e2d\u0e30\u0e44\u0e23",
        "\u0e40\u0e17\u0e48\u0e32\u0e44\u0e23",
        "\u0e40\u0e17\u0e48\u0e32\u0e44\u0e2b\u0e23\u0e48",
        "\u0e16\u0e39\u0e01\u0e04\u0e32\u0e14\u0e27\u0e48\u0e32",
        "\u0e15\u0e31\u0e27\u0e40\u0e25\u0e02",
        "\u0e0a\u0e48\u0e27\u0e07\u0e40\u0e27\u0e25\u0e32",
        "\u0e2d\u0e30\u0e44\u0e23",
        "\u0e41\u0e25\u0e30",
        "\u0e2b\u0e23\u0e37\u0e2d",
        "\u0e08\u0e32\u0e01",
    ]:
        th = th.replace(noise, " ")

    phrases: List[str] = []
    seen = set()

    for m in re.finditer(r"[A-Za-z][A-Za-z0-9&+./:-]*(?:\s+[A-Za-z][A-Za-z0-9&+./:-]*)+", raw):
        phrase = _normalize_match_text(m.group(0))
        if len(phrase) >= 4 and phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)

    for m in re.finditer(r"[ก-๙]{4,}", th):
        phrase = m.group(0).strip()
        if 4 <= len(phrase) <= 24 and phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)

    q_norm = _normalize_match_text(raw)
    synonym_groups = [
        (
            ["สูงวัย", "ผู้สูงอายุ", "elderly", "senior"],
            ["ผู้สูงอายุ", "สังคมผู้สูงอายุ", "สูงวัย"],
        ),
        (
            ["pet parent", "สัตว์เลี้ยง", "pet"],
            ["pet parent", "ตลาดสัตว์เลี้ยง", "สัตว์เลี้ยง"],
        ),
    ]
    for triggers, expansions in synonym_groups:
        if any(trigger in q_norm for trigger in triggers):
            for phrase in expansions:
                phrase_norm = _normalize_match_text(phrase)
                if phrase_norm not in seen:
                    seen.add(phrase_norm)
                    phrases.append(phrase_norm)

    return phrases


def question_prefers_single_article(question: str) -> bool:
    th = question or ""
    return any(
        k in th for k in [
            "\u0e1a\u0e17\u0e04\u0e27\u0e32\u0e21",
            "\u0e1a\u0e17\u0e04\u0e27\u0e32\u0e21\u0e19\u0e35\u0e49",
            "\u0e02\u0e48\u0e32\u0e27\u0e19\u0e35\u0e49",
            "\u0e2b\u0e19\u0e49\u0e32\u0e19\u0e35\u0e49",
            "\u0e43\u0e19\u0e1a\u0e17\u0e04\u0e27\u0e32\u0e21",
            "\u0e43\u0e19\u0e02\u0e48\u0e32\u0e27",
        ]
    )


def chunk_source_key(chunk: RetrievedChunk) -> str:
    meta = chunk.meta or {}
    return (
        meta.get("source_url")
        or meta.get("source_path")
        or f"{meta.get('source_type', 'unknown')}:{meta.get('page', 0)}"
    )


def rerank_chunks(question: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    if not chunks:
        return []

    phrases = extract_query_phrases(question)
    prefer_single = question_prefers_single_article(question)
    scored_rows = []
    source_stats: Dict[str, Dict[str, float]] = {}

    for chunk in chunks:
        text_norm = _normalize_match_text(chunk.text)
        source_key = chunk_source_key(chunk)
        source_type = (chunk.meta or {}).get("source_type", "")
        phrase_score = 0.0
        matches = 0

        for phrase in phrases:
            if phrase and phrase in text_norm:
                matches += 1
                phrase_score += 0.14 if (" " in phrase or len(phrase) >= 8) else 0.06

        scored_rows.append((chunk, source_key, source_type, phrase_score, matches))

        stats = source_stats.setdefault(source_key, {"score": 0.0, "matches": 0.0, "is_web_page": 0.0})
        stats["score"] += float(chunk.score or 0.0) + phrase_score
        stats["matches"] += matches
        if source_type == "web_page":
            stats["is_web_page"] = 1.0

    if prefer_single:
        web_rows = [row for row in scored_rows if row[2] == "web_page"]
        if web_rows:
            scored_rows = web_rows

        best_source = max(
            scored_rows,
            key=lambda row: (
                source_stats[row[1]]["matches"],
                source_stats[row[1]]["is_web_page"],
                source_stats[row[1]]["score"],
            ),
        )[1]
        scored_rows = [row for row in scored_rows if row[1] == best_source]

    ranked = []
    for chunk, source_key, source_type, phrase_score, matches in scored_rows:
        source_bonus = 0.03 * min(source_stats[source_key]["matches"], 3.0)
        if prefer_single and source_type == "web_page":
            source_bonus += 0.02
        combined = float(chunk.score or 0.0) + phrase_score + source_bonus
        ranked.append((combined, phrase_score, matches, float(chunk.score or 0.0), chunk))

    ranked.sort(key=lambda row: (row[0], row[1], row[2], row[3]), reverse=True)
    return [row[4] for row in ranked]


def dedupe_chunks(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    out: List[RetrievedChunk] = []
    seen = set()
    for chunk in chunks:
        meta = chunk.meta or {}
        key = (
            meta.get("namespace", ""),
            meta.get("source_url", ""),
            meta.get("source_path", ""),
            meta.get("page", 0),
            meta.get("chunk_index", 0),
            chunk.text,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out


def _chunk_matches_phrases(chunk: RetrievedChunk, phrases: List[str]) -> bool:
    text_norm = _normalize_match_text(chunk.text)
    return any(phrase in text_norm for phrase in phrases if phrase)


def _chunk_phrase_match_count(chunk: RetrievedChunk, phrases: List[str]) -> int:
    text_norm = _normalize_match_text(chunk.text)
    return sum(1 for phrase in phrases if phrase and phrase in text_norm)


def _question_wants_numeric_detail(question: str) -> bool:
    q = question or ""
    return any(token in q for token in ["ปี", "%", "สัดส่วน", "ตัวเลข", "ช่วงเวลา", "เท่าไร", "เท่าไหร่"])


def _select_focus_chunks(chunks: List[RetrievedChunk], phrases: List[str], window: int = 1) -> List[RetrievedChunk]:
    if not chunks or not phrases:
        return chunks

    indexed = list(chunks)
    matched_positions = [i for i, chunk in enumerate(indexed) if _chunk_matches_phrases(chunk, phrases)]
    if not matched_positions:
        return chunks

    keep = set()
    for pos in matched_positions:
        start = max(0, pos - window)
        end = min(len(indexed), pos + window + 1)
        for i in range(start, end):
            keep.add(i)

    focused = [indexed[i] for i in sorted(keep)]
    return focused or chunks


def is_boilerplate_chunk(chunk: RetrievedChunk) -> bool:
    text = _normalize_match_text(chunk.text)
    boilerplate_signals = [
        "คุกกี้", "cookie", "privacy", "นโยบายความเป็นส่วนตัว",
        "subscribe", "notifications", "sponsored", "undo",
        "อ่านเพิ่มเติม", "powered by", "taboola",
    ]
    if any(signal in text for signal in boilerplate_signals):
        return True

    meta = chunk.meta or {}
    source_type = meta.get("source_type", "")
    chunk_index = int(meta.get("chunk_index", 0) or 0)
    if source_type == "web_page" and chunk_index >= 12 and ("amarin tvundo" in text or "sponsored" in text):
        return True

    return False


def trim_chunk_to_section(chunk: RetrievedChunk, phrases: List[str]) -> RetrievedChunk:
    text = chunk.text or ""
    text_norm = _normalize_match_text(text)
    match_positions = [text_norm.find(phrase) for phrase in phrases if phrase and text_norm.find(phrase) >= 0]
    if not match_positions:
        return chunk

    anchor = min(match_positions)
    starts = [m.start() for m in re.finditer(r"(?<!\d)(\d+)\.", text)]

    start_idx = 0
    for pos in starts:
        if pos <= anchor:
            start_idx = pos
        else:
            break

    end_idx = len(text)
    for pos in starts:
        if pos > anchor:
            end_idx = pos
            break

    trimmed = text[start_idx:end_idx].strip()
    if not trimmed:
        return chunk

    return RetrievedChunk(text=trimmed, score=chunk.score, meta=chunk.meta)


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

    async def _expand_from_primary_source(self, question: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not chunks:
            return chunks

        phrases = extract_query_phrases(question)
        if not phrases:
            return chunks

        primary = chunks[0]
        meta = primary.meta or {}
        namespace = meta.get("namespace", "")
        source_url = meta.get("source_url", "")
        source_path = meta.get("source_path", "")
        if not namespace or (not source_url and not source_path):
            return chunks

        source_chunks = await self.store.get_source_chunks(
            namespace=namespace,
            source_url=source_url,
            source_path=source_path,
            limit=50,
        )
        if not source_chunks:
            return chunks

        source_chunks = [chunk for chunk in source_chunks if not is_boilerplate_chunk(chunk)]
        if not source_chunks:
            return chunks

        boosted = []
        for chunk in source_chunks:
            text_norm = _normalize_match_text(chunk.text)
            if any(phrase in text_norm for phrase in phrases):
                boosted.append(chunk)

        if not boosted:
            if question_prefers_single_article(question):
                return dedupe_chunks(source_chunks[:10])
            return chunks

        focused = _select_focus_chunks(source_chunks, phrases, window=0)
        focused_boosted = [trim_chunk_to_section(chunk, phrases) for chunk in focused if _chunk_matches_phrases(chunk, phrases)]

        if focused_boosted:
            wants_numeric = _question_wants_numeric_detail(question)
            scored = []
            for chunk in focused_boosted:
                text = chunk.text or ""
                match_count = _chunk_phrase_match_count(chunk, phrases)
                numeric_bonus = 0
                if wants_numeric and (re.search(r"\b20\d{2}\b", text) or "%" in text):
                    numeric_bonus = 2
                scored.append((match_count + numeric_bonus, match_count, len(text), chunk))

            best_score = max(row[0] for row in scored)
            best_chunks = [row[3] for row in scored if row[0] == best_score]
            return dedupe_chunks(best_chunks[:2])

        return dedupe_chunks(chunks + boosted)

    async def route(self, question: str) -> str:
        base = heuristic_route(question)
        if base == "structured_rag" and normalize_key_from_question(question):
            return base
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
                source_contains = infer_prefer_domain(question)
                for ns in namespaces_order:
                    facts.extend(await self.store.query_structured(ns, key=key, limit=30, source_contains=source_contains))
                if facts:
                    if key == "numbered_section_title":
                        ordered = sorted(
                            facts,
                            key=lambda f: (
                                int(f.get("year") or 0),
                                str(f.get("value") or ""),
                            ),
                        )
                        lines = []
                        for idx, f in enumerate(ordered[:25], start=1):
                            lines.append(f"{idx}. {f['value']}")
                        return {
                            "route": "structured_rag",
                            "answer": "\n".join(lines),
                            "chunks": [],
                            "tavily_used": False,
                        }
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

        all_chunks = rerank_chunks(question, all_chunks)
        all_chunks = await self._expand_from_primary_source(question, all_chunks)
        all_chunks = rerank_chunks(question, all_chunks)
        all_chunks = all_chunks[:top_k]
        ctx = format_context(all_chunks, limit_chars=8500)
        direct_focus_match = (
            question_prefers_single_article(question)
            and any(_chunk_matches_phrases(chunk, extract_query_phrases(question)) for chunk in all_chunks)
        )

        # ถ้ามี context -> พยายามตอบจาก RAG ก่อนเสมอ
        if ctx.strip():
            try:
                ans = await self.llm.generate(SYNTH_PROMPT.format(question=question, context=ctx))

                # ✅ ถ้าพอแล้ว ให้ตอบเลย (ไม่สนใจ score threshold แข็งๆ)
                if is_context_sufficient(all_chunks, ctx) or direct_focus_match:
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

