# backend/rag/ingest.py
from __future__ import annotations

import os
import re
import asyncio
from typing import Any, Dict, List, Tuple

from backend.utils.text import normalize_text, chunk_text
from backend.utils.jsonl import iter_jsonl
from backend.rag.rag_store import RAGStore
from backend.rag.structured_extractor import extract_facts_llm


# ============================
# Chunking (for embeddings)
# ============================
# nomic-embed-text บางครั้งจะ error ถ้า input ยาวเกิน context -> ต้อง "split" (ไม่ truncate)
EMBED_MAX_CHARS = 5000
EMBED_OVERLAP = 200


def _split_by_chars(text: str, max_chars: int = EMBED_MAX_CHARS, overlap: int = EMBED_OVERLAP) -> List[str]:
    """
    Split แบบไม่ทำข้อมูลหาย: ตัดเป็นช่วง ๆ (มี overlap)
    - ไม่ truncate ทิ้ง
    - ใช้ overlap กันเนื้อหาขาดช่วง (ช่วย semantic retrieval)
    """
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]

    out: List[str] = []
    step = max_chars - max(0, overlap)
    if step <= 0:
        step = max_chars

    i = 0
    while i < len(t):
        out.append(t[i : i + max_chars])
        i += step

    return [x.strip() for x in out if x.strip()]


def _ensure_embed_safe_chunks(chunks: List[str], max_chars: int = EMBED_MAX_CHARS) -> List[str]:
    """
    รับ chunks จาก chunk_text แล้ว “แตกต่อ” เฉพาะตัวที่ใหญ่เกิน max_chars
    เพื่อให้ Ollama embeddings ไม่พัง (ไม่ทำข้อมูลหาย)
    """
    out: List[str] = []
    for c in chunks:
        c = (c or "").strip()
        if not c:
            continue
        if len(c) <= max_chars:
            out.append(c)
        else:
            out.extend(_split_by_chars(c, max_chars=max_chars, overlap=EMBED_OVERLAP))
    return out


# ============================
# Facts extraction (avoid huge context)
# ============================
# แนวคิด:
# - ห้ามส่ง “ก้อนยักษ์” เข้า LLM เพื่อ extract facts
# - ให้เลือก “top chunks” ที่น่ามีตัวเลข/ข้อมูลสำคัญ แล้วค่อย extract ทีละ chunk
FACTS_TOP_K = 10
FACTS_MAX_CONCURRENCY = 3
FACTS_MIN_CHARS = 120

# กัน LLM input ยาวเกิน (สำหรับ llama3.1:8b)
FACTS_MAX_CHARS_PER_CALL = 4500

_FACT_KEYWORDS = [
    # Thai business/fiscal
    "รายได้", "กำไร", "ขาดทุน", "ยอดขาย", "มูลค่า", "งบ", "สัดส่วน", "เติบโต", "อัตรา", "ร้อยละ", "เปอร์เซ็นต์",
    "ล้าน", "พันล้าน", "บาท", "ปี", "ไตรมาส", "yoy", "qoq",
    # English business/fiscal
    "revenue", "profit", "loss", "sales", "earnings", "growth", "percent", "market", "budget", "cost",
]


def _fact_chunk_score(t: str) -> float:
    """
    ให้คะแนน chunk เพื่อเลือก top chunks ที่เหมาะกับการ extract facts
    heuristic เน้น:
    - มีตัวเลข/ปี/%/เงิน
    - มีคำสำคัญด้านธุรกิจ
    - ความยาวพอเหมาะ (ยาวพอมีสาระ)
    """
    if not t:
        return 0.0
    s = t.lower()
    digits = len(re.findall(r"\d", s))
    perc = len(re.findall(r"%|เปอร์เซ็นต์|ร้อยละ", s))
    money = len(re.findall(r"บาท|ล้าน|พันล้าน|million|billion|usd|thb", s))
    year = len(re.findall(r"\b20\d{2}\b|ปี\s*20\d{2}|พ\.ศ\.\s*\d{4}", s))
    kw = sum(1 for k in _FACT_KEYWORDS if k.lower() in s)

    # base length (cap เพื่อไม่ให้ chunk ยักษ์ชนะเสมอ)
    length_score = min(len(t), 2000) / 2000.0

    return (
        length_score * 2.0
        + digits * 0.15
        + perc * 2.0
        + money * 1.5
        + year * 2.0
        + kw * 1.2
    )


def _select_top_fact_chunks(chunks: List[str], top_k: int = FACTS_TOP_K) -> List[str]:
    """
    เลือก top chunks สำหรับ extract facts
    - ตัด chunk ที่สั้นเกิน
    - เรียงตาม heuristic score
    """
    cleaned: List[str] = []
    for c in chunks:
        c = (c or "").strip()
        if len(c) < FACTS_MIN_CHARS:
            continue
        cleaned.append(c)

    if not cleaned:
        return []

    ranked = sorted(cleaned, key=_fact_chunk_score, reverse=True)
    return ranked[: max(1, int(top_k))]


def _clip_for_facts(text: str, max_chars: int = FACTS_MAX_CHARS_PER_CALL) -> str:
    """
    กัน LLM context overflow:
    - "clip" เฉพาะตอนส่งเข้า LLM (ไม่ได้ clip ตอน ingest embeddings)
    - คลิปนี้ไม่กระทบการเก็บข้อมูลใน RAG chunks เพราะเราเก็บครบแล้ว
    """
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip()


def _dedupe_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    ลบซ้ำ facts แบบอนุรักษ์นิยม:
    ใช้ (entity, key, value, year, unit) เป็นคีย์
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    for f in facts or []:
        entity = str(f.get("entity") or "").strip().lower()
        key = str(f.get("key") or "").strip().lower()
        value = str(f.get("value") or "").strip().lower()
        year = int(f.get("year") or 0)
        unit = str(f.get("unit") or "").strip().lower()
        sig = (entity, key, value, year, unit)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(f)
    return out


async def _extract_facts_from_chunks(
    chunks_for_facts: List[str],
    entity_hint: str,
) -> List[Dict[str, Any]]:
    """
    extract facts ทีละ chunk (หลีกเลี่ยงส่ง text ยาวเกิน)
    ทำแบบ async + จำกัด concurrency
    """
    sem = asyncio.Semaphore(FACTS_MAX_CONCURRENCY)

    async def _one(c: str) -> List[Dict[str, Any]]:
        c = _clip_for_facts(c, FACTS_MAX_CHARS_PER_CALL)
        if len(c) < FACTS_MIN_CHARS:
            return []
        async with sem:
            try:
                return await extract_facts_llm(c, entity_hint=entity_hint)
            except Exception:
                return []

    tasks = [_one(c) for c in (chunks_for_facts or [])]
    if not tasks:
        return []

    results = await asyncio.gather(*tasks)
    merged: List[Dict[str, Any]] = []
    for arr in results:
        if arr:
            merged.extend(arr)

    return _dedupe_facts(merged)


# ============================
# Ingest main functions
# ============================
async def ingest_text_blob(
    store: RAGStore,
    namespace: str,
    text: str,
    meta: Dict[str, Any],
    entity_hint: str,
) -> Dict[str, int]:
    """
    Ingest "text blob" เข้าระบบ RAG:
    1) normalize text
    2) chunk_text -> ensure embed safe (split เพิ่มถ้า chunk ใหญ่เกิน)
    3) upsert chunks + embeddings
    4) extract facts แบบไม่ส่งก้อนยักษ์:
       - เลือก top chunks แล้ว extract ทีละ chunk (หรือถ้าเป็นหน้า pdf ก็ถือว่าเล็กอยู่แล้ว)
    """
    text = normalize_text(text)
    if len(text) < 30:
        return {"chunks": 0, "facts": 0}

    # --------- 1) chunks for embeddings (keep ALL content, no truncate) ---------
    chunks = chunk_text(text)
    chunks = _ensure_embed_safe_chunks(chunks, max_chars=EMBED_MAX_CHARS)

    metas: List[Dict[str, Any]] = []
    total = len(chunks)
    for idx, _ in enumerate(chunks, start=1):
        m = dict(meta)
        m["chunk_index"] = idx
        m["chunk_total"] = total
        metas.append(m)

    added_chunks = await store.upsert_chunks(namespace, chunks, metas)

    # --------- 2) facts extraction (avoid huge context) ---------
    # policy:
    # - pdf_page: โดยธรรมชาติเล็กอยู่แล้ว -> extract จากหน้าเดียวได้
    # - web_page / long blob: extract จาก top chunks
    source_type = (meta.get("source_type") or "").strip().lower()

    facts: List[Dict[str, Any]] = []
    try:
        if source_type in ("pdf_page", "pdf_figure_ocr"):
            # โดยปกติข้อความต่อหน้าไม่ยาวมาก
            clipped = _clip_for_facts(text, FACTS_MAX_CHARS_PER_CALL)
            facts = await extract_facts_llm(clipped, entity_hint=entity_hint)
            facts = _dedupe_facts(facts or [])
        else:
            # web_page, image captions, อื่น ๆ: เลือก top chunks แล้วค่อย extract ทีละ chunk
            # (ใช้ chunks หลัง ensure_embed_safe แล้ว เพื่อไม่ให้บาง chunk ยาวเกิน)
            top_chunks = _select_top_fact_chunks(chunks, top_k=FACTS_TOP_K)
            facts = await _extract_facts_from_chunks(top_chunks, entity_hint=entity_hint)
    except Exception:
        facts = []

    added_facts = 0
    if facts:
        added_facts = await store.upsert_facts(namespace, facts, meta)

    return {"chunks": added_chunks, "facts": added_facts}


async def ingest_site_folder(store: RAGStore, namespace: str, site_folder: str) -> Dict[str, Any]:
    total_chunks = total_facts = skipped = 0

    # ---------------------------
    # Web page content
    # ---------------------------
    content_path = os.path.join(site_folder, "content.txt")
    if os.path.exists(content_path):
        text = open(content_path, "r", encoding="utf-8", errors="ignore").read()
        r = await ingest_text_blob(
            store,
            namespace,
            text,
            {
                "source_type": "web_page",
                "source_path": content_path,
                "source_url": "",
                "page": 0,
            },
            entity_hint="web_page",
        )
        total_chunks += r["chunks"]
        total_facts += r["facts"]

    # ---------------------------
    # Image understanding captions (already short)
    # ---------------------------
    under_path = os.path.join(site_folder, "images_understanding.jsonl")
    for obj in iter_jsonl(under_path) or []:
        if obj.get("type") != "image_understanding":
            continue
        if obj.get("is_duplicate"):
            continue
        if not obj.get("keep_for_rag"):
            continue

        cap_th = (obj.get("caption_th") or "").strip()
        cap_en = (obj.get("caption_en") or "").strip()
        image_type = (obj.get("image_type") or "other").strip()
        rel = float(obj.get("relevance_score") or 0.0)

        text = "\n".join(
            [
                f"caption_th: {cap_th}",
                f"caption_en: {cap_en}",
                f"image_type: {image_type}",
                f"relevance_score: {rel}",
            ]
        ).strip()

        if not text:
            skipped += 1
            continue

        r = await ingest_text_blob(
            store,
            namespace,
            text,
            {
                "source_type": "web_image_caption",
                "source_path": obj.get("image_path") or "",
                "page": 0,
                "relevance_score": rel,
                "image_type": image_type,
            },
            entity_hint="image",
        )
        total_chunks += r["chunks"]
        total_facts += r["facts"]

    return {"site_folder": site_folder, "chunks": total_chunks, "facts": total_facts, "skipped": skipped}


async def ingest_main_folder(store: RAGStore, namespace: str, main_folder: str) -> Dict[str, Any]:
    total_chunks = total_facts = skipped = 0

    # ---------------------------
    # Each site folder
    # ---------------------------
    for name in sorted(os.listdir(main_folder)):
        site_folder = os.path.join(main_folder, name)
        if not os.path.isdir(site_folder):
            continue
        if not (name[:1].isdigit() and "_" in name):
            continue

        r = await ingest_site_folder(store, namespace, site_folder)
        total_chunks += r["chunks"]
        total_facts += r["facts"]
        skipped += r["skipped"]

    # ---------------------------
    # OCR results (per page already)
    # ---------------------------
    ocr_root = os.path.join(main_folder, "ocr_results")
    if os.path.isdir(ocr_root):
        for d in sorted(os.listdir(ocr_root)):
            pdf_dir = os.path.join(ocr_root, d)
            if not os.path.isdir(pdf_dir):
                continue

            docs_jsonl = os.path.join(pdf_dir, "docs.jsonl")
            for obj in iter_jsonl(docs_jsonl) or []:
                if obj.get("type") == "page_text":
                    r = await ingest_text_blob(
                        store,
                        namespace,
                        obj.get("text") or "",
                        {
                            "source_type": "pdf_page",
                            "source_path": obj.get("source_pdf") or "",
                            "page": int(obj.get("page") or 0),
                            "extract_method": obj.get("extract_method") or "",
                        },
                        entity_hint="pdf",
                    )
                    total_chunks += r["chunks"]
                    total_facts += r["facts"]

                elif obj.get("type") == "figure":
                    img_ocr = (obj.get("image_ocr") or "").strip()
                    if len(img_ocr) >= 10:
                        r = await ingest_text_blob(
                            store,
                            namespace,
                            img_ocr,
                            {
                                "source_type": "pdf_figure_ocr",
                                "source_path": obj.get("image_path") or "",
                                "page": int(obj.get("page") or 0),
                            },
                            entity_hint="pdf_figure",
                        )
                        total_chunks += r["chunks"]
                        total_facts += r["facts"]
                    else:
                        skipped += 1

    return {
        "namespace": namespace,
        "folder": main_folder,
        "added_chunks": total_chunks,
        "added_facts": total_facts,
        "skipped": skipped,
    }
