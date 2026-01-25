"""
image_understanding.py

สรุป:
ไฟล์นี้เป็นโมดูล “วิเคราะห์ความหมายของรูป” แบบแยกส่วน (run แยกได้ / import ไปใช้ใน main.py ได้)

ความสามารถหลัก:
- รับโฟลเดอร์รูป images_dir (jpg/png/webp)
- เรียก Ollama Vision model (เช่น MiniCPM-V 2.6) เพื่อ:
  1) บรรยายภาพ (caption)
  2) จัดประเภทภาพเชิงธุรกิจ (label/tags)
  3) ประเมินว่า “เกี่ยวข้องกับ context” ไหม (relevance_score 0..1)
  4) ตรวจว่ารูปนี้ควร “เก็บเข้าระบบ RAG” ไหม (keep_for_rag True/False)

- เขียนผลเป็น JSONL: out_jsonl (1 บรรทัดต่อ 1 รูป)

รองรับ 2 แบบบริบท (context):
1) context_text: ข้อความก้อนเดียว (เช่น content.txt ของเว็บ)
2) context_map: dict mapping {image_path: context_string} (เหมาะกับรูปจาก PDF ที่มี caption_hint ต่อรูป)

หมายเหตุสำคัญ:
- ถ้า Ollama ไม่พร้อม / โมเดลไม่รองรับ vision -> จะคืนค่า caption/labels ว่าง และ relevance_score=0
- ใช้ TF-IDF + cosine similarity เป็น baseline เพื่อทำ relevance แบบเบาๆ ไม่ต้อง embedding model

รันเดี่ยว:
python image_understanding.py --images_dir "path/to/images" --out "images_understanding.jsonl"

ENV:
- OLLAMA_HOST (default http://localhost:11434)
- OLLAMA_VISION_MODEL (default minicpm-v:2.6)
"""

from __future__ import annotations

import os
import io
import re
import json
import time
import base64
import hashlib
import argparse
from typing import Dict, Any, Optional, List, Tuple

import requests
import unicodedata

from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Config
# ----------------------------

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp")

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ----------------------------
# Helpers: filesystem / hashing / jsonl
# ----------------------------

def sha1_bytes(b: bytes) -> str:
    """ทำ SHA1 ของ bytes (ใช้ dedupe รูปเหมือนกัน)"""
    return hashlib.sha1(b).hexdigest()


def safe_mkdir(p: str) -> None:
    """สร้างโฟลเดอร์ถ้ายังไม่มี"""
    os.makedirs(p, exist_ok=True)


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    """append 1 record ลง jsonl"""
    safe_mkdir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def list_images(images_dir: str) -> List[str]:
    """ลิสต์ไฟล์รูปทั้งหมดใต้โฟลเดอร์ (recursive)"""
    out: List[str] = []
    if not images_dir or not os.path.isdir(images_dir):
        return out
    for root, _, files in os.walk(images_dir):
        for fn in files:
            if fn.lower().endswith(SUPPORTED_EXTS):
                out.append(os.path.join(root, fn))
    return sorted(out)


# ----------------------------
# Helpers: text normalization + relevance baseline
# ----------------------------

def normalize_text(s: str) -> str:
    """ทำความสะอาดข้อความเบาๆ ให้เหมาะกับ similarity"""
    s = s or ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[^\S\r\n]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def compute_relevance_score(context_text: str, caption: str) -> float:
    """
    ประเมินความเกี่ยวข้อง caption กับ context ด้วย TF-IDF cosine similarity
    คืนค่า 0..1
    """
    if not context_text or not caption:
        return 0.0

    t1 = normalize_text(context_text)[:6000]
    t2 = normalize_text(caption)[:600]

    try:
        vec = TfidfVectorizer(stop_words=None)
        X = vec.fit_transform([t1, t2])
        score = float(cosine_similarity(X[0], X[1])[0][0])
        if score < 0:
            return 0.0
        if score > 1:
            return 1.0
        return score
    except Exception:
        return 0.0


# ----------------------------
# Ollama Vision call
# ----------------------------

def call_ollama_vision(
    image_path: str,
    prompt: str,
    ollama_host: str,
    model: str,
    timeout_s: int = 90
) -> str:
    """
    เรียก Ollama vision: POST {ollama_host}/api/generate
    ส่ง images เป็น base64 ใน field "images"
    คืน response text (string) ถ้า fail -> ""
    """
    try:
        with open(image_path, "rb") as f:
            b = f.read()

        img_b64 = base64.b64encode(b).decode("utf-8")
        payload = {
            "model": model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
        }

        r = requests.post(
            f"{ollama_host}/api/generate",
            json=payload,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout_s,
        )
        if r.status_code != 200:
            return ""

        data = r.json()
        return (data.get("response") or "").strip()
    except Exception:
        return ""


def build_prompt(context: str) -> str:
    """
    prompt สำหรับให้โมเดลส่งออกเป็น JSON เท่านั้น
    """
    context = normalize_text(context)[:1200]  # จำกัดความยาว context ที่ส่งไป
    return (
        "คุณเป็นผู้ช่วยวิเคราะห์รูปภาพสำหรับงานสรุปข้อมูลเชิงธุรกิจ\n"
        "ตอบกลับเป็น JSON เท่านั้น (ห้ามมีข้อความอื่น)\n"
        "สคีมา JSON:\n"
        "{\n"
        '  "caption_th": "บรรยายภาพสั้นๆ ภาษาไทย",\n'
        '  "caption_en": "short English caption",\n'
        '  "labels": ["tag1","tag2","tag3"],\n'
        '  "image_type": "chart|table|document|logo|product|people|scene|infographic|other",\n'
        '  "has_readable_text": true|false,\n'
        '  "business_value": "high|medium|low",\n'
        '  "notes": "ข้อสังเกตเพิ่มเติมสั้นๆ"\n'
        "}\n\n"
        "ข้อกำหนด:\n"
        "- labels ให้เป็นคำสั้นๆ (ไทยหรืออังกฤษก็ได้) ไม่เกิน 8 คำ\n"
        "- ถ้าเป็นกราฟ ให้ระบุว่าเป็นกราฟอะไร (line/bar/pie ฯลฯ) และหัวข้อโดยรวมใน caption_th\n"
        "- has_readable_text: true ถ้ามีข้อความในภาพที่น่าจะอ่านได้ (เช่น สไลด์/เอกสาร/กราฟมีตัวหนังสือ)\n\n"
        "บริบท (context) ที่อาจเกี่ยวข้องกับรูป (ถ้ามี):\n"
        f"{context if context else '(ไม่มี)'}\n"
    )


def parse_json_from_model(text: str) -> Dict[str, Any]:
    """
    พยายาม parse JSON จาก output ของโมเดลแบบทนทาน:
    - ถ้ามี text ครอบๆ จะพยายามหา {...} ก้อนแรก
    - ถ้า parse ไม่ได้ -> {}
    """
    if not text:
        return {}
    t = text.strip()

    # ตัดโค้ดบล็อกถ้ามี
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    # ถ้าไม่ใช่ JSON ตรงๆ ให้หา {...}
    if not (t.startswith("{") and t.endswith("}")):
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if m:
            t = m.group(0)

    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
        return {}
    except Exception:
        return {}


# ----------------------------
# Public API
# ----------------------------

def understand_images(
    images_dir: str,
    out_jsonl: str,
    context_text: str = "",
    context_map: Optional[Dict[str, str]] = None,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
    model: str = DEFAULT_VISION_MODEL,
    max_images: int = 10_000,
    sleep_s: float = 0.0,
    dedupe: bool = True,
) -> Dict[str, Any]:
    """
    วิเคราะห์รูปทั้งหมดใน images_dir แล้วเขียนผลเป็น JSONL

    Params:
    - images_dir: โฟลเดอร์รูป
    - out_jsonl: path ไฟล์ผลลัพธ์ jsonl
    - context_text: context กลาง (เช่น content.txt)
    - context_map: context รายรูป {image_path: hint} (override context_text)
    - ollama_host/model: config สำหรับ vision
    - max_images: จำกัดจำนวนรูป
    - sleep_s: หน่วงระหว่างรูป (กัน rate)
    - dedupe: กันรูปซ้ำด้วย sha1

    Return:
    - dict summary {total, processed, kept_for_rag, out_jsonl}
    """
    imgs = list_images(images_dir)[:max_images]
    processed = 0
    kept = 0
    seen_hash = set()

    for ipath in imgs:
        # อ่าน bytes เพื่อ hash + validate
        try:
            with open(ipath, "rb") as f:
                b = f.read()
        except Exception:
            continue

        h = sha1_bytes(b)
        if dedupe and h in seen_hash:
            continue
        seen_hash.add(h)

        # ตรวจเปิดรูปได้จริง (กันไฟล์เสีย)
        try:
            Image.open(io.BytesIO(b)).verify()
        except Exception:
            continue

        ctx = context_text or ""
        if context_map and ipath in context_map:
            ctx = context_map.get(ipath, "") or ""

        prompt = build_prompt(ctx)
        raw = call_ollama_vision(
            image_path=ipath,
            prompt=prompt,
            ollama_host=ollama_host,
            model=model,
            timeout_s=90,
        )
        obj = parse_json_from_model(raw)

        caption_th = (obj.get("caption_th") or "").strip()
        caption_en = (obj.get("caption_en") or "").strip()
        labels = obj.get("labels") or []
        if not isinstance(labels, list):
            labels = []

        image_type = (obj.get("image_type") or "other").strip()
        has_text = bool(obj.get("has_readable_text")) if "has_readable_text" in obj else False
        biz_value = (obj.get("business_value") or "low").strip()
        notes = (obj.get("notes") or "").strip()

        # ทำ relevance จาก caption_th ถ้าไม่มีใช้ caption_en
        caption_for_rel = caption_th if caption_th else caption_en
        score = compute_relevance_score(ctx, caption_for_rel)

        # rule เก็บเข้า RAG (ปรับได้)
        # - ถ้า relevance สูงพอ
        # - หรือเป็น chart/table/document/infographic
        keep_for_rag = (score >= 0.12) or (image_type in ("chart", "table", "document", "infographic"))

        if keep_for_rag:
            kept += 1

        append_jsonl(out_jsonl, {
            "type": "image_understanding",
            "image_path": ipath,
            "sha1": h,
            "image_type": image_type,
            "labels": labels[:8],
            "caption_th": caption_th,
            "caption_en": caption_en,
            "has_readable_text": has_text,
            "business_value": biz_value,
            "notes": notes,
            "context_used": normalize_text(ctx)[:800],
            "relevance_score": score,
            "keep_for_rag": keep_for_rag,
            "model": model,
            "ollama_host": ollama_host,
        })

        processed += 1
        if sleep_s > 0:
            time.sleep(sleep_s)

    return {
        "total": len(imgs),
        "processed": processed,
        "kept_for_rag": kept,
        "out_jsonl": out_jsonl,
    }


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Image Understanding via Ollama Vision -> JSONL")
    ap.add_argument("--images_dir", required=True, help="Folder containing images")
    ap.add_argument("--out", required=True, help="Output JSONL path")

    ap.add_argument("--context_file", default="", help="Optional text file used as context (e.g., content.txt)")
    ap.add_argument("--ollama_host", default=DEFAULT_OLLAMA_HOST)
    ap.add_argument("--model", default=DEFAULT_VISION_MODEL)

    ap.add_argument("--max_images", type=int, default=10_000)
    ap.add_argument("--sleep_s", type=float, default=0.0)
    ap.add_argument("--no_dedupe", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    ctx = ""
    if args.context_file and os.path.exists(args.context_file):
        try:
            with open(args.context_file, "r", encoding="utf-8") as f:
                ctx = f.read()
        except Exception:
            ctx = ""

    summary = understand_images(
        images_dir=args.images_dir,
        out_jsonl=args.out,
        context_text=ctx,
        context_map=None,
        ollama_host=args.ollama_host,
        model=args.model,
        max_images=args.max_images,
        sleep_s=args.sleep_s,
        dedupe=(not args.no_dedupe),
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
