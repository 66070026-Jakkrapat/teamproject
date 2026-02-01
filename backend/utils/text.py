# backend/utils/text.py
from __future__ import annotations

import re
import unicodedata

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_ZW_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")

# พยายามตัดตาม “จบประโยค” ทั้งอังกฤษ/ไทยแบบประมาณการ
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?\u0E2F\u0E46])\s+")


def normalize_text(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ")
    s = _CONTROL_CHARS_RE.sub("", s)
    s = _ZW_RE.sub("", s)
    s = re.sub(r"[^\S\r\n]+", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def contains_thai(s: str) -> bool:
    return any("\u0E00" <= ch <= "\u0E7F" for ch in (s or ""))


def _split_long_block(block: str, max_len: int) -> list[str]:
    """
    แตก block ยาวๆ ให้ไม่เกิน max_len
    ลำดับ:
    1) split ตามประโยค (ถ้าพอช่วยได้)
    2) ถ้ายังมีชิ้นที่ยาวเกิน -> split แบบ sliding window ด้วยตัวอักษร
    """
    block = (block or "").strip()
    if not block:
        return []
    if len(block) <= max_len:
        return [block]

    # 1) พยายามตัดตามประโยค
    sents = [x.strip() for x in _SENT_SPLIT_RE.split(block) if x.strip()]
    if len(sents) <= 1:
        sents = [block]  # ตัดไม่ออก ก็ใช้ทั้งก้อน

    pieces: list[str] = []
    buf = ""
    for s in sents:
        if not s:
            continue
        if len(s) > max_len:
            # 2) ถ้าประโยคเดียวก็ยังยาวเกิน -> sliding window
            if buf:
                pieces.append(buf.strip())
                buf = ""
            step = max(1, max_len)
            i = 0
            while i < len(s):
                pieces.append(s[i:i + max_len].strip())
                i += step
            continue

        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_len:
            buf = (buf + " " + s).strip()
        else:
            pieces.append(buf.strip())
            buf = s

    if buf:
        pieces.append(buf.strip())

    # กันหลุด: ถ้ายังมีอันยาวเกิน (กรณี whitespace แปลกๆ) -> บังคับตัด
    out: list[str] = []
    for p in pieces:
        if len(p) <= max_len:
            out.append(p)
        else:
            i = 0
            while i < len(p):
                out.append(p[i:i + max_len].strip())
                i += max_len
    return [x for x in out if x]


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    """
    ✅ รับประกันว่า "ทุก chunk" จะไม่ยาวเกิน chunk_size
    - แบ่งตามย่อหน้า (\n\n)
    - ถ้าย่อหน้ายาวเกิน chunk_size -> แตกเป็นชิ้นย่อย
    - ทำ overlap ระหว่าง chunk เพื่อไม่เสียบริบท
    """
    text = normalize_text(text)
    if not text:
        return []

    # แยกย่อหน้า
    paras = re.split(r"\n\s*\n", text)

    chunks: list[str] = []
    buf = ""

    for p in paras:
        p = (p or "").strip()
        if not p:
            continue

        # ถ้าย่อหน้าเดียวใหญ่เกิน -> แตกย่อหน้าก่อน
        if len(p) > chunk_size:
            # flush buf ก่อน
            if buf:
                chunks.append(buf.strip())
                buf = ""

            parts = _split_long_block(p, chunk_size)
            chunks.extend(parts)
            continue

        # ย่อหน้าปกติ: pack ลง buf
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= chunk_size:
            buf = (buf + "\n\n" + p).strip()
        else:
            chunks.append(buf.strip())
            buf = p

    if buf:
        chunks.append(buf.strip())

    # Overlap (เหมือนเดิม แต่กันกรณี overlap > len(prev))
    if overlap > 0 and len(chunks) > 1:
        out: list[str] = []
        prev = ""
        for c in chunks:
            if prev:
                ov = prev[-min(overlap, len(prev)):]
                out.append((ov + "\n" + c).strip())
            else:
                out.append(c)
            prev = c
        return out

    return chunks
