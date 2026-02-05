# backend/ocr/ocr_pipeline.py
from __future__ import annotations

import os
import re
import json
import hashlib
from typing import Any, Dict, List, Optional, Callable, Tuple

# ✅ ต้องอยู่ก่อน "from paddleocr import PaddleOCR"
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "true")

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

from backend.utils.text import normalize_text, contains_thai
from backend.utils.jsonl import safe_mkdir

# ============================
# OCR Config (เหมือน ocr__.py)
# ============================
PRIMARY_LANG = "th"
FALLBACK_LANG = "en"

MIN_TEXT_CHARS = 300
OCR_DPI = 300
MAX_IMAGE_PER_PAGE = 10

ProgressCB = Callable[[Dict[str, Any]], None]


# ============================
# Small Utils (ให้เหมือน ocr__.py)
# ============================
def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """เขียน JSONL แบบ append เหมือน ocr__.py"""
    safe_mkdir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm_text(s: str) -> str:
    """normalize แบบเดียวกับ ocr__.py (และใกล้เคียง normalize_text)"""
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def detect_lang_hint(text: str) -> str:
    """เหมือน ocr__.py"""
    if not text:
        return "unknown"
    has_th = contains_thai(text)
    has_en = bool(re.search(r"[A-Za-z]", text))
    if has_th and has_en:
        return "mixed"
    if has_th:
        return "th"
    if has_en:
        return "en"
    return "unknown"


# ============================
# Core OCR Helpers (เหมือน ocr__.py)
# ============================
def render_page_image(page: fitz.Page, dpi: int = OCR_DPI) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def extract_text_layer(page: fitz.Page) -> str:
    # ใช้ normalize_text เดิมของโปรเจกต์ + กันช่องว่างแบบ ocr__.py
    return norm_text(normalize_text(page.get_text("text") or ""))


def paddle_ocr_text(ocr: PaddleOCR, img: Image.Image, min_conf: float = 0.0) -> str:
    """
    OCR image -> text (เหมือน ocr__.py)
    min_conf=0.0 ใช้กับหน้า PDF (เอาทุกบรรทัด)
    min_conf>0 ใช้กับรูป web (กรอง noise)
    """
    arr = np.array(img)
    res = ocr.ocr(arr, cls=True)
    lines: List[str] = []
    for block in (res or []):
        for item in block:
            txt = item[1][0]
            conf = float(item[1][1])
            if txt and conf >= float(min_conf):
                lines.append(txt)
    return norm_text("\n".join(lines))


def extract_images_from_page(
    doc: fitz.Document,
    page: fitz.Page,
    out_dir: str,
    max_images: int = MAX_IMAGE_PER_PAGE,
) -> List[str]:
    safe_mkdir(out_dir)
    saved: List[str] = []
    img_list = page.get_images(full=True) or []
    for i, img_info in enumerate(img_list[:max_images]):
        xref = img_info[0]
        base = doc.extract_image(xref)
        if not base:
            continue

        img_bytes = base.get("image", b"")
        ext = base.get("ext", "png")
        if not img_bytes or len(img_bytes) < 10_000:
            continue

        name = f"p{page.number + 1}_img{i + 1}.{ext}"
        path = os.path.join(out_dir, name)
        with open(path, "wb") as f:
            f.write(img_bytes)
        saved.append(path)

    return saved


def build_page_caption_hint(page_text: str, max_chars: int = 600) -> str:
    t = (page_text or "").strip()
    return t[:max_chars]


# ============================
# PDF -> JSONL (ทำงานเหมือน ocr__.py)
# ============================
def process_pdf_to_jsonl(
    pdf_path: str,
    out_root: str,
    ocr_th: PaddleOCR,
    ocr_en: PaddleOCR,
    progress_cb: Optional[ProgressCB] = None,
) -> str:
    safe_mkdir(out_root)

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = os.path.join(out_root, pdf_name)
    safe_mkdir(out_dir)

    images_dir = os.path.join(out_dir, "images")
    safe_mkdir(images_dir)

    jsonl_path = os.path.join(out_dir, "docs.jsonl")

    doc = fitz.open(pdf_path)
    with open(pdf_path, "rb") as f:
        pdf_sha1 = sha1_bytes(f.read())

    total_pages = len(doc)

    for pno in range(total_pages):
        page = doc[pno]

        text_layer = extract_text_layer(page)
        use_ocr = len(text_layer) < MIN_TEXT_CHARS
        prefer_th = contains_thai(text_layer)

        if use_ocr:
            img = render_page_image(page, dpi=OCR_DPI)
            try:
                page_text = paddle_ocr_text(ocr_th if prefer_th else ocr_en, img, min_conf=0.0)
                method = "paddleocr_th" if prefer_th else "paddleocr_en"

                if len(page_text) < 50:
                    page_text2 = paddle_ocr_text(ocr_en if prefer_th else ocr_th, img, min_conf=0.0)
                    if len(page_text2) > len(page_text):
                        page_text = page_text2
                        method = "paddleocr_fallback"
            except Exception:
                page_text = ""
                method = "paddleocr_error"
        else:
            page_text = text_layer
            method = "text_layer"

        caption_hint = build_page_caption_hint(page_text)

        rows: List[Dict[str, Any]] = []

        if (page_text or "").strip():
            rows.append(
                {
                    "type": "page_text",
                    "source_pdf": pdf_path,
                    "pdf_sha1": pdf_sha1,
                    "page": pno + 1,
                    "extract_method": method,
                    "text": page_text,
                }
            )

        saved_imgs = extract_images_from_page(doc, page, images_dir)
        for ipath in saved_imgs:
            try:
                im = Image.open(ipath).convert("RGB")
            except Exception:
                continue

            try:
                img_ocr = paddle_ocr_text(ocr_th if prefer_th else ocr_en, im, min_conf=0.55)
                if len(img_ocr) < 30:
                    img_ocr2 = paddle_ocr_text(ocr_en if prefer_th else ocr_th, im, min_conf=0.55)
                    if len(img_ocr2) > len(img_ocr):
                        img_ocr = img_ocr2
            except Exception:
                img_ocr = ""

            rows.append(
                {
                    "type": "figure",
                    "source_pdf": pdf_path,
                    "pdf_sha1": pdf_sha1,
                    "page": pno + 1,
                    "image_path": ipath,
                    "image_ocr": img_ocr,
                    "caption_hint": caption_hint,
                }
            )

        if rows:
            save_jsonl(jsonl_path, rows)

        if progress_cb:
            progress_cb({"stage": "page_done", "current_page": pno + 1, "total_pages": total_pages})

    doc.close()
    return jsonl_path


def list_pdf_files(files_dir: str) -> List[str]:
    pdfs: List[str] = []
    for root, _, files in os.walk(files_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, fn))
    return sorted(pdfs)


def process_folder_pdfs(
    files_dir: str,
    out_root: str,
    progress_cb: Optional[ProgressCB] = None,
    ocr_th: Optional[PaddleOCR] = None,  # ✅ เหมือน ocr__.py: รับจากภายนอกได้
    ocr_en: Optional[PaddleOCR] = None,  # ✅
) -> List[str]:
    pdf_paths = list_pdf_files(files_dir)
    total = len(pdf_paths)

    def emit(payload: Dict[str, Any]) -> None:
        if progress_cb:
            progress_cb(payload)

    if total == 0:
        emit(
            {
                "stage": "done",
                "total_files": 0,
                "done_files": 0,
                "percent": 100,
                "current_file": "",
                "message": "no pdf found",
            }
        )
        return []

    # ✅ ถ้าไม่ได้ส่งโมเดลมา ค่อย init เอง
    if ocr_th is None or ocr_en is None:
        ocr_th = PaddleOCR(use_angle_cls=True, lang=PRIMARY_LANG)
        ocr_en = PaddleOCR(use_angle_cls=True, lang=FALLBACK_LANG)

    outputs: List[str] = []
    emit({"stage": "start", "total_files": total, "done_files": 0, "percent": 0, "current_file": ""})

    for idx, pdf_path in enumerate(pdf_paths, start=1):
        emit(
            {
                "stage": "file_start",
                "total_files": total,
                "done_files": idx - 1,
                "percent": int(((idx - 1) / total) * 100),
                "current_file": pdf_path,
                "current_index": idx,
            }
        )

        out_jsonl = process_pdf_to_jsonl(pdf_path, out_root, ocr_th, ocr_en, progress_cb=progress_cb)
        outputs.append(out_jsonl)

        emit(
            {
                "stage": "file_done",
                "total_files": total,
                "done_files": idx,
                "percent": int((idx / total) * 100),
                "current_file": pdf_path,
                "current_index": idx,
                "last_output": out_jsonl,
            }
        )

    emit({"stage": "done", "total_files": total, "done_files": total, "percent": 100, "current_file": ""})
    return outputs


# ============================
# Web Image OCR Helpers (เหมือน ocr__.py)
# ============================
def init_ocr_models() -> Tuple[PaddleOCR, PaddleOCR]:
    ocr_th = PaddleOCR(use_angle_cls=True, lang=PRIMARY_LANG)
    ocr_en = PaddleOCR(use_angle_cls=True, lang=FALLBACK_LANG)
    return ocr_th, ocr_en


def ocr_image_auto(
    image_path: str,
    ocr_th: PaddleOCR,
    ocr_en: PaddleOCR,
    hint_text: str = "",
    web_min_conf: float = 0.45,
) -> Dict[str, Any]:
    try:
        im = Image.open(image_path).convert("RGB")
    except Exception:
        return {"ocr_text": "", "extract_method": "image_open_error", "language": "unknown"}

    hint_lang = detect_lang_hint(hint_text)
    prefer_th = hint_lang in ("th", "mixed")

    try:
        text1 = paddle_ocr_text(ocr_th if prefer_th else ocr_en, im, min_conf=web_min_conf)
        method = "paddleocr_th" if prefer_th else "paddleocr_en"

        if len(text1) < 30:
            text2 = paddle_ocr_text(ocr_en if prefer_th else ocr_th, im, min_conf=web_min_conf)
            if len(text2) > len(text1):
                text1 = text2
                method = "paddleocr_fallback"

        lang = detect_lang_hint(text1) if text1 else (hint_lang or "unknown")
        return {"ocr_text": text1, "extract_method": method, "language": lang}
    except Exception:
        return {"ocr_text": "", "extract_method": "paddleocr_error", "language": hint_lang or "unknown"}
