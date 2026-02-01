# backend/ocr/ocr_pipeline.py
from __future__ import annotations

import os

# ✅ ต้องอยู่ก่อน "from paddleocr import PaddleOCR"
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "true")

import re
import json
import hashlib
from typing import Any, Dict, List, Optional, Callable, Tuple

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

from backend.utils.text import normalize_text, contains_thai
from backend.utils.jsonl import append_jsonl, safe_mkdir

PRIMARY_LANG = "th"
FALLBACK_LANG = "en"
MIN_TEXT_CHARS = 300
OCR_DPI = 300
MAX_IMAGE_PER_PAGE = 10

ProgressCB = Callable[[Dict[str, Any]], None]

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def render_page_image(page: fitz.Page, dpi: int = OCR_DPI) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def extract_text_layer(page: fitz.Page) -> str:
    return normalize_text(page.get_text("text") or "")

def extract_images_from_page(doc: fitz.Document, page: fitz.Page, out_dir: str, max_images: int = MAX_IMAGE_PER_PAGE) -> List[str]:
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

def paddle_ocr_page(ocr: PaddleOCR, img: Image.Image) -> str:
    arr = np.array(img)
    res = ocr.ocr(arr, cls=True)
    lines: List[str] = []
    for block in (res or []):
        for item in block:
            txt = item[1][0]
            if txt:
                lines.append(txt)
    return normalize_text("\n".join(lines))

def paddle_ocr_image_text_only(ocr: PaddleOCR, img: Image.Image, conf_threshold: float = 0.55) -> str:
    arr = np.array(img)
    res = ocr.ocr(arr, cls=True)
    lines: List[str] = []
    for block in (res or []):
        for item in block:
            txt = item[1][0]
            conf = float(item[1][1])
            if txt and conf >= conf_threshold:
                lines.append(txt)
    return normalize_text("\n".join(lines))

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

    docs_jsonl = os.path.join(out_dir, "docs.jsonl")

    doc = fitz.open(pdf_path)
    pdf_sha1 = sha1_bytes(open(pdf_path, "rb").read())
    total_pages = len(doc)

    for pno in range(total_pages):
        page = doc[pno]
        text_layer = extract_text_layer(page)
        use_ocr = len(text_layer) < MIN_TEXT_CHARS
        prefer_th = contains_thai(text_layer)

        if use_ocr:
            img = render_page_image(page, dpi=OCR_DPI)
            try:
                if prefer_th:
                    page_text = paddle_ocr_page(ocr_th, img)
                    method = "paddleocr_th"
                    if len(page_text) < 50:
                        t2 = paddle_ocr_page(ocr_en, img)
                        if len(t2) > len(page_text):
                            page_text, method = t2, "paddleocr_fallback"
                else:
                    page_text = paddle_ocr_page(ocr_en, img)
                    method = "paddleocr_en"
                    if len(page_text) < 50:
                        t2 = paddle_ocr_page(ocr_th, img)
                        if len(t2) > len(page_text):
                            page_text, method = t2, "paddleocr_fallback"
            except Exception:
                page_text, method = "", "paddleocr_error"
        else:
            page_text, method = text_layer, "text_layer"

        if page_text.strip():
            append_jsonl(docs_jsonl, {
                "type": "page_text",
                "source_pdf": pdf_path,
                "pdf_sha1": pdf_sha1,
                "page": pno + 1,
                "extract_method": method,
                "text": page_text,
            })

        saved_imgs = extract_images_from_page(doc, page, images_dir)
        for ipath in saved_imgs:
            try:
                im = Image.open(ipath).convert("RGB")
            except Exception:
                continue
            try:
                if prefer_th:
                    img_ocr = paddle_ocr_image_text_only(ocr_th, im)
                    if len(img_ocr) < 30:
                        img2 = paddle_ocr_image_text_only(ocr_en, im)
                        if len(img2) > len(img_ocr):
                            img_ocr = img2
                else:
                    img_ocr = paddle_ocr_image_text_only(ocr_en, im)
                    if len(img_ocr) < 30:
                        img2 = paddle_ocr_image_text_only(ocr_th, im)
                        if len(img2) > len(img_ocr):
                            img_ocr = img2
            except Exception:
                img_ocr = ""

            append_jsonl(docs_jsonl, {
                "type": "figure",
                "source_pdf": pdf_path,
                "pdf_sha1": pdf_sha1,
                "page": pno + 1,
                "image_path": ipath,
                "image_ocr": img_ocr,
            })

        if progress_cb:
            progress_cb({"stage": "page_done", "current_page": pno + 1, "total_pages": total_pages})

    doc.close()
    return docs_jsonl

def list_pdf_files(root_dir: str) -> List[str]:
    pdfs: List[str] = []
    for root, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, fn))
    return sorted(pdfs)

def process_folder_pdfs(files_dir: str, out_root: str, progress_cb: Optional[ProgressCB] = None) -> List[str]:
    pdf_paths = list_pdf_files(files_dir)
    total = len(pdf_paths)

    if progress_cb:
        progress_cb({"stage": "start", "total_files": total, "done_files": 0, "percent": 0, "message": "starting ocr"})

    ocr_th = PaddleOCR(use_angle_cls=True, lang=PRIMARY_LANG)
    ocr_en = PaddleOCR(use_angle_cls=True, lang=FALLBACK_LANG)

    outputs: List[str] = []
    for idx, pdf_path in enumerate(pdf_paths, start=1):
        if progress_cb:
            progress_cb({
                "stage": "file_start",
                "total_files": total,
                "done_files": idx - 1,
                "percent": int(((idx - 1) / max(total, 1)) * 100),
                "current_file": pdf_path,
                "message": f"processing {idx}/{total}",
            })

        out_jsonl = process_pdf_to_jsonl(pdf_path, out_root, ocr_th, ocr_en, progress_cb)
        outputs.append(out_jsonl)

        if progress_cb:
            progress_cb({
                "stage": "file_done",
                "total_files": total,
                "done_files": idx,
                "percent": int((idx / max(total, 1)) * 100),
                "current_file": pdf_path,
                "last_output": out_jsonl,
                "message": f"done {idx}/{total}",
            })

    if progress_cb:
        progress_cb({"stage": "done", "total_files": total, "done_files": total, "percent": 100, "message": "all done"})
    return outputs
