"""
ocr_pipeline.py

สรุปภาพรวม (Pipeline OCR สำหรับโปรเจค Web Scraping + RAG + AI Agent)

ไฟล์นี้มีหน้าที่:
1) OCR ไฟล์ PDF ที่ดาวน์โหลดมา
   - ถ้า PDF มี text layer เยอะพอ -> ใช้ text layer (เร็ว/แม่น/ไม่เสียเวลา)
   - ถ้า text layer น้อย/เป็นภาพสแกน -> render หน้าเป็นรูป แล้ว OCR ด้วย PaddleOCR
   - ดึงรูปที่ฝังอยู่ใน PDF (กราฟ/ตาราง/รูป) แล้ว OCR “เฉพาะข้อความในรูป” เพิ่ม
   - บันทึกผลลง JSONL: <out_root>/<pdf_name>/docs.jsonl
     โดยแต่ละบรรทัดเป็น chunk พร้อม metadata (หน้า, วิธีดึง, path รูป, ocr ของรูป ฯลฯ)

2) OCR รูปภาพที่ดาวน์โหลดจากเว็บ (jpg/png/webp)
   - OCR ด้วย PaddleOCR (th เป็นหลัก) และมี fallback en ถ้าผลลัพธ์สั้นผิดปกติ
   - บันทึกผลลง JSONL: <out_root>/images.jsonl

3) รองรับ progress callback
   - ทุกฟังก์ชันหลักรับ progress_cb: Callable[[dict], None]
   - จะ emit payload เป็นระยะ (start/file_start/page_done/file_done/done, img_start/img_done/img_finished)

Dependencies:
- pymupdf (fitz)
- pillow
- numpy
- paddleocr

หมายเหตุ:
- ไฟล์นี้ “ไม่ทำ caption/relevance” ของรูปจากเว็บ (อันนั้นอยู่ใน web_scraping.py ที่เรียก Ollama Vision)
- แต่ไฟล์นี้จะ OCR รูปจากเว็บเพื่อเก็บ text จากรูปเข้าระบบ RAG ได้ด้วย
"""

# ----------------------------
# Imports
# ----------------------------

import os  # ใช้จัดการ path และเดินไฟล์ในโฟลเดอร์
import re  # ใช้ทำความสะอาดข้อความด้วย regex
import json  # ใช้เขียน JSONL
import hashlib  # ใช้ทำ sha1 เพื่อตรวจไฟล์/เอกสาร
from typing import List, Dict, Any, Optional, Callable, Tuple  # ใช้ typing ให้ชัดเจน

import fitz  # PyMuPDF สำหรับอ่าน/เรนเดอร์ PDF
from PIL import Image  # Pillow สำหรับอ่าน/บันทึกรูปภาพ
import numpy as np  # ใช้แปลงรูปเป็น array ให้ PaddleOCR
from paddleocr import PaddleOCR  # โมเดล OCR


# ----------------------------
# OCR Config
# ----------------------------

PRIMARY_LANG = "th"  # ภาษา OCR หลัก (ไทย)
FALLBACK_LANG = "en"  # ภาษา OCR สำรอง (อังกฤษ)

MIN_TEXT_CHARS = 300  # ถ้า text layer ในหน้า PDF น้อยกว่าเกณฑ์นี้ -> ถือว่าเป็นสแกน/ควร OCR
OCR_DPI = 300  # DPI สำหรับ render PDF page -> image (ยิ่งสูงยิ่งชัด แต่ช้าลง/กิน RAM)
MAX_IMAGE_PER_PAGE = 10  # จำนวนรูปสูงสุดที่จะ extract จาก 1 หน้า PDF (กันไฟล์หนักเกิน)

# ประเภท callback สำหรับส่ง progress กลับไปให้ฝั่ง API (web_scraping.py)
ProgressCB = Callable[[Dict[str, Any]], None]  # callback(payload: dict) -> None


# ----------------------------
# Helpers: filesystem / hashing / JSONL
# ----------------------------

def safe_mkdir(p: str) -> None:
    """
    '''
    สร้างโฟลเดอร์ให้แน่ใจว่ามีอยู่ (ถ้าไม่มีให้สร้าง)
    - ใช้ exist_ok=True เพื่อไม่ error ถ้ามีอยู่แล้ว
    '''
    """
    os.makedirs(p, exist_ok=True)  # สร้างโฟลเดอร์แบบไม่ error ถ้ามีอยู่แล้ว


def sha1_bytes(b: bytes) -> str:
    """
    '''
    ทำ SHA1 ของ bytes (ใช้ตรวจซ้ำ/ทำ id เอกสาร)
    - คืนค่าเป็น hex string
    '''
    """
    return hashlib.sha1(b).hexdigest()  # คืนค่า sha1 เป็นสตริง


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """
    '''
    เขียนข้อมูลหลายแถวลงไฟล์ JSONL (append)
    - JSONL = 1 บรรทัดต่อ 1 JSON object
    - ใช้ ensure_ascii=False เพื่อเก็บภาษาไทยได้
    '''
    """
    with open(path, "a", encoding="utf-8") as f:  # เปิดไฟล์แบบ append (เพิ่มท้าย)
        for r in rows:  # วนทุกแถว
            f.write(json.dumps(r, ensure_ascii=False) + "\n")  # เขียน JSON 1 บรรทัด


# ----------------------------
# Helpers: text normalization / language hint
# ----------------------------

def norm_text(s: str) -> str:
    """
    '''
    ทำความสะอาดข้อความแบบเบาๆ
    - แทน nbsp
    - ลดช่องว่างซ้ำ
    - ลด newline เยอะๆ
    - trim หัว/ท้าย
    '''
    """
    s = (s or "").replace("\u00a0", " ")  # แทน non-breaking space ให้เป็น space ปกติ
    s = re.sub(r"[ \t]{2,}", " ", s)  # ลดช่องว่าง/แท็บซ้ำให้เหลือ 1
    s = re.sub(r"\n{3,}", "\n\n", s)  # ลดบรรทัดว่างติดกันเยอะๆ ให้เหลือ 2 บรรทัด
    return s.strip()  # ตัดช่องว่างหัวท้าย


def contains_thai(s: str) -> bool:
    """
    '''
    ตรวจว่ามีอักษรไทยในสตริงหรือไม่ (heuristic)
    - ใช้ unicode range: \u0E00 - \u0E7F
    '''
    """
    if not s:  # ถ้าไม่มีข้อความ
        return False  # คืน False
    return any("\u0E00" <= ch <= "\u0E7F" for ch in s)  # ถ้ามีตัวไทยสักตัว -> True


# ----------------------------
# Helpers: PDF render / text layer / image extraction
# ----------------------------

def render_page_image(page: fitz.Page, dpi: int = OCR_DPI) -> Image.Image:
    """
    '''
    เรนเดอร์หน้า PDF เป็นรูปภาพ (PIL Image)
    - ใช้ dpi เพื่อกำหนดความละเอียด
    - alpha=False เพื่อให้ได้ RGB ที่ง่ายต่อ OCR
    '''
    """
    zoom = dpi / 72.0  # แปลง dpi เป็น zoom factor (PDF default 72dpi)
    mat = fitz.Matrix(zoom, zoom)  # สร้าง matrix สำหรับ scale
    pix = page.get_pixmap(matrix=mat, alpha=False)  # render หน้าเป็น pixmap
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)  # แปลงเป็น PIL Image
    return img  # คืนภาพที่เรนเดอร์แล้ว


def extract_text_layer(page: fitz.Page) -> str:
    """
    '''
    ดึง text layer จากหน้า PDF (ถ้ามี)
    - ถ้าเป็น PDF ที่มีข้อความจริง (ไม่ใช่สแกน) จะได้ผลดีและเร็ว
    '''
    """
    return norm_text(page.get_text("text") or "")  # ดึงข้อความและ normalize


def extract_images_from_page(
    doc: fitz.Document,
    page: fitz.Page,
    out_dir: str,
    max_images: int = MAX_IMAGE_PER_PAGE
) -> List[str]:
    """
    '''
    ดึงรูปที่ฝังอยู่ในหน้า PDF (เช่น รูป, กราฟ, ตาราง)
    - เซฟรูปลง out_dir
    - จำกัดจำนวนรูปต่อหน้าเพื่อกันไฟล์บวม
    - คืน list ของ path รูปที่เซฟสำเร็จ
    '''
    """
    safe_mkdir(out_dir)  # สร้างโฟลเดอร์ output รูป
    saved: List[str] = []  # list เก็บ path รูปที่บันทึก
    img_list = page.get_images(full=True) or []  # ดึงรายการรูปในหน้านั้น
    for i, img_info in enumerate(img_list[:max_images]):  # วนรูปตามลำดับ (ตัดที่ max_images)
        xref = img_info[0]  # xref คือ id รูปใน PDF
        base = doc.extract_image(xref)  # ดึง bytes ของรูป
        if not base:  # ถ้าดึงไม่ได้
            continue  # ข้ามรูปนี้
        img_bytes = base.get("image", b"")  # bytes ของรูป
        ext = base.get("ext", "png")  # นามสกุลรูป (เดาจาก PDF)
        if not img_bytes or len(img_bytes) < 10_000:  # ถ้า bytes ว่างหรือเล็กเกินไป
            continue  # ข้าม (มักเป็นรูปเล็ก/ไอคอน)
        name = f"p{page.number + 1}_img{i + 1}.{ext}"  # ตั้งชื่อไฟล์ (ผูกกับหน้าและลำดับ)
        path = os.path.join(out_dir, name)  # path เต็มของไฟล์รูป
        with open(path, "wb") as f:  # เปิดไฟล์เขียนแบบ binary
            f.write(img_bytes)  # เขียน bytes ลงไฟล์
        saved.append(path)  # เก็บ path ที่บันทึกสำเร็จ
    return saved  # คืน list ของรูปที่เซฟได้


def build_page_caption_hint(page_text: str, max_chars: int = 600) -> str:
    """
    '''
    สร้าง “hint” สั้นๆ จากข้อความในหน้า
    - เอาไว้แนบ metadata ของรูปในหน้าเดียวกัน (ช่วย RAG ให้มีบริบท)
    '''
    """
    t = (page_text or "").strip()  # trim
    return t[:max_chars]  # ตัดให้สั้นเพื่อไม่ให้ metadata หนัก


# ----------------------------
# OCR: Paddle helpers
# ----------------------------

def paddle_ocr_page(ocr: PaddleOCR, img: Image.Image) -> str:
    """
    '''
    OCR รูปทั้งหน้า (เหมาะกับ PDF ที่เป็นสแกน)
    - คืนค่าเป็นข้อความหลายบรรทัด
    '''
    """
    arr = np.array(img)  # แปลง PIL -> numpy array
    res = ocr.ocr(arr, cls=True)  # เรียก OCR (cls=True ให้หมุนแก้เอียง)
    lines: List[str] = []  # เก็บข้อความทีละบรรทัด
    for block in (res or []):  # วน block ผล OCR
        for item in block:  # วน item ใน block
            txt = item[1][0]  # ข้อความที่ OCR ได้
            if txt:  # ถ้าไม่ว่าง
                lines.append(txt)  # เพิ่มเข้าลิสต์
    return norm_text("\n".join(lines))  # รวมเป็นข้อความและ normalize


def paddle_ocr_image_text_only(ocr: PaddleOCR, img: Image.Image, conf_threshold: float = 0.55) -> str:
    """
    '''
    OCR รูปภาพ (เน้น “เฉพาะข้อความที่มั่นใจพอ”)
    - ใช้กับรูปกราฟ/ตาราง/รูปจากเว็บ
    - conf_threshold ปรับได้ (ยิ่งสูงยิ่งกรองเข้ม)
    '''
    """
    arr = np.array(img)  # แปลง PIL -> numpy array
    res = ocr.ocr(arr, cls=True)  # เรียก OCR
    lines: List[str] = []  # เก็บข้อความ
    for block in (res or []):  # วนผลลัพธ์
        for item in block:  # วนรายการ
            txt = item[1][0]  # ข้อความ
            conf = float(item[1][1])  # ความมั่นใจ
            if txt and conf >= conf_threshold:  # ถ้าข้อความมีค่า และมั่นใจพอ
                lines.append(txt)  # เก็บ
    return norm_text("\n".join(lines))  # รวมและ normalize


def ocr_with_fallback(
    ocr_primary: PaddleOCR,
    ocr_fallback: PaddleOCR,
    img: Image.Image,
    min_chars: int = 50
) -> Tuple[str, str]:
    """
    '''
    OCR ด้วย primary ก่อน แล้ว fallback ถ้าข้อความสั้นผิดปกติ
    - คืนค่า (text, method)
      method: "primary" | "fallback"
    '''
    """
    text1 = paddle_ocr_page(ocr_primary, img)  # OCR ด้วย primary
    if len(text1) >= min_chars:  # ถ้าผลลัพธ์ยาวพอ
        return text1, "primary"  # ใช้ผล primary
    text2 = paddle_ocr_page(ocr_fallback, img)  # ลอง OCR ด้วย fallback
    if len(text2) > len(text1):  # ถ้า fallback ดีกว่า
        return text2, "fallback"  # ใช้ผล fallback
    return text1, "primary"  # ไม่งั้นใช้ primary เดิม


# ----------------------------
# Core: PDF -> JSONL
# ----------------------------

def process_pdf_to_jsonl(
    pdf_path: str,
    out_root: str,
    ocr_th: PaddleOCR,
    ocr_en: PaddleOCR,
    progress_cb: Optional[ProgressCB] = None,
) -> str:
    """
    '''
    OCR ไฟล์ PDF หนึ่งไฟล์และบันทึกเป็น JSONL

    Output structure:
    - out_dir = <out_root>/<pdf_name>/
      - docs.jsonl
      - images/ (รูปที่ extract จาก PDF)

    JSONL rows:
    1) type="page_text"
       - text ของหน้า (จาก text layer หรือ OCR)
    2) type="figure"
       - รูปที่ฝังอยู่ในหน้า + OCR ของรูป + hint จากข้อความในหน้า

    กลไกเลือกวิธี:
    - ถ้า text layer ยาวพอ -> ใช้ text_layer
    - ถ้า text layer สั้น -> render เป็นภาพแล้ว OCR
    - เลือกภาษา OCR ตามสัญญาณภาษาไทยจาก text layer (ถ้ามี) + fallback อีกภาษา

    progress_cb:
    - จะ emit {"stage":"page_done","current_page":...,"total_pages":...} ทุกหน้า
    '''
    """
    safe_mkdir(out_root)  # สร้างโฟลเดอร์ root output

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]  # ชื่อไฟล์ PDF แบบไม่มีนามสกุล
    out_dir = os.path.join(out_root, pdf_name)  # โฟลเดอร์ output สำหรับ PDF นี้
    safe_mkdir(out_dir)  # สร้างโฟลเดอร์

    images_dir = os.path.join(out_dir, "images")  # โฟลเดอร์เก็บรูปที่ extract จาก PDF
    safe_mkdir(images_dir)  # สร้างโฟลเดอร์รูป

    jsonl_path = os.path.join(out_dir, "docs.jsonl")  # path ไฟล์ jsonl output

    doc = fitz.open(pdf_path)  # เปิด PDF ด้วย PyMuPDF
    with open(pdf_path, "rb") as f:  # เปิดไฟล์ PDF แบบ binary
        pdf_sha1 = sha1_bytes(f.read())  # ทำ sha1 ของทั้งไฟล์

    total_pages = len(doc)  # จำนวนหน้า PDF ทั้งหมด

    for pno in range(total_pages):  # วนทีละหน้า (0-index)
        page = doc[pno]  # ดึง page object

        text_layer = extract_text_layer(page)  # ดึง text layer
        use_ocr = len(text_layer) < MIN_TEXT_CHARS  # ถ้า text layer สั้น -> ควร OCR

        prefer_th = contains_thai(text_layer)  # ถ้ามีไทยใน text layer -> ให้ prefer ไทย

        if use_ocr:  # ถ้าต้อง OCR
            img = render_page_image(page, dpi=OCR_DPI)  # render หน้าเป็นรูป
            try:  # ครอบ try กัน OCR พัง
                if prefer_th:  # ถ้าหน้านี้ดูเหมือนมีไทย
                    page_text, method_flag = ocr_with_fallback(ocr_th, ocr_en, img, min_chars=50)  # OCR ไทยก่อน
                    method = "paddleocr_th" if method_flag == "primary" else "paddleocr_fallback"  # ตั้งชื่อวิธี
                else:  # ถ้าไม่ค่อยมีไทย
                    page_text, method_flag = ocr_with_fallback(ocr_en, ocr_th, img, min_chars=50)  # OCR อังกฤษก่อน
                    method = "paddleocr_en" if method_flag == "primary" else "paddleocr_fallback"  # ตั้งชื่อวิธี
            except Exception:  # ถ้า OCR error
                page_text = ""  # ตั้งค่าว่าง
                method = "paddleocr_error"  # method บอกว่า error
        else:  # ถ้าไม่ต้อง OCR
            page_text = text_layer  # ใช้ text layer
            method = "text_layer"  # method เป็น text_layer

        caption_hint = build_page_caption_hint(page_text)  # hint สั้นๆ จากข้อความหน้า

        rows: List[Dict[str, Any]] = []  # เก็บแถวที่จะเขียนลง JSONL

        if (page_text or "").strip():  # ถ้ามีข้อความหน้า
            rows.append({  # เพิ่ม row สำหรับ page_text
                "type": "page_text",  # ประเภท chunk
                "source_pdf": pdf_path,  # path PDF ต้นทาง
                "pdf_sha1": pdf_sha1,  # sha1 ของไฟล์ PDF
                "page": pno + 1,  # หน้าของ PDF (1-index เพื่อคนอ่านง่าย)
                "extract_method": method,  # วิธีดึงข้อความ
                "text": page_text,  # เนื้อหา
            })  # จบ dict

        saved_imgs = extract_images_from_page(doc, page, images_dir)  # ดึงรูปที่ฝังอยู่ในหน้าและเซฟลงโฟลเดอร์

        for ipath in saved_imgs:  # วนแต่ละรูปที่เซฟได้
            try:
                im = Image.open(ipath).convert("RGB")  # เปิดรูปให้เป็น RGB
            except Exception:
                continue  # ถ้าเปิดไม่ได้ ข้ามรูปนี้

            try:  # OCR รูปด้วยภาษาเดียวกับหน้า + fallback
                if prefer_th:  # ถ้าหน้านี้ดูเหมือนภาษาไทย
                    img_ocr = paddle_ocr_image_text_only(ocr_th, im)  # OCR ไทยก่อน
                    if len(img_ocr) < 30:  # ถ้าสั้นมาก
                        img_ocr2 = paddle_ocr_image_text_only(ocr_en, im)  # ลองอังกฤษ
                        if len(img_ocr2) > len(img_ocr):  # ถ้าดีกว่า
                            img_ocr = img_ocr2  # ใช้อังกฤษแทน
                else:  # ถ้าหน้านี้ดูเหมือนอังกฤษ
                    img_ocr = paddle_ocr_image_text_only(ocr_en, im)  # OCR อังกฤษก่อน
                    if len(img_ocr) < 30:  # ถ้าสั้นมาก
                        img_ocr2 = paddle_ocr_image_text_only(ocr_th, im)  # ลองไทย
                        if len(img_ocr2) > len(img_ocr):  # ถ้าดีกว่า
                            img_ocr = img_ocr2  # ใช้ไทยแทน
            except Exception:
                img_ocr = ""  # ถ้า OCR รูป error -> ว่าง

            rows.append({  # เพิ่ม row สำหรับ figure
                "type": "figure",  # ประเภท chunk
                "source_pdf": pdf_path,  # path PDF
                "pdf_sha1": pdf_sha1,  # sha1 ไฟล์
                "page": pno + 1,  # หน้า
                "image_path": ipath,  # path รูปที่ extract
                "image_ocr": img_ocr,  # ข้อความจาก OCR รูป
                "caption_hint": caption_hint,  # hint บริบทจากข้อความหน้า
            })  # จบ dict

        if rows:  # ถ้ามีแถวอย่างน้อย 1 แถว
            save_jsonl(jsonl_path, rows)  # เขียนลง JSONL

        if progress_cb:  # ถ้ามี callback
            progress_cb({  # ส่ง progress รายหน้า
                "stage": "page_done",  # stage
                "current_page": pno + 1,  # หน้าปัจจุบัน
                "total_pages": total_pages,  # จำนวนหน้าทั้งหมด
            })  # จบ payload

    doc.close()  # ปิดเอกสาร PDF
    return jsonl_path  # คืน path ไฟล์ output JSONL


# ----------------------------
# Folder utilities: list PDFs / images
# ----------------------------

def list_pdf_files(files_dir: str) -> List[str]:
    """
    '''
    เดินหาไฟล์ .pdf ทั้งหมดใต้โฟลเดอร์ files_dir
    - คืนเป็น list ที่ sort แล้ว
    '''
    """
    pdfs: List[str] = []  # list เก็บ path pdf
    for root, _, files in os.walk(files_dir):  # เดินไฟล์ในทุกโฟลเดอร์ย่อย
        for fn in files:  # วนชื่อไฟล์
            if fn.lower().endswith(".pdf"):  # ถ้าลงท้าย .pdf
                pdfs.append(os.path.join(root, fn))  # เก็บ path เต็ม
    return sorted(pdfs)  # คืน list เรียง


def list_image_files(root_dir: str) -> List[str]:
    """
    '''
    เดินหาไฟล์รูป (jpg/jpeg/png/webp) ทั้งหมดใต้โฟลเดอร์ root_dir
    - คืนเป็น list ที่ sort แล้ว
    '''
    """
    exts = (".jpg", ".jpeg", ".png", ".webp")  # นามสกุลรูปที่รองรับ
    paths: List[str] = []  # list เก็บ path รูป
    for root, _, files in os.walk(root_dir):  # เดินทุกโฟลเดอร์ย่อย
        for fn in files:  # วนชื่อไฟล์
            if fn.lower().endswith(exts):  # ถ้าตรงนามสกุลที่รองรับ
                paths.append(os.path.join(root, fn))  # เก็บ path เต็ม
    return sorted(paths)  # คืน list เรียง


# ----------------------------
# Public API: process folder PDFs
# ----------------------------

def process_folder_pdfs(
    files_dir: str,
    out_root: str,
    progress_cb: Optional[ProgressCB] = None
) -> List[str]:
    """
    '''
    OCR ไฟล์ PDF ทั้งหมดใต้ files_dir แล้วบันทึกผลลง out_root

    จะ emit progress (ถ้ามี progress_cb):
    - {"stage":"start", ...}
    - {"stage":"file_start", ...}
    - (page-level) {"stage":"page_done", ...}
    - {"stage":"file_done", ...}
    - {"stage":"done", ...}

    คืนค่า:
    - list ของ path jsonl ที่สร้างได้ (ต่อ 1 PDF = 1 jsonl)
    '''
    """
    pdf_paths = list_pdf_files(files_dir)  # หา pdf ทั้งหมด
    total = len(pdf_paths)  # จำนวน pdf ทั้งหมด

    def emit(payload: Dict[str, Any]) -> None:
        """helper ส่ง progress ถ้ามี callback"""
        if progress_cb:  # ถ้ามี callback
            progress_cb(payload)  # ส่ง payload

    emit({  # แจ้งเริ่มต้น
        "stage": "start",  # stage
        "total_files": total,  # ไฟล์ทั้งหมด
        "done_files": 0,  # ทำเสร็จแล้วกี่ไฟล์
        "percent": 0,  # เปอร์เซ็นต์โดยรวม
        "current_index": 0,  # index ปัจจุบัน
        "current_file": "",  # ไฟล์ปัจจุบัน
        "message": "starting ocr",  # ข้อความ
    })  # จบ payload

    ocr_th = PaddleOCR(use_angle_cls=True, lang=PRIMARY_LANG)  # สร้าง OCR ไทย (reuse ทั้งโฟลเดอร์)
    ocr_en = PaddleOCR(use_angle_cls=True, lang=FALLBACK_LANG)  # สร้าง OCR อังกฤษ (reuse ทั้งโฟลเดอร์)

    outputs: List[str] = []  # เก็บ path output jsonl

    for idx, pdf_path in enumerate(pdf_paths, start=1):  # วนทีละไฟล์ pdf (1-index)
        emit({  # แจ้งเริ่มไฟล์
            "stage": "file_start",  # stage
            "total_files": total,  # จำนวนไฟล์ทั้งหมด
            "done_files": idx - 1,  # เสร็จก่อนหน้านี้
            "percent": int(((idx - 1) / max(total, 1)) * 100),  # % โดยรวมก่อนเริ่มไฟล์นี้
            "current_index": idx,  # ไฟล์ลำดับที่เท่าไร
            "current_file": pdf_path,  # path ไฟล์ที่กำลังทำ
            "current_page": 0,  # page เริ่มต้น
            "total_pages": 0,  # total pages ยังไม่รู้
            "message": f"processing file {idx}/{total}",  # ข้อความ
        })  # จบ payload

        def page_progress(p: Dict[str, Any]) -> None:
            """รับ progress รายหน้าจาก process_pdf_to_jsonl แล้ว map เป็น payload ที่ web_scraping.py คาดหวัง"""
            emit({  # ส่งต่อให้ชั้นบน
                "stage": p.get("stage", "page_done"),  # stage
                "total_files": total,  # total files
                "done_files": idx - 1,  # done files ก่อนหน้า
                "percent": int(((idx - 1) / max(total, 1)) * 100),  # percent โดยรวม (ระดับไฟล์)
                "current_index": idx,  # index ไฟล์
                "current_file": pdf_path,  # ไฟล์ปัจจุบัน
                "current_page": p.get("current_page", 0),  # หน้าปัจจุบัน
                "total_pages": p.get("total_pages", 0),  # จำนวนหน้าทั้งหมด
                "message": f"page {p.get('current_page', 0)}/{p.get('total_pages', 0)}",  # ข้อความ
            })  # จบ payload

        out_jsonl = process_pdf_to_jsonl(  # OCR PDF ไฟล์นี้
            pdf_path=pdf_path,  # path pdf
            out_root=out_root,  # โฟลเดอร์ output
            ocr_th=ocr_th,  # ocr ไทย
            ocr_en=ocr_en,  # ocr อังกฤษ
            progress_cb=page_progress,  # callback รายหน้า
        )  # ได้ path jsonl

        outputs.append(out_jsonl)  # เก็บ output

        emit({  # แจ้งจบไฟล์นี้
            "stage": "file_done",  # stage
            "total_files": total,  # total files
            "done_files": idx,  # ทำเสร็จแล้วกี่ไฟล์
            "percent": int((idx / max(total, 1)) * 100),  # percent รวม
            "current_index": idx,  # index
            "current_file": pdf_path,  # ไฟล์
            "current_page": 0,  # reset
            "total_pages": 0,  # reset
            "last_output": out_jsonl,  # output ล่าสุด
            "message": f"done file {idx}/{total}",  # ข้อความ
        })  # จบ payload

    emit({  # แจ้งจบทั้งหมด
        "stage": "done",  # stage
        "total_files": total,  # total
        "done_files": total,  # done
        "percent": 100,  # 100%
        "current_index": total,  # index สุดท้าย
        "current_file": "",  # ไม่มีไฟล์
        "message": "all done",  # ข้อความ
    })  # จบ payload

    return outputs  # คืน list output jsonl


# ----------------------------
# Public API: process folder images (web downloaded images)
# ----------------------------

def process_folder_images(
    root_dir: str,
    out_root: str,
    progress_cb: Optional[ProgressCB] = None
) -> List[str]:
    """
    '''
    OCR รูปทั้งหมดใต้ root_dir แล้วเขียนผลเป็น JSONL ใน out_root

    Output:
    - <out_root>/images.jsonl
      แต่ละบรรทัด:
      {
        "type": "web_image",
        "image_path": "...",
        "text": "..."
      }

    กลไก OCR:
    - ใช้ OCR ไทยเป็นหลัก
    - ถ้าได้ข้อความสั้นมาก (<30) จะลอง OCR อังกฤษแล้วเลือกอันที่ยาวกว่า

    progress:
    - {"stage":"img_start", ...}
    - {"stage":"img_done", ...}
    - {"stage":"img_finished", ...}

    คืนค่า:
    - list ของ path output jsonl (ปกติจะมี 1 ไฟล์)
    '''
    """
    safe_mkdir(out_root)  # สร้างโฟลเดอร์ output ถ้ายังไม่มี

    img_paths = list_image_files(root_dir)  # หาไฟล์รูปทั้งหมด
    total = len(img_paths)  # จำนวนรูปทั้งหมด
    outputs: List[str] = []  # list เก็บ path output jsonl (สุดท้ายมักมี 1 ไฟล์)

    ocr_th = PaddleOCR(use_angle_cls=True, lang=PRIMARY_LANG)  # OCR ไทย (reuse ทั้งโฟลเดอร์)
    ocr_en = PaddleOCR(use_angle_cls=True, lang=FALLBACK_LANG)  # OCR อังกฤษ (fallback)

    out_jsonl = os.path.join(out_root, "images.jsonl")  # ตั้งชื่อไฟล์ output jsonl

    def emit(payload: Dict[str, Any]) -> None:
        """helper ส่ง progress ถ้ามี callback"""
        if progress_cb:  # ถ้ามี callback
            progress_cb(payload)  # ส่ง payload

    emit({  # แจ้งเริ่ม OCR รูป
        "stage": "img_start",  # stage
        "total_images": total,  # จำนวนรูปทั้งหมด
        "done_images": 0,  # ทำเสร็จแล้วกี่รูป
        "percent": 0,  # % โดยรวม
        "message": "starting image ocr",  # ข้อความ
    })  # จบ payload

    for idx, ipath in enumerate(img_paths, start=1):  # วนทีละรูป
        try:
            im = Image.open(ipath).convert("RGB")  # เปิดรูปและแปลงเป็น RGB
        except Exception:
            continue  # เปิดไม่ได้ก็ข้าม

        try:
            txt1 = paddle_ocr_image_text_only(ocr_th, im)  # OCR ไทยก่อน
            txt = txt1  # ตั้งค่าเริ่มต้นเป็นผลไทย
            if len(txt1) < 30:  # ถ้าข้อความสั้นมาก
                txt2 = paddle_ocr_image_text_only(ocr_en, im)  # ลอง OCR อังกฤษ
                if len(txt2) > len(txt1):  # ถ้าอังกฤษยาวกว่า
                    txt = txt2  # ใช้อังกฤษแทน
        except Exception:
            txt = ""  # ถ้า OCR error ให้เป็นว่าง

        save_jsonl(out_jsonl, [{  # เขียนผลลง jsonl (append)
            "type": "web_image",  # ประเภท
            "image_path": ipath,  # path รูป
            "text": txt,  # ข้อความ OCR
        }])  # จบ list rows

        outputs.append(out_jsonl)  # เก็บ output path (ซ้ำได้ เดี๋ยวค่อย unique)

        emit({  # แจ้ง progress รายรูป
            "stage": "img_done",  # stage
            "total_images": total,  # จำนวนทั้งหมด
            "done_images": idx,  # ทำเสร็จแล้ว
            "percent": int(idx * 100 / max(total, 1)),  # คิด %
            "current_file": ipath,  # รูปปัจจุบัน
            "message": f"image {idx}/{total}",  # ข้อความ
        })  # จบ payload

    emit({  # แจ้งจบ OCR รูปทั้งหมด
        "stage": "img_finished",  # stage
        "total_images": total,  # total
        "done_images": total,  # done
        "percent": 100,  # 100%
        "message": "image ocr done",  # ข้อความ
    })  # จบ payload

    return sorted(list(set(outputs)))  # คืน list output แบบไม่ซ้ำและเรียง
