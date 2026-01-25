"""
web_scraping.py

สรุปภาพรวม:
- FastAPI endpoint /scrape เพื่อ scrape เว็บตาม keyword (Google search ผ่าน Playwright)
- ดึง: web text + ไฟล์ (pdf/เอกสาร) + รูปภาพ
- บันทึกโครงสร้างโฟลเดอร์: data_<keyword>_<timestamp>/
  - final_data.csv (รวมผลจากหลายเว็บ)
  - <site_folder>/
      - content.txt (text จากหน้า)
      - images/ (รูปที่ดาวน์โหลด)
      - images_meta.jsonl (caption + relevance)
      - files/ (ไฟล์ที่ดาวน์โหลด)
- ส่งงาน OCR เป็น background queue:
  - OCR จะ process PDF/รูป (ตาม pipeline ใน ocr_pipeline.py)
  - เช็คสถานะได้ที่ /ocr/status/{job_id}

ความสามารถใหม่ (สำคัญ):
- หลังดาวน์โหลดรูปจากเว็บ จะเรียก Ollama Vision model (เช่น MiniCPM-V 2.6)
  เพื่อสร้าง caption + ประเมินความเกี่ยวข้องกับ text ของเว็บ (relevance score)

สิ่งที่ต้องมี:
- ติดตั้ง: fastapi uvicorn playwright bs4 requests pandas pillow python-dotenv scikit-learn
- Playwright install browser: playwright install
- Ollama รันอยู่ที่ http://localhost:11434 และมี model vision เช่น minicpm-v:2.6

ENV:
- OLLAMA_HOST (default: http://localhost:11434)
- OLLAMA_VISION_MODEL (default: minicpm-v:2.6)
"""

from fastapi import FastAPI  # นำเข้า FastAPI สำหรับสร้าง API
import uvicorn  # ใช้รันเซิร์ฟเวอร์

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError  # Playwright แบบ sync
from bs4 import BeautifulSoup  # HTML parsing
from urllib.parse import urljoin  # รวม URL relative -> absolute

import os  # path / env
import re  # regex
import io  # bytes buffer
import time  # sleep/timeout
import random  # สุ่มชื่อไฟล์
import hashlib  # sha1
import datetime  # timestamp
import requests  # http client
import pandas as pd  # ทำตาราง + export csv
import unicodedata  # normalize string
from PIL import Image  # ตรวจภาพ + ขนาด

from dotenv import load_dotenv  # โหลด .env
import threading  # ทำ worker thread
import queue  # queue งาน
import uuid  # job id
import traceback  # stack trace
from typing import Dict, Any, Optional, List, Tuple  # typing

from sklearn.feature_extraction.text import TfidfVectorizer  # ใช้หา similarity
from sklearn.metrics.pairwise import cosine_similarity  # cosine similarity

from ocr_pipeline import process_folder_pdfs, process_folder_images  # OCR pipeline

# Use this > http://localhost:8000/docs When run a code

# ----------------------------
# ENV + APP 
# ----------------------------

load_dotenv()  # โหลดค่า ENV จากไฟล์ .env

app = FastAPI()  # สร้างแอป FastAPI

# ----------------------------
# OCR Job Registry (in-memory) + Worker Queue
# ----------------------------

OCR_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue()  # คิวเก็บงาน OCR
OCR_STATUS: Dict[str, Dict[str, Any]] = {}  # สถานะงาน OCR แต่ละ job_id
OCR_LOCK = threading.Lock()  # lock กัน race condition
OCR_WORKER_STARTED = False  # flag ว่าเริ่ม worker แล้วหรือยัง


def ocr_worker_loop() -> None:
    """
    '''
    Worker loop สำหรับทำ OCR แบบ background
    - รอรับ job จาก OCR_QUEUE (blocking)
    - อัปเดต OCR_STATUS ตาม progress
    - เรียก process_folder_pdfs + process_folder_images
    - เก็บ outputs และ error ลง status
    '''
    """
    print("✅ OCR worker started", flush=True)  # log ว่า worker เริ่มแล้ว

    while True:  # loop ตลอดเวลา
        job = OCR_QUEUE.get()  # ดึงงานจากคิวแบบ block
        if job is None:  # ถ้าเจอ sentinel -> ออกจาก loop
            OCR_QUEUE.task_done()  # บอกคิวว่าทำงานเสร็จแล้ว
            break  # ออกจาก while

        job_id = job["job_id"]  # อ่าน job_id
        files_dir = job["files_dir"]  # โฟลเดอร์หลักที่มี pdf/images
        out_root = job["out_root"]  # โฟลเดอร์ output ของ OCR

        try:
            with OCR_LOCK:  # ล็อคก่อนแก้ OCR_STATUS
                OCR_STATUS[job_id]["status"] = "running"  # เปลี่ยนสถานะเป็น running
                OCR_STATUS[job_id]["started_at"] = datetime.datetime.now().isoformat()  # เวลาเริ่ม
                OCR_STATUS[job_id]["progress"]["stage"] = "scanning"  # stage
                OCR_STATUS[job_id]["progress"]["message"] = "scanning files..."  # ข้อความ

            def progress_cb(p: dict) -> None:
                """
                '''
                callback รับ progress จาก ocr_pipeline.py
                - อัปเดต OCR_STATUS[job_id]["progress"] ด้วย payload p
                '''
                """
                with OCR_LOCK:  # ล็อค
                    if job_id in OCR_STATUS:  # กัน job หาย
                        OCR_STATUS[job_id]["progress"].update(p)  # merge progress
                        OCR_STATUS[job_id]["progress"]["updated_at"] = datetime.datetime.now().isoformat()  # เวลาล่าสุด

            # 1) OCR PDFs
            pdf_outputs = process_folder_pdfs(files_dir=files_dir, out_root=out_root, progress_cb=progress_cb)  # ทำ OCR pdf

            # 2) OCR Images (รูปจากเว็บที่ดาวน์โหลด)
            img_out_root = os.path.join(out_root, "web_images")  # แยกโฟลเดอร์ผล OCR รูป
            img_outputs = process_folder_images(root_dir=files_dir, out_root=img_out_root, progress_cb=progress_cb)  # ทำ OCR รูป

            outputs = {"pdf": pdf_outputs, "images": img_outputs}  # รวม outputs

            with OCR_LOCK:  # ล็อค
                OCR_STATUS[job_id]["status"] = "done"  # done
                OCR_STATUS[job_id]["outputs"] = outputs  # เก็บ outputs
                OCR_STATUS[job_id]["finished_at"] = datetime.datetime.now().isoformat()  # เวลาจบ
                OCR_STATUS[job_id]["progress"]["stage"] = "done"  # stage done
                OCR_STATUS[job_id]["progress"]["percent"] = 100  # 100%
                OCR_STATUS[job_id]["progress"]["message"] = "ocr completed"  # msg

        except Exception as e:
            with OCR_LOCK:  # ล็อค
                OCR_STATUS[job_id]["status"] = "error"  # error
                OCR_STATUS[job_id]["error"] = str(e)  # error string
                OCR_STATUS[job_id]["traceback"] = traceback.format_exc()  # stack trace
                OCR_STATUS[job_id]["finished_at"] = datetime.datetime.now().isoformat()  # เวลาจบ
                OCR_STATUS[job_id]["progress"]["stage"] = "failed"  # stage fail
                OCR_STATUS[job_id]["progress"]["message"] = "ocr failed"  # msg fail

        finally:
            OCR_QUEUE.task_done()  # บอกคิวว่าทำงานเสร็จ


def ensure_ocr_worker_started() -> None:
    """
    '''
    เริ่ม OCR worker thread เพียงครั้งเดียว
    - ถ้าเริ่มแล้วจะ return
    - ถ้ายังไม่เริ่ม จะสร้าง daemon thread แล้ว start
    '''
    """
    global OCR_WORKER_STARTED  # ใช้ global flag
    if OCR_WORKER_STARTED:  # ถ้าเริ่มแล้ว
        return  # ไม่ทำอะไร
    t = threading.Thread(target=ocr_worker_loop, daemon=True)  # สร้าง thread daemon
    t.start()  # start
    OCR_WORKER_STARTED = True  # ตั้ง flag เริ่มแล้ว


# ----------------------------
# Config
# ----------------------------

USER_AGENT = (  # user-agent ลดการโดนบล็อกเบื้องต้น
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

BLOCK_KEYWORDS = [  # คำที่เจอบ่อยเวลาถูกบล็อก
    "imperva", "security check", "hcaptcha", "verify you are human",
    "access denied", "cloudflare", "captcha"
]

IGNORE_IMAGE_KEYWORDS = [  # คำใน URL ที่มักเป็นภาพไร้สาระ
    "logo", "icon", "sprite", "button", "spacer", "blank", "pixel",
    "avatar", "badge", "placeholder"
]

ALLOWED_FILE_CT = {  # mapping content-type -> extension
    "application/pdf": ".pdf",
    "application/octet-stream": ".bin",
    "application/zip": ".zip",
    "application/x-zip-compressed": ".zip",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
}

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # host ของ Ollama
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "minicpm-v:2.6")  # รุ่น vision model

# ----------------------------
# Output Base Directory
# ----------------------------
# โฟลเดอร์กลางสำหรับเก็บผล scrape ทั้งหมด (กันรก root โปรเจค)
# โครงสร้างจะเป็น: scraped_outputs/data_<keyword>_<timestamp>/
OUTPUT_BASE_DIR = os.path.join(os.getcwd(), "scraped_outputs")


# ----------------------------
# Utilities
# ----------------------------

def safe_mkdir(path: str) -> None:
    """
    '''
    สร้างโฟลเดอร์ ถ้าไม่มี
    - ใช้ exist_ok=True เพื่อไม่ error หากมีอยู่แล้ว
    '''
    """
    os.makedirs(path, exist_ok=True)  # สร้างโฟลเดอร์


def sha1_bytes(b: bytes) -> str:
    """
    '''
    ทำ sha1 ของ bytes
    - ใช้ตรวจ duplicate (รูป/ไฟล์เหมือนกัน)
    '''
    """
    return hashlib.sha1(b).hexdigest()  # คืนค่า sha1 hex


def safe_filename(name: str, max_len: int = 50) -> str:
    """
    '''
    ทำชื่อไฟล์ให้ปลอดภัยกับ Windows/Linux
    - normalize NFKC
    - ลบอักขระต้องห้าม
    - จำกัดความยาว
    '''
    """
    name = (name or "")  # กัน None
    name = unicodedata.normalize("NFKC", name)  # normalize
    name = re.sub(r"[\\/*?\"<>|:]", "", name)  # ลบ char ผิดกฎ
    name = re.sub(r"\s+", " ", name).strip()  # จัด space
    name = name[:max_len].rstrip(" .").strip()  # จำกัดและลบจุดท้าย
    return name if name else "site"  # ถ้าว่างให้ใช้ site


def now_stamp() -> str:
    """
    '''
    คืน timestamp สำหรับตั้งชื่อโฟลเดอร์/ไฟล์
    '''
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # รูปแบบ yyyyMMdd_HHmmss


def is_probably_blocked(text: str) -> bool:
    """
    '''
    heuristics ตรวจว่าอาจโดน block / หน้า captcha
    - มีคำใน BLOCK_KEYWORDS
    - หรือมีข้อความน้อยผิดปกติ (<250 char)
    '''
    """
    t = (text or "").lower()  # lower-case
    if any(k in t for k in BLOCK_KEYWORDS):  # มี keyword บล็อก
        return True  # blocked
    if len(t.strip()) < 250:  # ข้อความน้อยผิดปกติ
        return True  # blocked
    return False  # ปกติ


def normalize_text(raw: str, min_line_len: int = 10) -> str:
    """
    '''
    ทำความสะอาด text
    - normalize unicode
    - ลด whitespace ซ้ำ
    - ตัดบรรทัดสั้นมาก
    - ลบซ้ำ (dedupe line)
    '''
    """
    if not raw:  # ถ้าไม่มี
        return ""  # คืนว่าง
    s = unicodedata.normalize("NFKC", raw)  # normalize
    s = s.replace("\u00a0", " ")  # nbsp -> space
    s = re.sub(r"[^\S\r\n]+", " ", s)  # space ชนิดอื่น -> space
    s = re.sub(r"\n{3,}", "\n\n", s)  # newline เยอะเกิน -> 2
    s = re.sub(r"[ \t]{2,}", " ", s)  # space/tab ซ้ำ -> 1

    lines: List[str] = []  # เก็บบรรทัด
    seen = set()  # ใช้ dedupe
    for line in s.splitlines():  # loop ทีละบรรทัด
        line = line.strip()  # trim
        if len(line) < min_line_len:  # สั้นเกิน
            continue  # ข้าม
        key = re.sub(r"\s+", " ", line)  # normalize เพื่อ dedupe
        if key in seen:  # ถ้าเคยเห็นแล้ว
            continue  # ข้าม
        seen.add(key)  # mark seen
        lines.append(line)  # เก็บบรรทัด
    return "\n".join(lines)  # รวมกลับเป็น text


def content_text_from_html(html: str) -> str:
    """
    '''
    ดึง text หลักจาก HTML
    - ตัด script/style/noscript/svg/canvas
    - ตัดส่วน header/footer/nav/aside
    - เน้นอ่านจาก article/main/body
    '''
    """
    soup = BeautifulSoup(html or "", "html.parser")  # parse

    for tag in soup(["script", "style", "noscript", "svg", "canvas"]):  # tag ไม่เอา
        tag.decompose()  # ลบออก

    for tag_name in ["header", "footer", "nav", "aside"]:  # โครงสร้างที่มักไร้สาระ
        for t in soup.find_all(tag_name):  # หา tag ทั้งหมด
            t.decompose()  # ลบทิ้ง

    main = soup.find("article") or soup.find("main") or soup.body  # เลือก main content
    text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)  # ดึงข้อความ
    return normalize_text(text, min_line_len=10)  # clean


def looks_like_file_url(u: str) -> bool:
    """
    '''
    heuristic ว่า URL น่าจะเป็นลิงก์ไฟล์ดาวน์โหลด
    '''
    """
    ul = (u or "").lower()  # lower
    if any(x in ul for x in [".pdf", "getmedia", "download", "attachment", "file", "/getmedia/", ".pdf.aspx"]):
        return True  # น่าจะใช่
    return False  # ไม่น่าใช่


def is_direct_pdf_url(u: str) -> bool:
    """
    '''
    ตรวจ URL ที่ลงท้าย .pdf หรือมี .pdf? เพื่อจัดเป็น direct pdf
    '''
    """
    ul = (u or "").lower()  # lower
    return ul.endswith(".pdf") or ".pdf?" in ul  # เงื่อนไข


def goto_safely(page, url: str, timeout_ms: int = 90000) -> None:
    """
    '''
    ไปหน้าเว็บแบบปลอดภัยขึ้น
    - wait domcontentloaded
    - พยายามรอ networkidle สั้นๆ (ถ้า timeout ก็ข้าม)
    - หน่วงอีกนิดให้ DOM update
    '''
    """
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)  # ไปหน้า
    try:
        page.wait_for_load_state("networkidle", timeout=5000)  # รอ network idle
    except PWTimeoutError:
        pass  # ถ้า timeout ไม่เป็นไร
    page.wait_for_timeout(1500)  # รอเพิ่มให้ render เสถียร


# ----------------------------
# AI Image Captioning + Relevance
# ----------------------------

def caption_image_with_ollama(image_path: str, prompt: str) -> str:
    """
    '''
    เรียก Ollama Vision model เพื่อบรรยายภาพ (caption)
    - ใช้ endpoint: POST {OLLAMA_HOST}/api/generate
    - ส่ง image เป็น base64 ใน field "images"
    - ส่ง prompt เป็นภาษาไทย/อังกฤษได้

    คืนค่า:
    - string caption (ถ้า fail จะคืน "")

    หมายเหตุ:
    - ต้องมี Ollama รันอยู่ และ model รองรับ vision (เช่น minicpm-v:2.6)
    '''
    """
    try:  # ครอบ try กันพัง
        import base64  # import ในฟังก์ชันเพื่อไม่บังคับตอน import module

        with open(image_path, "rb") as f:  # เปิดไฟล์ภาพ
            b = f.read()  # อ่าน bytes

        img_b64 = base64.b64encode(b).decode("utf-8")  # bytes -> base64 str

        payload = {  # สร้าง payload ให้ Ollama
            "model": OLLAMA_VISION_MODEL,  # รุ่นโมเดล
            "prompt": prompt,  # prompt
            "images": [img_b64],  # แนบภาพ
            "stream": False,  # ไม่สตรีม
        }

        r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=90)  # ยิง request
        if r.status_code != 200:  # ถ้าไม่ 200
            return ""  # fail -> คืนว่าง

        data = r.json()  # อ่าน json
        return (data.get("response") or "").strip()  # response text

    except Exception:
        return ""  # ถ้า exception -> คืนว่าง


def compute_relevance_score(page_text: str, image_caption: str) -> float:
    """
    '''
    ประเมินความเกี่ยวข้องระหว่างรูป (caption) กับเนื้อหาเว็บ (page_text)
    - ใช้ TF-IDF + cosine similarity
    - คืนค่า score 0..1 (โดยประมาณ)

    เหตุผล:
    - เบา, ไม่ต้องมี embedding model เพิ่ม
    - ใช้เป็น baseline ได้ดีสำหรับ “เกี่ยว/ไม่เกี่ยว”
    '''
    """
    if not page_text or not image_caption:  # ถ้าขาดอย่างใดอย่างหนึ่ง
        return 0.0  # score 0

    # จำกัดความยาวเพื่อความเร็ว (กัน text ยาวมาก)
    t1 = page_text[:6000]  # ตัดเหลือ 6000 ตัวอักษร
    t2 = image_caption[:600]  # caption มักสั้น ตัดเหลือ 600

    try:
        vec = TfidfVectorizer(stop_words=None)  # สร้าง vectorizer
        X = vec.fit_transform([t1, t2])  # fit+transform สองข้อความ
        score = float(cosine_similarity(X[0], X[1])[0][0])  # cosine
        if score < 0:  # กันค่าติดลบ
            return 0.0
        if score > 1:  # กันค่าเกิน 1
            return 1.0
        return score
    except Exception:
        return 0.0


def save_jsonl(path: str, row: Dict[str, Any]) -> None:
    """
    '''
    append 1 แถวลง jsonl
    '''
    """
    import json  # import local
    with open(path, "a", encoding="utf-8") as f:  # เปิดแบบ append
        f.write(json.dumps(row, ensure_ascii=False) + "\n")  # เขียน 1 บรรทัด


# ----------------------------
# Image Selection & Download
# ----------------------------

def extract_image_candidates(html: str, base_url: str) -> List[str]:
    """
    '''
    ดึง candidate รูปจาก HTML
    - og:image / twitter:image
    - img tags ใน article/main/body
    - เก็บเป็น absolute url
    - dedupe
    '''
    """
    soup = BeautifulSoup(html or "", "html.parser")  # parse
    urls: List[str] = []  # เก็บ url

    def push(u: Optional[str]) -> None:
        """เพิ่ม URL เข้า list ถ้า valid"""
        if not u:  # ถ้า None/ว่าง
            return  # ไม่ทำ
        full = urljoin(base_url, u)  # รวมเป็น absolute
        if full.startswith("http"):  # ต้องเป็น http
            urls.append(full)  # เพิ่มลง list

    og = soup.find("meta", property="og:image")  # og:image
    tw = soup.find("meta", attrs={"name": "twitter:image"})  # twitter:image
    push(og.get("content") if og else None)  # push og
    push(tw.get("content") if tw else None)  # push tw

    container = soup.find("article") or soup.find("main") or soup.body  # container หลัก
    if container:  # ถ้ามี
        for img in container.find_all("img"):  # วน img
            push(img.get("src"))  # src
            push(img.get("data-src"))  # data-src
            push(img.get("data-lazy-src"))  # lazy

    uniq: List[str] = []  # list ไม่ซ้ำ
    seen = set()  # set กันซ้ำ
    for u in urls:  # วนทุก url
        if u in seen:  # ถ้าเคยเห็น
            continue  # ข้าม
        seen.add(u)  # mark
        uniq.append(u)  # เพิ่ม
    return uniq  # คืน list


def is_good_image_url(url: str) -> bool:
    """
    '''
    กรอง URL รูปที่ไม่น่าใช้
    - data:
    - มี keyword พวก logo/icon
    - svg
    '''
    """
    ul = (url or "").lower()  # lower
    if ul.startswith("data:"):  # base64 inline
        return False
    if any(k in ul for k in IGNORE_IMAGE_KEYWORDS):  # พวก logo/icon
        return False
    if ul.endswith(".svg"):  # svg มักไม่เหมาะสำหรับ OCR/caption
        return False
    return True


def download_images(
    session: requests.Session,
    image_urls: List[str],
    save_folder: str,
    page_text_for_relevance: str,
    max_images: int = 10
) -> Tuple[int, str]:
    """
    '''
    ดาวน์โหลดรูปจาก list url และทำ “AI caption + relevance”
    - บันทึกรูปลง save_folder/images
    - บันทึก metadata ลง save_folder/images_meta.jsonl

    เงื่อนไขคัดรูป:
    - content-type ต้องเป็น image
    - ขนาดไฟล์ > 20KB (ตัดรูปเล็ก/ไอคอน)
    - ตรวจขนาดรูป (>= 350x250)
    - dedupe ด้วย sha1

    คืนค่า:
    - (จำนวนรูปที่เซฟ, path ของ jsonl metadata)
    '''
    """
    safe_mkdir(save_folder)  # สร้างโฟลเดอร์
    meta_path = os.path.join(save_folder, "images_meta.jsonl")  # path metadata
    count = 0  # นับรูป
    seen_hash = set()  # กันรูปซ้ำ
    headers = {"User-Agent": USER_AGENT}  # headers

    for u in image_urls:  # วนทุก url
        if count >= max_images:  # ถ้าเกิน max
            break  # หยุด
        if not is_good_image_url(u):  # ถ้า url ไม่ดี
            continue  # ข้าม

        try:
            r = session.get(u, timeout=15, headers=headers, stream=True)  # ยิง request
            if r.status_code != 200:  # ถ้าไม่สำเร็จ
                continue  # ข้าม

            ct = (r.headers.get("content-type") or "").lower()  # content-type
            if "image" not in ct or "svg" in ct:  # ต้องเป็น image และไม่ใช่ svg
                continue  # ข้าม

            data = r.content  # bytes รูป
            if len(data) < 20 * 1024:  # <20KB
                continue  # ข้าม

            h = sha1_bytes(data)  # hash
            if h in seen_hash:  # ถ้าซ้ำ
                continue  # ข้าม
            seen_hash.add(h)  # mark

            try:
                img = Image.open(io.BytesIO(data))  # เปิดภาพ
                w, h2 = img.size  # ขนาด
                if w < 350 or h2 < 250:  # เล็กเกิน
                    continue  # ข้าม
            except Exception:
                continue  # ถ้าเปิดไม่ได้ -> ข้าม

            ext = "jpg"  # default ext
            if "png" in ct:  # ถ้าเป็น png
                ext = "png"
            elif "webp" in ct:  # ถ้าเป็น webp
                ext = "webp"

            fname = os.path.join(save_folder, f"img_{count+1}_{random.randint(100,999)}.{ext}")  # ตั้งชื่อไฟล์
            with open(fname, "wb") as f:  # เปิดไฟล์เขียน
                f.write(data)  # เขียนรูป

            # ---- AI Caption ----
            cap_prompt = (
                "อธิบายภาพนี้สั้นๆ ว่าคืออะไร (เน้นบริบทธุรกิจ/การเงินถ้ามี) "
                "ถ้าเป็นกราฟ/ตาราง ให้บอกว่าเป็นกราฟอะไรและพูดถึงอะไร"
            )  # prompt ภาษาไทย

            caption = caption_image_with_ollama(fname, cap_prompt)  # เรียก vision
            score = compute_relevance_score(page_text_for_relevance, caption)  # similarity

            save_jsonl(meta_path, {  # บันทึก metadata
                "type": "web_image",
                "source_url": u,
                "saved_path": fname,
                "sha1": h,
                "caption": caption,
                "relevance_score": score,
                "created_at": datetime.datetime.now().isoformat(),
            })

            count += 1  # เพิ่ม count

        except Exception:
            continue  # ถ้า download fail ก็ข้าม

    return count, meta_path  # คืนจำนวนรูป และ path metadata


# ----------------------------
# PDF / File Download
# ----------------------------

def extract_file_links(html: str, base_url: str) -> List[str]:
    """
    '''
    ดึงลิงก์ที่ “ดูเหมือนไฟล์” จาก HTML
    - จาก <a href="...">
    - จาก onclick="...'url'..."
    - dedupe
    '''
    """
    soup = BeautifulSoup(html or "", "html.parser")  # parse
    links: List[str] = []  # list ลิงก์

    for a in soup.find_all("a", href=True):  # loop <a>
        href = a.get("href")  # เอา href
        full = urljoin(base_url, href)  # ทำ absolute
        if full.startswith("http") and looks_like_file_url(full):  # ถ้าดูเหมือนไฟล์
            links.append(full)  # เก็บ

    for tag in soup.select("[onclick]"):  # loop element ที่มี onclick
        onclick = tag.get("onclick") or ""  # string onclick
        m = re.search(r"""['"]([^'"]+)['"]""", onclick)  # หา url ใน quote
        if m:  # ถ้าเจอ
            full = urljoin(base_url, m.group(1))  # ทำ absolute
            if full.startswith("http") and looks_like_file_url(full):  # เช็ค
                links.append(full)  # เก็บ

    uniq: List[str] = []  # ไม่ซ้ำ
    seen = set()  # set
    for u in links:  # วน
        if u in seen:  # ถ้าซ้ำ
            continue  # ข้าม
        seen.add(u)  # mark
        uniq.append(u)  # เพิ่ม
    return uniq  # คืน


def guess_ext_from_ct(ct: str) -> str:
    """
    '''
    เดานามสกุลไฟล์จาก content-type
    - ถ้าไม่รู้จัก: ถ้ามี pdf ให้ .pdf ไม่งั้น .bin
    '''
    """
    ct = (ct or "").split(";")[0].strip().lower()  # ตัด ;charset และ normalize
    return ALLOWED_FILE_CT.get(ct, ".pdf" if "pdf" in ct else ".bin")  # map หรือเดา


def save_response_content(save_path: str, content: bytes) -> None:
    """
    '''
    เซฟ bytes ลงไฟล์
    '''
    """
    with open(save_path, "wb") as f:  # เปิดเขียน
        f.write(content)  # เขียน


def download_direct_file_if_any(context, url: str, save_folder: str, referer: str = "") -> int:
    """
    '''
    ดาวน์โหลดไฟล์โดยตรงผ่าน Playwright context.request
    - ใช้ตอน URL เป็น direct pdf
    - เช็ค content-type เป็น pdf/octet-stream
    - เซฟเป็นไฟล์ลง save_folder
    '''
    """
    safe_mkdir(save_folder)  # สร้างโฟลเดอร์
    headers = {"User-Agent": USER_AGENT}  # headers
    if referer:  # ถ้ามี referer
        headers["Referer"] = referer  # ใส่ referer

    try:
        resp = context.request.get(url, headers=headers, timeout=30000)  # request
        if not resp.ok:  # ถ้าไม่ ok
            return 0  # ไม่ได้ไฟล์

        ct = (resp.headers.get("content-type") or "").lower()  # content-type
        if "pdf" not in ct and "octet-stream" not in ct:  # ถ้าไม่ใช่ไฟล์
            return 0

        data = resp.body()  # bytes
        if not data or len(data) < 10 * 1024:  # ไฟล์เล็กเกิน
            return 0

        ext = guess_ext_from_ct(ct)  # เดา ext
        out = os.path.join(save_folder, f"direct_{random.randint(100,999)}{ext}")  # ตั้งชื่อ
        save_response_content(out, data)  # เซฟ
        return 1  # ได้ 1 ไฟล์
    except Exception:
        return 0  # fail


def download_files_via_playwright_request(
    context,
    urls: List[str],
    save_folder: str,
    base_url: str,
    max_files: int = 100
) -> int:
    """
    '''
    ดาวน์โหลดไฟล์จาก list url ผ่าน Playwright context.request
    - ส่ง referer เป็น base_url เพื่อลด 403
    - skip text/html
    - เซฟไฟล์ลง save_folder
    '''
    """
    safe_mkdir(save_folder)  # สร้างโฟลเดอร์
    count = 0  # นับไฟล์

    for u in urls:  # วนลิงก์
        if count >= max_files:  # เกิน max
            break  # หยุด
        try:
            resp = context.request.get(  # request
                u,
                headers={"User-Agent": USER_AGENT, "Referer": base_url},
                timeout=30000,
            )
            if not resp.ok:  # ไม่ ok
                continue  # ข้าม

            ct = (resp.headers.get("content-type") or "").lower()  # content-type
            if "text/html" in ct:  # ถ้าเป็น html
                continue  # ข้าม

            data = resp.body()  # bytes
            if not data or len(data) < 10 * 1024:  # เล็กเกิน
                continue  # ข้าม

            ext = guess_ext_from_ct(ct)  # เดา ext
            out = os.path.join(save_folder, f"file_{count+1}_{random.randint(100,999)}{ext}")  # ตั้งชื่อ
            save_response_content(out, data)  # เซฟ
            count += 1  # เพิ่ม count
        except Exception:
            continue  # fail ก็ข้าม

    return count  # คืนจำนวนไฟล์


# ----------------------------
# Main Scraper
# ----------------------------

def google_search_targets(page, keyword: str, max_links: int) -> List[Dict[str, str]]:
    """
    '''
    ค้น Google ด้วย Playwright แล้วดึงผลลัพธ์ (title,url)
    - ไป google.com
    - พิมพ์ keyword
    - รอผลลัพธ์
    - ดึง a:has(h3)

    หมายเหตุ:
    - Google อาจ block ได้ง่าย ถ้าโดนให้เปลี่ยนไปใช้ Bing หรือ SerpAPI ภายหลัง
    '''
    """
    page.goto("https://www.google.com/", wait_until="domcontentloaded", timeout=60000)  # ไปหน้า google
    page.locator("textarea[name='q'], input[name='q']").fill(keyword)  # ใส่คำค้น
    page.keyboard.press("Enter")  # กด Enter
    page.wait_for_selector("a:has(h3)", timeout=15000)  # รอ selector ผลลัพธ์

    items: List[Dict[str, str]] = []  # list results
    for a in page.locator("a:has(h3)").all()[: max_links * 2]:  # อ่านเผื่อคัด
        try:
            url = a.get_attribute("href")  # href
            title = a.inner_text().strip()  # ข้อความ
            if not url or "google.com" in url:  # ตัดลิงก์ google
                continue
            if url.startswith("/url?"):  # ลิงก์ redirect ของ google
                continue
            items.append({"url": url, "title": title})  # เก็บ
            if len(items) >= max_links:  # ครบ
                break
        except Exception:
            continue

    return items  # คืน


def scrape_one(page, context, site: Dict[str, str], site_folder: str) -> Optional[Dict[str, Any]]:
    """
    '''
    scrape หนึ่งเว็บ
    - ถ้า URL เป็น direct pdf -> download แล้ว return record
    - ถ้าเป็นเว็บ:
      - goto
      - ดึง visible text + html
      - parse เอา main content text
      - ดึงรูป + ไฟล์
      - ดาวน์โหลดรูป (พร้อม caption+relevance)
      - ดาวน์โหลดไฟล์
      - บันทึก content.txt เพื่อใช้งานต่อใน RAG

    คืน:
    - dict record สำหรับรวมเป็น DataFrame
    - หรือ None ถ้าไม่เจอ content ที่มีสาระ
    '''
    """
    safe_mkdir(site_folder)  # สร้างโฟลเดอร์เว็บ
    url = site["url"]  # url
    title = site["title"]  # title

    try:
        # ----- direct pdf -----
        if is_direct_pdf_url(url):  # ถ้าเป็นลิงก์ pdf ตรงๆ
            dl_folder = os.path.join(site_folder, "files")  # โฟลเดอร์ไฟล์
            saved = download_direct_file_if_any(context, url, dl_folder, referer=url)  # โหลด
            return {  # คืน record
                "Source": title,
                "URL": url,
                "Tool": "DirectPDF",
                "Content": f"Direct PDF file. Saved={saved}.",
                "ImagesSaved": 0,
                "FilesSaved": saved,
                "Folder": site_folder,
            }

        # ----- open page -----
        goto_safely(page, url)  # ไปหน้าเว็บ
        page.wait_for_timeout(1200)  # หน่วง

        raw_visible = page.inner_text("body")  # text ทั้งหน้า
        if is_probably_blocked(raw_visible):  # ถ้าเหมือนโดนบล็อก
            try:
                page.reload(wait_until="domcontentloaded", timeout=45000)  # reload
                page.wait_for_timeout(1200)  # หน่วง
                raw_visible = page.inner_text("body")  # อ่านใหม่
            except Exception:
                pass  # ถ้าพังไม่เป็นไร

        html = page.content()  # html
        content = content_text_from_html(html)  # main content

        if not content or len(content) < 300:  # ถ้า main content สั้น
            content = normalize_text(raw_visible, min_line_len=10)  # fallback จาก visible text

        if not content or len(content) < 80:  # ถ้ายังสั้นมาก
            return None  # ข้ามเว็บนี้

        # บันทึก content.txt เพื่อใช้ downstream (RAG/Agent)
        with open(os.path.join(site_folder, "content.txt"), "w", encoding="utf-8") as f:
            f.write(content)

        # ----- extract assets -----
        img_candidates = extract_image_candidates(html, url)  # ลิงก์รูป
        file_links = extract_file_links(html, url)  # ลิงก์ไฟล์

        s = requests.Session()  # session
        s.headers.update({"User-Agent": USER_AGENT})  # set UA

        # ----- download images + caption -----
        img_folder = os.path.join(site_folder, "images")  # โฟลเดอร์รูป
        img_count, meta_path = download_images(  # ดาวน์โหลดรูป
            session=s,
            image_urls=img_candidates,
            save_folder=img_folder,
            page_text_for_relevance=content,
            max_images=10,
        )

        # ----- download files -----
        dl_folder = os.path.join(site_folder, "files")  # โฟลเดอร์ไฟล์
        file_count = download_files_via_playwright_request(  # download file
            context=context,
            urls=file_links,
            save_folder=dl_folder,
            base_url=url,
            max_files=100,
        )

        return {  # record
            "Source": title,
            "URL": url,
            "Tool": "Playwright",
            "Content": content,
            "ImagesSaved": img_count,
            "FilesSaved": file_count,
            "ImagesMeta": meta_path,
            "Folder": site_folder,
        }

    except Exception as e:
        print(f"      ❌ Error: {e}", flush=True)  # log
        return None  # fail -> None


def run_scraper_logic(keyword: str, max_links: int = 5) -> Dict[str, Any]:
    """
    '''
    logic หลักสำหรับ scrape หลายเว็บ
    - สร้างโฟลเดอร์ data_<keyword>_<timestamp>
    - เปิด Chrome persistent context (ลด captcha บ้าง + ใช้ session ได้)
    - google_search_targets เพื่อได้ list เว็บ
    - scrape_one ทีละเว็บ แล้วรวมผล
    - export final_data.csv
    - enqueue OCR job แล้วคืน ocr_job_id + status url
    '''
    """
    ts = now_stamp()  # timestamp

    # สร้างโฟลเดอร์กลาง (scraped_outputs) แล้วค่อยสร้างโฟลเดอร์งานย่อย data_*
    safe_mkdir(OUTPUT_BASE_DIR)  # สร้างโฟลเดอร์กลาง
    main_folder = os.path.join(OUTPUT_BASE_DIR, f"data_{safe_filename(keyword, 30)}_{ts}")  # โฟลเดอร์หลัก
    safe_mkdir(main_folder)  # สร้างโฟลเดอร์หลัก (งานนี้)

    # ย้าย chrome profile ไปอยู่ใต้ scraped_outputs ด้วย (ไม่รก root โปรเจค)
    user_data_dir = os.path.join(OUTPUT_BASE_DIR, "chrome_user_data")  # โฟลเดอร์ profile chrome
    safe_mkdir(user_data_dir)  # สร้าง

    collected: List[Dict[str, Any]] = []  # list เก็บผล

    with sync_playwright() as p:  # start playwright
        context = p.chromium.launch_persistent_context(  # เปิด chrome persistent
            user_data_dir=user_data_dir,
            headless=False,
            channel="chrome",
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--start-maximized"],
            viewport={"width": 1366, "height": 768},
            locale="th-TH",
            ignore_https_errors=True,
        )

        page = context.pages[0] if context.pages else context.new_page()  # ใช้ page แรก หรือเปิดใหม่

        try:
            targets = google_search_targets(page, keyword, max_links=max_links)  # search google
            print(f"🔎 พบ {len(targets)} เว็บ", flush=True)  # log

            for i, site in enumerate(targets, start=1):  # วนเว็บ
                folder_name = f"{i}_{safe_filename(site['title'], 25)}"  # ตั้งชื่อโฟลเดอร์เว็บ
                site_folder = os.path.join(main_folder, folder_name)  # path เว็บ
                safe_mkdir(site_folder)  # สร้าง

                print(f"\n[{i}/{len(targets)}] 🚀 {site['title'][:60]}", flush=True)  # log

                item = scrape_one(page, context, site, site_folder)  # scrape เว็บเดียว
                if item:  # ถ้ามีผล
                    collected.append(item)  # เก็บ

                time.sleep(1)  # หน่วง ลด rate

        finally:
            context.close()  # ปิด context

    if not collected:  # ถ้าไม่เจอข้อมูลเลย
        return {"status": "failed", "message": "No data found", "folder": main_folder}  # fail

    df = pd.DataFrame(collected)  # ทำ DataFrame
    csv_path = os.path.join(main_folder, "final_data.csv")  # path csv
    df.to_csv(csv_path, index=False, encoding="utf-8-sig", errors="replace")  # เซฟ csv

    # ---- enqueue OCR job (background) ----
    ensure_ocr_worker_started()  # เริ่ม worker

    job_id = uuid.uuid4().hex[:12]  # job id สั้น
    out_root = os.path.join(main_folder, "ocr_results")  # output OCR

    with OCR_LOCK:  # lock
        OCR_STATUS[job_id] = {  # สร้าง status เริ่มต้น
            "status": "queued",
            "created_at": datetime.datetime.now().isoformat(),
            "main_folder": main_folder,
            "out_root": out_root,
            "progress": {
                "stage": "queued",
                "percent": 0,
                "done_files": 0,
                "total_files": 0,
                "current_index": 0,
                "current_file": "",
                "current_page": 0,
                "total_pages": 0,
                "message": "waiting in queue",
            },
        }

    OCR_QUEUE.put({  # push งานเข้าคิว
        "job_id": job_id,
        "files_dir": main_folder,
        "out_root": out_root,
    })

    return {  # คืน response
        "status": "success",
        "folder": main_folder,
        "file_path": csv_path,
        "data": collected,
        "ocr_job_id": job_id,
        "ocr_status_url": f"/ocr/status/{job_id}",
    }


# ----------------------------
# FastAPI Endpoints
# ----------------------------

@app.get("/")
def read_root() -> Dict[str, str]:
    """
    '''
    health check endpoint
    '''
    """
    return {"message": "AI Scraper API is running!"}  # response


@app.get("/scrape")
def trigger_scrape(keyword: str, amount: int = 5) -> Dict[str, Any]:
    """
    '''
    trigger scrape
    - keyword: คำค้น
    - amount: จำนวนเว็บ
    '''
    """
    print(f"🔔 คำสั่ง: '{keyword}' จำนวน {amount} เว็บ", flush=True)  # log
    return run_scraper_logic(keyword, amount)  # ทำงานจริง


@app.get("/ocr/status/{job_id}")
def ocr_status(job_id: str) -> Dict[str, Any]:
    """
    '''
    ดูสถานะงาน OCR ตาม job_id
    '''
    """
    with OCR_LOCK:  # lock
        st = OCR_STATUS.get(job_id)  # ดึง status
    if not st:  # ถ้าไม่เจอ
        return {"status": "not_found", "job_id": job_id}  # not found
    return {"job_id": job_id, **st}  # คืน status


if __name__ == "__main__":  # ถ้ารันไฟล์นี้โดยตรง
    uvicorn.run(app, host="0.0.0.0", port=8000)  # รัน server
