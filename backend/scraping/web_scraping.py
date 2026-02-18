# backend/scraping/web_scraping.py
from __future__ import annotations

import os
import re
import io
import json
import time
import random
import hashlib
import traceback
import unicodedata
import base64
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse, parse_qs, unquote

import requests
import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError
from PIL import Image

from backend.settings import settings
from backend.utils.text import normalize_text
from backend.utils.jsonl import append_jsonl, safe_mkdir
from backend.utils.job_manager import log
from backend.scraping.recaptcha_solver import RecaptchaSolver


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)

BLOCK_KEYWORDS = [
    "imperva", "security check", "hcaptcha", "verify you are human",
    "access denied", "cloudflare", "captcha"
]

JUNK_CLASS_ID_RE = re.compile(
    r"(nav|navbar|menu|header|footer|aside|sidebar|widget|breadcrumb|"
    r"cookie|consent|popup|modal|overlay|banner|subscribe|newsletter|"
    r"share|social|follow|comment|remark|related|recommend|tag|topic|"
    r"ads|advert|sponsor|promotion|promo|brand|login|register|"
    r"search|pagination|pager|toolbar|sticky|floating)",
    re.IGNORECASE
)

BOILERPLATE_LINE_RE = [
    re.compile(r"(^|\b)(cookie|consent|privacy|terms|นโยบาย|คุกกี้|ความเป็นส่วนตัว|ข้อกำหนด)\b", re.IGNORECASE),
    re.compile(r"(^|\b)(subscribe|newsletter|ติดตาม|สมัครรับข่าวสาร)\b", re.IGNORECASE),
    re.compile(r"(^|\b)(share|แชร์|follow us|ติดตามเรา)\b", re.IGNORECASE),
    re.compile(r"(^|\b)(log in|login|sign in|register|สมัครสมาชิก|เข้าสู่ระบบ)\b", re.IGNORECASE),
    re.compile(r"(^|\b)(advert|advertisement|sponsor|โฆษณา)\b", re.IGNORECASE),
]

IGNORE_IMAGE_KEYWORDS = [
    "logo", "icon", "sprite", "button", "spacer", "blank", "pixel",
    "avatar", "badge", "placeholder"
]

ALLOWED_FILE_CT = {
    "application/pdf": ".pdf",
    "application/octet-stream": ".bin",
    "application/zip": ".zip",
    "application/x-zip-compressed": ".zip",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
}

# ── ค่า threshold: ถ้า PyMuPDF extract ได้ text น้อยกว่านี้ต่อหน้า → fallback OCR
PDF_MIN_CHARS_PER_PAGE = 50


def extract_pdf_text(
    pdf_path: str,
    ocr_lang: str = "th",
    min_chars_per_page: int = PDF_MIN_CHARS_PER_PAGE,
) -> str:
    """
    Extract text จาก PDF โดย:
    1. ลอง PyMuPDF ก่อน (เร็ว เหมาะกับ digital PDF)
    2. ถ้าหน้าไหน text น้อยกว่า min_chars_per_page → แปลงเป็นภาพแล้วใช้ PaddleOCR

    Args:
        pdf_path: path ของไฟล์ PDF
        ocr_lang: ภาษาสำหรับ PaddleOCR เช่น "th", "en", "ch"
        min_chars_per_page: threshold ตัดสินว่าหน้าไหนต้อง OCR

    Returns:
        str: text ทั้งหมดจาก PDF (ทุกหน้า)
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed: pip install pymupdf")

    all_pages: List[str] = []
    ocr_model = None  # lazy init — สร้าง PaddleOCR เมื่อต้องใช้จริงเท่านั้น

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Cannot open PDF: {pdf_path} — {e}")

    for page_num in range(len(doc)):
        page = doc[page_num]

        # ── Step 1: PyMuPDF extract text ──────────────────────────────
        mupdf_text = (page.get_text("text") or "").strip()

        if len(mupdf_text) >= min_chars_per_page:
            # ✅ digital PDF — ใช้ text จาก PyMuPDF เลย
            all_pages.append(mupdf_text)
            continue

        # ── Step 2: น้อยกว่า threshold → OCR ────────────────────────
        try:
            if ocr_model is None:
                try:
                    from paddleocr import PaddleOCR
                except ImportError:
                    raise ImportError("PaddleOCR not installed: pip install paddleocr")
                # use_angle_cls=True ช่วยรองรับข้อความหมุน
                ocr_model = PaddleOCR(use_angle_cls=True, lang=ocr_lang, show_log=False)

            # แปลง page เป็นภาพ (300 dpi)
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")

            # PaddleOCR รับ numpy array หรือ bytes
            import numpy as np
            from PIL import Image as PILImage

            img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(img)

            result = ocr_model.ocr(img_np, cls=True)

            ocr_lines: List[str] = []
            if result:
                for line_group in result:
                    if not line_group:
                        continue
                    for line in line_group:
                        # line = [[box], [text, confidence]]
                        if line and len(line) >= 2:
                            txt = (line[1][0] or "").strip()
                            if txt:
                                ocr_lines.append(txt)

            ocr_text = "\n".join(ocr_lines).strip()
            all_pages.append(ocr_text if ocr_text else mupdf_text)

        except Exception as ocr_err:
            # OCR ล้มเหลว → ใช้ mupdf_text แม้จะสั้น
            all_pages.append(mupdf_text)

    doc.close()

    # รวมทุกหน้า คั่นด้วย newline สองชั้น
    return "\n\n".join(p for p in all_pages if p).strip()


def extract_pdfs_in_folder(
    files_dir: str,
    pdf_texts_dir: str,
    ocr_lang: str = "th",
) -> List[Dict[str, Any]]:
    """
    วน extract text จากทุก PDF ใน files_dir
    บันทึกผลเป็น .txt แต่ละไฟล์ใน pdf_texts_dir

    Returns:
        List of { "pdf_path", "txt_path", "char_count", "error" }
    """
    safe_mkdir(pdf_texts_dir)
    results: List[Dict[str, Any]] = []

    pdf_files = [
        f for f in os.listdir(files_dir)
        if f.lower().endswith(".pdf")
    ]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(files_dir, pdf_file)
        txt_filename = os.path.splitext(pdf_file)[0] + ".txt"
        txt_path = os.path.join(pdf_texts_dir, txt_filename)

        try:
            text = extract_pdf_text(pdf_path, ocr_lang=ocr_lang)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            results.append({
                "pdf_path": pdf_path,
                "txt_path": txt_path,
                "char_count": len(text),
                "error": None,
            })
        except Exception as e:
            results.append({
                "pdf_path": pdf_path,
                "txt_path": "",
                "char_count": 0,
                "error": str(e),
            })

    return results


# ============================================================
# Playwright Session
# ============================================================
class PlaywrightSession:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self._p = None
        self.context = None
        self.page = None

    def start(self):
        headless = bool(getattr(settings, "BROWSER_HEADLESS", False))
        slowmo = int(getattr(settings, "BROWSER_SLOWMO_MS", 0))
        channel = getattr(settings, "BROWSER_CHANNEL", None) or "chrome"

        user_data_dir = getattr(settings, "BROWSER_USER_DATA_DIR", None)
        if not user_data_dir:
            user_data_dir = os.path.join(
                getattr(settings, "OUTPUT_BASE_DIR", os.getcwd()), "chrome_user_data"
            )
        safe_mkdir(user_data_dir)

        self._p = sync_playwright().start()
        self.context = self._p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=headless,
            slow_mo=slowmo,
            channel=channel,
            args=["--start-maximized", "--disable-blink-features=AutomationControlled", "--no-sandbox"],
            viewport=None,
            locale="th-TH",
            user_agent=USER_AGENT,
            ignore_https_errors=True,
        )
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        return self

    def close(self):
        try:
            if self.context:
                self.context.close()
        except Exception:
            pass
        try:
            if self._p:
                self._p.stop()
        except Exception:
            pass


def _goto_safely(page, url: str, timeout_ms: int = 90_000) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    try:
        page.wait_for_load_state("networkidle", timeout=8_000)
    except PWTimeoutError:
        pass
    page.wait_for_timeout(1200)


# ----------------------------
# Utilities
# ----------------------------
def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def safe_filename(name: str, max_len: int = 60) -> str:
    name = unicodedata.normalize("NFKC", name or "")
    name = re.sub(r"[\\/*?\"<>|:]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name[:max_len].rstrip(" .").strip()
    return name if name else "site"


def is_probably_blocked(text: str) -> bool:
    t = (text or "").lower().strip()
    if any(k in t for k in BLOCK_KEYWORDS):
        return True
    return len(t) < 250


def _normalize_visible_text(raw: str, min_line_len: int = 10) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", raw)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[^\S\r\n]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)

    lines: List[str] = []
    seen = set()
    for line in s.splitlines():
        line = line.strip()
        if len(line) < min_line_len:
            continue
        key = re.sub(r"\s+", " ", line)
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return "\n".join(lines)


def content_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "canvas"]):
        tag.decompose()
    for tag_name in ["header", "footer", "nav", "aside"]:
        for t in soup.find_all(tag_name):
            t.decompose()
    main = soup.find("article") or soup.find("main") or soup.body
    text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)
    return normalize_text(text)


def extract_page_title(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    ogt = soup.find("meta", property="og:title")
    if ogt and ogt.get("content"):
        return ogt["content"].strip()
    t = soup.find("title")
    if t:
        return t.get_text(" ", strip=True)
    h1 = soup.find("h1")
    return h1.get_text(" ", strip=True) if h1 else ""


def _is_junk_node(tag) -> bool:
    try:
        cid = " ".join([tag.get("id") or "", " ".join(tag.get("class") or [])]).strip()
    except Exception:
        cid = ""
    return bool(cid and JUNK_CLASS_ID_RE.search(cid))


def _drop_junk_blocks(soup: BeautifulSoup) -> None:
    for t in soup(["script", "style", "noscript", "svg", "canvas", "iframe"]):
        t.decompose()
    for tag_name in ["header", "footer", "nav", "aside", "form"]:
        for t in soup.find_all(tag_name):
            t.decompose()
    for t in soup.find_all(True):
        if _is_junk_node(t):
            t.decompose()


def _get_text_and_link_len(node) -> tuple[int, int, int]:
    txt = node.get_text("\n", strip=True)
    text_len = len(txt.strip())
    link_text_len = sum(len(a.get_text(" ", strip=True)) for a in node.find_all("a"))
    blocks_count = len(node.find_all(["p", "h1", "h2", "h3", "h4", "li"]))
    return text_len, link_text_len, blocks_count


def _pick_best_main_container(soup: BeautifulSoup):
    candidates = []
    for n in soup.select("article, main, [role='main']"):
        if not _is_junk_node(n):
            candidates.append(n)
    for n in soup.select("section, div"):
        if not _is_junk_node(n):
            candidates.append(n)

    best, best_score = None, -1.0
    for n in candidates:
        try:
            tlen, llen, blocks = _get_text_and_link_len(n)
            if tlen < 400:
                continue
            ld = llen / max(tlen, 1)
            score = max(tlen - llen, 0) * (1.0 - min(ld, 0.95)) * (1.0 + min(blocks, 60) / 12.0)
            if ld > 0.55:
                score *= 0.25
            if score > best_score:
                best_score = score
                best = n
        except Exception:
            continue
    return best or (soup.find("article") or soup.find("main") or soup.body or soup)


def _drop_boilerplate_lines(text: str) -> str:
    if not text:
        return ""
    lines = [ln.strip() for ln in text.splitlines()]
    out, seen = [], set()
    for ln in lines:
        if len(ln) < 8:
            continue
        if any(rx.search(ln) for rx in BOILERPLATE_LINE_RE):
            continue
        words = re.split(r"\s+", ln)
        if len(words) >= 6 and sum(1 for w in words if len(w) <= 4) / max(len(words), 1) > 0.65:
            continue
        key = re.sub(r"\s+", " ", ln).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(ln)
    return "\n".join(out)


def content_main_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    _drop_junk_blocks(soup)
    main = _pick_best_main_container(soup)
    text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)
    return _drop_boilerplate_lines(normalize_text(text)).strip()


# ----------------------------
# Files & Images
# ----------------------------
def looks_like_file_url(u: str) -> bool:
    ul = (u or "").lower()
    return any(x in ul for x in [".pdf", "getmedia", "download", "attachment", "file", "/getmedia/", ".pdf.aspx"])


def extract_file_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html or "", "html.parser")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        full = urljoin(base_url, a.get("href"))
        if full.startswith("http") and looks_like_file_url(full):
            links.append(full)
    for tag in soup.select("[onclick]"):
        onclick = tag.get("onclick") or ""
        m = re.search(r"""['"]([^'"]+)['"]""", onclick)
        if m:
            full = urljoin(base_url, m.group(1))
            if full.startswith("http") and looks_like_file_url(full):
                links.append(full)
    uniq, seen = [], set()
    for u in links:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def guess_ext_from_ct(ct: str) -> str:
    ct = (ct or "").split(";")[0].strip().lower()
    return ALLOWED_FILE_CT.get(ct, ".pdf" if "pdf" in ct else ".bin")


def download_files_via_requests(urls: List[str], save_folder: str, base_url: str, max_files: int = 50) -> int:
    safe_mkdir(save_folder)
    count = 0
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT, "Referer": base_url})
    for u in urls:
        if count >= max_files:
            break
        try:
            resp = sess.get(u, timeout=30, stream=True)
            if not resp.ok:
                continue
            ct = (resp.headers.get("content-type") or "").lower()
            if "text/html" in ct:
                continue
            data = resp.content
            if not data or len(data) < 10 * 1024:
                continue
            ext = guess_ext_from_ct(ct)
            outp = os.path.join(save_folder, f"file_{count+1}_{random.randint(100,999)}{ext}")
            with open(outp, "wb") as f:
                f.write(data)
            count += 1
        except Exception:
            continue
    return count


def _pick_best_from_srcset(srcset: str) -> str:
    if not srcset:
        return ""
    candidates = []
    for p in [p.strip() for p in srcset.split(",") if p.strip()]:
        toks = p.split()
        if not toks:
            continue
        score = 0.0
        if len(toks) >= 2:
            s = toks[1].strip().lower()
            try:
                score = float(s[:-1]) if s.endswith("w") else float(s[:-1]) * 10000.0 if s.endswith("x") else 0.0
            except Exception:
                pass
        candidates.append((score, toks[0].strip()))
    if not candidates:
        return ""
    return sorted(candidates, reverse=True)[0][1]


def _extract_bg_image_urls(style_text: str) -> List[str]:
    if not style_text:
        return []
    return [u.strip() for u in re.findall(r'url\(\s*["\']?([^"\')]+)["\']?\s*\)', style_text, re.IGNORECASE) if u.strip()]


def extract_image_candidates(html: str, base_url: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html or "", "html.parser")
    items: List[Dict[str, Any]] = []

    def get_nearby_text(img_tag) -> str:
        texts = []
        fig = img_tag.find_parent("figure")
        if fig:
            cap = fig.find("figcaption")
            if cap:
                texts.append(cap.get_text(" ", strip=True))
            texts.append(fig.get_text(" ", strip=True))
        if img_tag.parent:
            texts.append(img_tag.parent.get_text(" ", strip=True))
        for fn in [img_tag.find_previous, img_tag.find_next]:
            el = fn(["p", "h1", "h2", "h3", "h4", "li"])
            if el:
                texts.append(el.get_text(" ", strip=True))
        return normalize_text("\n".join(t for t in texts if t))[:1200]

    def push(u: Optional[str], meta: Dict[str, Any]):
        if not u:
            return
        u = u.strip()
        full = urljoin(base_url, u)
        if not (full.startswith("http") or full.startswith("data:")):
            return
        items.append({**meta, "img_url": full})

    og = soup.find("meta", property="og:image")
    tw = soup.find("meta", attrs={"name": "twitter:image"})
    push(og.get("content") if og else None, {"alt": "og:image", "title": "", "nearby_text": ""})
    push(tw.get("content") if tw else None, {"alt": "twitter:image", "title": "", "nearby_text": ""})

    container = soup.find("article") or soup.find("main") or soup.body or soup
    for img in container.find_all("img"):
        alt = (img.get("alt") or "").strip()
        title = (img.get("title") or "").strip()
        nearby = get_nearby_text(img)
        push(_pick_best_from_srcset(img.get("srcset") or img.get("data-srcset") or ""), {"alt": alt, "title": title, "nearby_text": nearby})
        for key in ["src", "data-src", "data-lazy-src", "data-original", "data-url", "data-img-url"]:
            push(img.get(key), {"alt": alt, "title": title, "nearby_text": nearby})

    for pic in container.find_all("picture"):
        for s in pic.find_all("source"):
            push(_pick_best_from_srcset(s.get("srcset") or s.get("data-srcset") or ""), {"alt": "picture", "title": "", "nearby_text": ""})
        im = pic.find("img")
        if im:
            alt = (im.get("alt") or "").strip()
            nearby = get_nearby_text(im)
            push(_pick_best_from_srcset(im.get("srcset") or im.get("data-srcset") or ""), {"alt": alt, "title": "", "nearby_text": nearby})
            for key in ["src", "data-src", "data-lazy-src", "data-original"]:
                push(im.get(key), {"alt": alt, "title": "", "nearby_text": nearby})

    for tag in container.find_all(style=True):
        for u in _extract_bg_image_urls(tag.get("style") or ""):
            push(u, {"alt": "background-image", "title": "", "nearby_text": ""})

    uniq, seen = [], set()
    for it in items:
        u = it.get("img_url")
        if u and u not in seen:
            seen.add(u)
            uniq.append(it)
    return uniq


def is_good_image_url(url: str) -> bool:
    ul = (url or "").lower().strip()
    return bool(ul) and not any(k in ul for k in IGNORE_IMAGE_KEYWORDS)


def download_images(
    session: requests.Session,
    image_items: List[Dict[str, Any]],
    save_folder: str,
    max_images: Optional[int] = None,
    referer: str = ""
) -> List[Dict[str, Any]]:
    safe_mkdir(save_folder)
    saved: List[Dict[str, Any]] = []
    seen_sha1 = set()
    base_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "Accept-Language": "th-TH,th;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    }
    if referer:
        base_headers["Referer"] = referer

    for it in image_items:
        if max_images is not None and len(saved) >= max_images:
            break
        u = (it.get("img_url") or "").strip()
        if not u or not is_good_image_url(u):
            continue
        try:
            data, ct, w, h, ext = b"", "", 0, 0, "jpg"
            if u.startswith("data:"):
                m = re.match(r"data:([^;]+);base64,(.+)$", u, re.IGNORECASE | re.DOTALL)
                if not m:
                    continue
                ct = m.group(1).lower().strip()
                data = base64.b64decode(m.group(2))
            else:
                r = session.get(u, headers=base_headers, timeout=30, allow_redirects=True)
                if not r or r.status_code != 200:
                    continue
                ct = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
                data = r.content or b""
            if not data:
                continue
            sha1 = _sha1_bytes(data)
            if sha1 in seen_sha1:
                continue
            ext = "svg" if "svg" in ct or u.lower().endswith(".svg") else \
                  "png" if "png" in ct or u.lower().endswith(".png") else \
                  "webp" if "webp" in ct or u.lower().endswith(".webp") else \
                  "gif" if "gif" in ct or u.lower().endswith(".gif") else "jpg"
            if ext != "svg":
                try:
                    img = Image.open(io.BytesIO(data))
                    img = img.convert("RGBA") if img.mode in ("P", "LA") else img.convert("RGB")
                    w, h = img.size
                except Exception:
                    pass
            seen_sha1.add(sha1)
            outp = os.path.join(save_folder, f"img_{len(saved)+1}_{random.randint(100,999)}.{ext}")
            with open(outp, "wb") as f:
                f.write(data)
            saved.append({
                "saved_path": outp, "sha1": sha1, "source_url": u,
                "content_type": ct, "width": w, "height": h,
                "alt_text": (it.get("alt") or "").strip(),
                "title_text": (it.get("title") or "").strip(),
                "nearby_text": (it.get("nearby_text") or "").strip(),
            })
        except Exception:
            continue
    return saved


# ============================================================
# Google Search
# ============================================================
def _normalize_google_href(href: str) -> str:
    if not href:
        return ""
    if href.startswith("/url?"):
        return unquote(parse_qs(urlparse(href).query).get("q", [""])[0])
    return href


def _wait_any(page, selectors: List[str], timeout_ms: int = 45_000) -> str:
    start = time.time()
    while (time.time() - start) * 1000 < timeout_ms:
        for sel in selectors:
            try:
                if page.locator(sel).count() > 0:
                    return sel
            except Exception:
                pass
        page.wait_for_timeout(300)
    return ""


def _try_accept_google_consent(page) -> None:
    try:
        for txt in ["I agree", "Accept all", "Agree", "ยอมรับทั้งหมด", "ยอมรับ", "ยืนยัน", "ตกลง"]:
            btn = page.locator(f"button:has-text('{txt}')")
            if btn.count() > 0:
                btn.first.click(timeout=2000)
                page.wait_for_timeout(800)
                break
    except Exception:
        pass


def google_search_targets(job_id: str, page, keyword: str, max_links: int) -> List[Dict[str, str]]:
    log(job_id, "info", f"Google search targets: {keyword} (max_links={max_links})")
    page.goto("https://www.google.com/", wait_until="domcontentloaded", timeout=60_000)
    _try_accept_google_consent(page)

    box = page.locator("textarea[name='q'], input[name='q']").first
    box.click()
    box.fill(keyword)
    page.keyboard.press("Enter")

    try:
        page.wait_for_load_state("domcontentloaded", timeout=60_000)
    except Exception:
        pass
    try:
        page.wait_for_load_state("networkidle", timeout=8_000)
    except Exception:
        pass

    ok = _wait_any(page, ["#search a:has(h3)", "a:has(h3)", 'a[data-ved]:has(h3)'], timeout_ms=45_000)
    if not ok:
        log(job_id, "warn", "Google results selector not found.")
        return []

    items: List[Dict[str, str]] = []
    seen = set()

    def push(u: str, t: str):
        if not u.startswith("http") or "google.com" in urlparse(u).netloc:
            return
        key = u.split("#")[0].strip()
        if not key or key in seen:
            return
        seen.add(key)
        items.append({"url": u, "title": (t or u).strip()})

    anchors = page.locator("#search a:has(h3)")
    if anchors.count() == 0:
        anchors = page.locator("a:has(h3)")

    for i in range(min(anchors.count(), max_links * 8)):
        try:
            a = anchors.nth(i)
            u = _normalize_google_href(a.get_attribute("href") or "")
            t = (a.locator("h3").first.inner_text() or "").strip()
            if u.startswith("http") and "google.com" not in urlparse(u).netloc:
                push(u, t)
            if len(items) >= max_links:
                break
        except Exception:
            continue

    log(job_id, "info", f"Google targets found: {len(items)}")
    return items


# ============================================================
# Scrape one URL — บันทึกแค่ content.txt
# semantic chunking ทำใน RAG ingestion
# ============================================================
def scrape_single_url(
    job_id: str,
    url: str,
    folder: str,
    pw_page,
    pw_context,
    max_images: Optional[int] = None,
    max_files: int = 30,
) -> Dict[str, Any]:
    safe_mkdir(folder)
    images_dir = os.path.join(folder, "images")
    files_dir = os.path.join(folder, "files")
    safe_mkdir(images_dir)
    safe_mkdir(files_dir)

    html, final_url, title, downloaded_files = "", url, "", 0

    try:
        log(job_id, "info", f"Scraping (Playwright reuse): {url}")
        _goto_safely(pw_page, url, timeout_ms=90_000)

        try:
            if pw_page.locator('iframe[title="reCAPTCHA"]').count() > 0:
                log(job_id, "warn", "Captcha detected! Attempting solve...")
                solver = RecaptchaSolver(pw_page)
                if solver.solveCaptcha(max_retries=3):
                    try:
                        pw_page.wait_for_load_state("networkidle", timeout=10_000)
                    except PWTimeoutError:
                        pass
                    time.sleep(2.0)
                else:
                    log(job_id, "error", "Failed to solve Captcha.")
        except Exception as e:
            log(job_id, "warn", f"captcha check error: {e}")

        final_url = pw_page.url
        title = pw_page.title() or ""
        try:
            html = pw_page.content() or ""
        except Exception:
            html = ""

        raw_visible, sel_used = "", "none"
        try:
            for sel in ["article", "main", "[role='main']", "article [itemprop='articleBody']",
                        ".entry-content", ".post-content", ".article-content", ".content"]:
                loc = pw_page.locator(sel)
                if loc.count() > 0:
                    raw_visible = loc.first.inner_text(timeout=4000) or ""
                    sel_used = sel
                    break
        except Exception:
            sel_used = "selector_error"

        if not raw_visible.strip():
            try:
                raw_visible = pw_page.inner_text("body") or ""
                sel_used = "body"
            except Exception:
                sel_used = "body_error"

        visible_text = _normalize_visible_text(raw_visible, min_line_len=10)

        if is_probably_blocked(visible_text):
            log(job_id, "warn", f"Page looks blocked. Reload once... (sel={sel_used})")
            try:
                pw_page.reload(wait_until="domcontentloaded", timeout=45_000)
                try:
                    pw_page.wait_for_load_state("networkidle", timeout=8_000)
                except PWTimeoutError:
                    pass
                pw_page.wait_for_timeout(1200)
                try:
                    html = pw_page.content() or html
                except Exception:
                    pass
                raw2, sel2 = "", "none"
                try:
                    for sel in ["article", "main", "[role='main']"]:
                        loc = pw_page.locator(sel)
                        if loc.count() > 0:
                            raw2 = loc.first.inner_text(timeout=4000) or ""
                            sel2 = sel
                            break
                except Exception:
                    sel2 = "selector_error"
                if raw2.strip():
                    visible_text = _normalize_visible_text(raw2, min_line_len=10)
                    sel_used = sel2
                else:
                    try:
                        visible_text = _normalize_visible_text(pw_page.inner_text("body") or "", min_line_len=10)
                        sel_used = "body_after_reload"
                    except Exception:
                        pass
            except Exception:
                pass

    except Exception as e:
        log(job_id, "warn", f"Playwright reuse failed: {e}")
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if r.status_code == 200 and r.text.strip():
                final_url, html = r.url, r.text
        except Exception as ee:
            log(job_id, "error", f"requests fallback failed: {ee}")
        visible_text = content_text_from_html(html)

    if not title:
        title = extract_page_title(html)

    # เลือก text ที่ดีที่สุดจาก 3 candidates
    candidates = [
        ("main_html", content_main_text_from_html(html)),
        ("visible", visible_text),
        ("html_basic", content_text_from_html(html)),
    ]
    best_name, text, best_len = "", "", 0
    for name, txt in candidates:
        if len((txt or "").strip()) > best_len:
            best_len = len(txt.strip())
            text = txt.strip()
            best_name = name

    if len(text) < 250 and visible_text.strip():
        text = visible_text.strip()
        best_name = "visible_fallback"

    text2 = _drop_boilerplate_lines(text)
    if len(text2) >= 200:
        text = text2

    log(job_id, "info", f"Selected content source: {best_name}", {"len": len(text)})

    # ✅ บันทึกเฉพาะ content.txt — semantic chunking ทำตอน ingest
    content_path = os.path.join(folder, "content.txt")
    with open(content_path, "w", encoding="utf-8") as f:
        f.write(text or "")

    # Download files
    try:
        pdf_links = [u for u in extract_file_links(html, final_url) if ".pdf" in (u or "").lower()]
        downloaded_files = download_files_via_requests(pdf_links, files_dir, base_url=final_url, max_files=1)
    except Exception:
        downloaded_files = 0

    # ✅ Extract text จาก PDF ที่ดาวน์โหลดมา บันทึกไว้ใน pdf_texts/
    pdf_texts_dir = os.path.join(folder, "pdf_texts")
    pdf_extract_results: List[Dict[str, Any]] = []
    if downloaded_files > 0:
        try:
            pdf_extract_results = extract_pdfs_in_folder(
                files_dir=files_dir,
                pdf_texts_dir=pdf_texts_dir,
                ocr_lang="th",
            )
            log(job_id, "info", f"PDF extraction: {len(pdf_extract_results)} files", {
                "results": [{
                    "pdf": r["pdf_path"],
                    "chars": r["char_count"],
                    "error": r["error"],
                } for r in pdf_extract_results]
            })
        except Exception as e:
            log(job_id, "warn", f"PDF extraction failed: {e}")

    # Download images
    images_download_meta_jsonl = os.path.join(images_dir, "images_download_meta.jsonl")
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    saved_imgs = download_images(
        session=sess,
        image_items=extract_image_candidates(html, final_url),
        save_folder=images_dir,
        max_images=max_images,
        referer=final_url,
    )
    for it in saved_imgs:
        append_jsonl(images_download_meta_jsonl, {
            "type": "web_image_download",
            "page_url": final_url,
            "page_title": title,
            **{k: it.get(k, "") for k in ["source_url", "saved_path", "sha1", "alt_text", "title_text", "nearby_text"]},
            "width": it.get("width", 0),
            "height": it.get("height", 0),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    log(job_id, "info", f"Scrape finished: {final_url}", {
        "title": title, "content_len": len(text or ""),
        "images": len(saved_imgs), "files": downloaded_files,
    })

    try:
        with open(os.path.join(folder, "page_meta.json"), "w", encoding="utf-8") as mf:
            json.dump({"source_url": final_url, "title": title,
                       "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                      mf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return {
        "URL": final_url,
        "Title": title,
        "Folder": folder,
        "content_path": content_path,
        "images_dir": images_dir,
        "files_dir": files_dir,
        "pdf_texts_dir": pdf_texts_dir,                      # ✅ เพิ่ม
        "pdf_extracted": len(pdf_extract_results),           # ✅ เพิ่ม
        "images_download_meta_jsonl": images_download_meta_jsonl,
        "downloaded_images": len(saved_imgs),
        "downloaded_files": downloaded_files,
    }


# ============================================================
# Main entry
# ============================================================
def run_external_scrape(job_id: str, keyword: str, max_links: int = 5) -> Dict[str, Any]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_kw = re.sub(r"[^a-zA-Z0-9ก-๙_]+", "_", keyword)[:40]
    main_folder = os.path.join(settings.OUTPUT_BASE_DIR, f"data_{safe_kw}_{ts}")
    safe_mkdir(main_folder)
    collected: List[Dict[str, Any]] = []

    pw = PlaywrightSession(job_id).start()
    try:
        targets = google_search_targets(job_id, pw.page, keyword, max_links=max_links)
        log(job_id, "info", f"Search URLs => {len(targets)}", {"urls": [t['url'] for t in targets]})

        for idx, t in enumerate(targets, start=1):
            u = t.get("url", "")
            site_folder = os.path.join(main_folder, f"{idx:02d}_{safe_filename(urlparse(u).netloc, 40)}")
            safe_mkdir(site_folder)
            try:
                res = scrape_single_url(
                    job_id=job_id, url=u, folder=site_folder,
                    pw_page=pw.page, pw_context=pw.context,
                    max_images=None, max_files=30,
                )
                res.update({"Source": "google", "rank": idx, "search_title": t.get("title", "")})
                collected.append(res)
            except Exception as e:
                log(job_id, "error", f"Failed to scrape {u}: {e}", {"traceback": traceback.format_exc()})
            time.sleep(0.8 + random.uniform(0.2, 0.6))
    finally:
        pw.close()

    csv_path = os.path.join(main_folder, "final_data.csv")
    if collected:
        pd.DataFrame(collected).to_csv(csv_path, index=False, encoding="utf-8-sig")

    return {
        "main_folder": main_folder,
        "csv_path": csv_path,
        "items": collected,
        "urls": [x.get("URL") for x in collected],
        "keyword": keyword,
    }