# backend/scraping/web_scraping.py
from __future__ import annotations

import os
import re
import io
import time
import random
import hashlib
import traceback
import unicodedata
import base64
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, quote, urlparse, parse_qs, unquote, urlsplit

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

def _pick_best_from_srcset(srcset: str) -> str:
    """
    เลือก URL ที่ 'ใหญ่สุด' จาก srcset แบบง่าย ๆ
    รูปแบบ: "url1 320w, url2 640w" หรือ "url1 1x, url2 2x"
    """
    if not srcset:
        return ""
    parts = [p.strip() for p in srcset.split(",") if p.strip()]
    candidates = []
    for p in parts:
        toks = p.split()
        if not toks:
            continue
        url = toks[0].strip()
        score = 0.0
        if len(toks) >= 2:
            s = toks[1].strip().lower()
            try:
                if s.endswith("w"):
                    score = float(s[:-1])
                elif s.endswith("x"):
                    score = float(s[:-1]) * 10000.0
            except Exception:
                score = 0.0
        candidates.append((score, url))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _extract_bg_image_urls(style_text: str) -> List[str]:
    """
    ดึง url(...) จาก style background-image
    """
    if not style_text:
        return []
    # รองรับทั้ง url("...") url('...') url(...)
    urls = re.findall(r'url\(\s*["\']?([^"\')]+)["\']?\s*\)', style_text, flags=re.IGNORECASE)
    return [u.strip() for u in urls if u.strip()]


def requests_get_with_retries(
    session: requests.Session,
    url: str,
    headers: Dict[str, str],
    timeout: int = 30,
    max_retries: int = 3
) -> Optional[requests.Response]:
    """
    GET แบบมี retry/backoff สำหรับ 403/429/5xx/timeout
    """
    last_exc = None
    for i in range(max_retries):
        try:
            r = session.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=False)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.0 + (i * 1.5) + random.uniform(0.0, 0.5))
                continue
            if r.status_code == 403:
                # บางเว็บกัน hotlink: ลองพักแล้วค่อย retry
                time.sleep(0.8 + random.uniform(0.0, 0.6))
                continue
            return r
        except Exception as e:
            last_exc = e
            time.sleep(0.8 + (i * 1.2) + random.uniform(0.0, 0.4))
    return None


def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def safe_filename(name: str, max_len: int = 60) -> str:
    name = (name or "")
    name = unicodedata.normalize("NFKC", name)
    name = re.sub(r"[\\/*?\"<>|:]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name[:max_len].rstrip(" .").strip()
    return name if name else "site"


def is_probably_blocked(text: str) -> bool:
    """
    Heuristic ไม่ aggressive:
    - ถ้ามี keyword captcha/denied ชัดเจน -> blocked
    - ถ้า text สั้นมาก ไม่สรุปว่า blocked เสมอ (บางเพจสั้นจริง)
    """
    t = (text or "").lower().strip()
    if any(k in t for k in BLOCK_KEYWORDS):
        return True
    return False


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
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
    return uniq


def guess_ext_from_ct(ct: str) -> str:
    ct = (ct or "").split(";")[0].strip().lower()
    return ALLOWED_FILE_CT.get(ct, ".pdf" if "pdf" in ct else ".bin")


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

        parent = img_tag.parent
        if parent:
            texts.append(parent.get_text(" ", strip=True))

        prev = img_tag.find_previous(["p", "h1", "h2", "h3", "h4", "li"])
        nxt = img_tag.find_next(["p", "h1", "h2", "h3", "h4", "li"])
        if prev:
            texts.append(prev.get_text(" ", strip=True))
        if nxt:
            texts.append(nxt.get_text(" ", strip=True))

        merged = normalize_text("\n".join([t for t in texts if t]))
        return merged[:1200]

    def push(u: Optional[str], meta: Dict[str, Any]):
        if not u:
            return
        u = (u or "").strip()
        if not u:
            return
        full = urljoin(base_url, u)
        # data:image/... ก็ถือว่าเป็นรูป (โหลดจาก base64)
        if not (full.startswith("http") or full.startswith("data:")):
            return
        m = dict(meta or {})
        m["img_url"] = full
        items.append(m)

    # og/twitter
    og = soup.find("meta", property="og:image")
    tw = soup.find("meta", attrs={"name": "twitter:image"})
    push(og.get("content") if og else None, {"alt": "og:image", "title": "", "nearby_text": ""})
    push(tw.get("content") if tw else None, {"alt": "twitter:image", "title": "", "nearby_text": ""})

    container = soup.find("article") or soup.find("main") or soup.body or soup

    # 1) <img ...>
    for img in container.find_all("img"):
        alt = (img.get("alt") or "").strip()
        title = (img.get("title") or "").strip()
        nearby_text = get_nearby_text(img)

        # srcset priority
        srcset = img.get("srcset") or img.get("data-srcset") or ""
        best = _pick_best_from_srcset(srcset)
        push(best, {"alt": alt, "title": title, "nearby_text": nearby_text})

        # common lazy attrs
        for key in ["src", "data-src", "data-lazy-src", "data-original", "data-url", "data-img-url"]:
            push(img.get(key), {"alt": alt, "title": title, "nearby_text": nearby_text})

    # 2) <picture><source srcset=...>
    for pic in container.find_all("picture"):
        sources = pic.find_all("source")
        for s in sources:
            srcset = s.get("srcset") or s.get("data-srcset") or ""
            best = _pick_best_from_srcset(srcset)
            push(best, {"alt": "picture", "title": "", "nearby_text": ""})
        # fallback img in picture
        im = pic.find("img")
        if im:
            srcset = im.get("srcset") or im.get("data-srcset") or ""
            best = _pick_best_from_srcset(srcset)
            push(best, {"alt": (im.get("alt") or "").strip(), "title": "", "nearby_text": get_nearby_text(im)})
            for key in ["src", "data-src", "data-lazy-src", "data-original"]:
                push(im.get(key), {"alt": (im.get("alt") or "").strip(), "title": "", "nearby_text": get_nearby_text(im)})

    # 3) background-image ใน style
    for tag in container.find_all(style=True):
        style_text = tag.get("style") or ""
        for u in _extract_bg_image_urls(style_text):
            push(u, {"alt": "background-image", "title": "", "nearby_text": ""})

    # uniq
    uniq, seen = [], set()
    for it in items:
        u = it.get("img_url")
        if not u or u in seen:
            continue
        seen.add(u)
        uniq.append(it)
    return uniq


def is_good_image_url(url: str) -> bool:
    ul = (url or "").lower().strip()
    # ยอม data: ได้ (base64)
    if not ul:
        return False
    # ไม่ตัด svg แล้ว (โหลดเก็บได้เป็น .svg)
    return True



def download_images(
    session: requests.Session,
    image_items: List[Dict[str, Any]],
    save_folder: str,
    max_images: Optional[int] = None,   # ✅ None = ไม่จำกัด
    referer: str = ""
) -> List[Dict[str, Any]]:
    """
    ดาวน์โหลดรูปจาก candidates แล้วบันทึกลง images/
    ✅ โหลดให้ได้มากที่สุด:
    - ไม่ตัดรูปเล็ก
    - รองรับ srcset/picture/bg (มาจาก extractor)
    - รองรับ data:image/... (base64)
    - รองรับ svg (save เป็น .svg)
    - retry/backoff เมื่อ 403/429/5xx/timeout
    - กันซ้ำด้วย sha1
    """
    safe_mkdir(save_folder)
    saved: List[Dict[str, Any]] = []
    seen_sha1 = set()

    # headers base
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
            data = b""
            ct = ""
            w = 0
            h = 0
            ext = "jpg"

            # --- data URL ---
            if u.startswith("data:"):
                # data:image/png;base64,....
                m = re.match(r"data:([^;]+);base64,(.+)$", u, flags=re.IGNORECASE | re.DOTALL)
                if not m:
                    continue
                ct = (m.group(1) or "").lower().strip()
                b64 = m.group(2) or ""
                data = base64.b64decode(b64)
                if not data:
                    continue

            else:
                r = requests_get_with_retries(session, u, headers=base_headers, timeout=30, max_retries=3)
                if not r or r.status_code != 200:
                    continue

                ct = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
                data = r.content or b""
                if not data:
                    continue

            sha1 = _sha1_bytes(data)
            if sha1 in seen_sha1:
                continue

            # --- decide ext + try read size if raster ---
            if "svg" in ct or u.lower().endswith(".svg"):
                ext = "svg"
                # w/h ไม่ทราบสำหรับ svg
            elif "png" in ct or u.lower().endswith(".png"):
                ext = "png"
            elif "webp" in ct or u.lower().endswith(".webp"):
                ext = "webp"
            elif "gif" in ct or u.lower().endswith(".gif"):
                ext = "gif"
            else:
                ext = "jpg"

            if ext != "svg":
                try:
                    img = Image.open(io.BytesIO(data))
                    # บางไฟล์มี alpha
                    img = img.convert("RGBA") if img.mode in ("P", "LA") else img.convert("RGB")
                    w, h = img.size
                except Exception:
                    # ถ้าเปิดด้วย PIL ไม่ได้ แต่เป็น image/* ก็ยังเซฟไว้ก่อน
                    pass

            seen_sha1.add(sha1)

            outp = os.path.join(save_folder, f"img_{len(saved)+1}_{random.randint(100,999)}.{ext}")
            with open(outp, "wb") as f:
                f.write(data)

            saved.append({
                "saved_path": outp,
                "sha1": sha1,
                "source_url": u,
                "content_type": ct,
                "width": w,
                "height": h,
                "alt_text": (it.get("alt") or "").strip(),
                "title_text": (it.get("title") or "").strip(),
                "nearby_text": (it.get("nearby_text") or "").strip(),
            })

        except Exception:
            continue

    return saved



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


# ----------------------------
# Search helpers (FIXED)
# ----------------------------
def _ddg_decode_result_href(href: str) -> str:
    """
    DDG /html มักเป็น:
    - /l/?uddg=<encoded>
    - //duckduckgo.com/l/?uddg=<encoded>
    - https://duckduckgo.com/l/?uddg=<encoded>
    """
    if not href:
        return ""
    href = href.strip()

    if href.startswith("//"):
        href = "https:" + href

    if "duckduckgo.com/l/" in href or href.startswith("/l/"):
        if href.startswith("/l/"):
            href = "https://duckduckgo.com" + href
        try:
            qs = parse_qs(urlparse(href).query)
            uddg = (qs.get("uddg") or [""])[0]
            if uddg:
                return unquote(uddg)
        except Exception:
            return ""
        return ""
    return href if href.startswith("http") else ""


def duckduckgo_html_search(job_id: str, keyword: str, max_links: int = 5) -> List[str]:
    # ใช้ html.duckduckgo.com จะนิ่งกว่า duckduckgo.com/html
    url = f"https://html.duckduckgo.com/html/?q={quote(keyword)}"
    log(job_id, "info", f"DuckDuckGo search: {url}")

    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "th-TH,th;q=0.9,en;q=0.8",
        "Referer": "https://duckduckgo.com/",
    }

    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200 or not r.text.strip():
            return []

        soup = BeautifulSoup(r.text, "html.parser")

        raw_links: List[str] = []
        for a in soup.select("a.result__a[href], a[href].result__a"):
            raw_links.append(a.get("href") or "")

        # fallback selector เผื่อ class เปลี่ยน
        if not raw_links:
            for a in soup.select("a[href]"):
                href = a.get("href") or ""
                if "uddg=" in href or "duckduckgo.com/l/" in href:
                    raw_links.append(href)

        links: List[str] = []
        for href in raw_links:
            u = _ddg_decode_result_href(href)
            if u and u.startswith("http"):
                links.append(u)

        uniq, seen = [], set()
        for u in links:
            if u in seen:
                continue
            seen.add(u)
            uniq.append(u)

        return uniq[:max_links]
    except Exception:
        return []


def bing_html_search(job_id: str, keyword: str, max_links: int = 5) -> List[str]:
    url = f"https://www.bing.com/search?q={quote(keyword)}&count={max_links + 5}"
    log(job_id, "info", f"Bing search: {url}")

    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "th-TH,th;q=0.9,en;q=0.8",
        "Referer": "https://www.bing.com/",
    }

    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200 or not r.text.strip():
            return []

        soup = BeautifulSoup(r.text, "html.parser")
        links: List[str] = []
        for a in soup.select("li.b_algo h2 a[href]"):
            href = (a.get("href") or "").strip()
            if href.startswith("http"):
                links.append(href)

        uniq, seen = [], set()
        for u in links:
            if u in seen:
                continue
            seen.add(u)
            uniq.append(u)
        return uniq[:max_links]
    except Exception:
        return []


def _google_serp_parse_urls(html: str) -> List[str]:
    soup = BeautifulSoup(html or "", "html.parser")
    out: List[str] = []

    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        # google classic
        if href.startswith("/url?") and ("q=" in href or "url=" in href):
            qs = parse_qs(urlparse(href).query)
            q = (qs.get("q") or qs.get("url") or [""])[0]
            q = unquote(q)
            if q.startswith("http"):
                out.append(q)

    uniq, seen = [], set()
    for u in out:
        if u in seen:
            continue
        if "google.com" in urlparse(u).netloc:
            continue
        seen.add(u)
        uniq.append(u)
    return uniq


def google_search_playwright(job_id: str, keyword: str, max_links: int = 5) -> List[str]:
    qurl = f"https://www.google.com/search?q={quote(keyword)}&num={max_links + 6}"
    log(job_id, "info", f"Google search (Playwright): {qurl}")

    headless = bool(getattr(settings, "BROWSER_HEADLESS", False))
    slowmo = int(getattr(settings, "BROWSER_SLOWMO_MS", 0))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, slow_mo=slowmo)
        ctx = browser.new_context(user_agent=USER_AGENT, locale="th-TH")
        page = ctx.new_page()
        try:
            page.goto(qurl, timeout=60_000)

            # Handle consent pages (best-effort)
            try:
                # เผื่อเจอหน้าขอ consent
                for txt in ["I agree", "Accept all", "Agree", "ยอมรับทั้งหมด", "ยอมรับ", "ยืนยัน"]:
                    btn = page.locator(f"button:has-text('{txt}')")
                    if btn.count() > 0:
                        btn.first.click(timeout=2_000)
                        break
            except Exception:
                pass

            try:
                page.wait_for_load_state("networkidle", timeout=7_000)
            except PWTimeoutError:
                pass

            time.sleep(random.uniform(1.0, 1.8))
            html = page.content()
        finally:
            ctx.close()
            browser.close()

    urls = _google_serp_parse_urls(html)
    if urls:
        log(job_id, "info", f"Google Playwright found {len(urls)} candidates")
    return urls[:max_links]


def search_urls(job_id: str, keyword: str, max_links: int = 5) -> List[str]:
    """
    ✅ แก้ให้ “ได้ URL จริง” ก่อน:
    1) DuckDuckGo HTML (decode /l/?uddg= แล้ว)
    2) Bing HTML fallback
    3) Google Playwright (ไว้ท้ายสุด เพราะโดน consent/captcha บ่อย)
    """
    urls = duckduckgo_html_search(job_id, keyword, max_links=max_links)
    if urls:
        log(job_id, "info", f"DDG found {len(urls)} urls")
        return urls

    urls = bing_html_search(job_id, keyword, max_links=max_links)
    if urls:
        log(job_id, "info", f"Bing found {len(urls)} urls")
        return urls

    urls = google_search_playwright(job_id, keyword, max_links=max_links)
    if urls:
        return urls

    log(job_id, "warn", "No URLs found from DDG/Bing/Google.")
    return []


# ----------------------------
# Scrape one URL
# ----------------------------
def scrape_single_url(
    job_id: str,
    url: str,
    folder: str,
    max_images: Optional[int] = None,  # ✅ โหลดไม่จำกัด
    max_files: int = 30
) -> Dict[str, Any]:
    """
    Scrape 1 URL:
    - ใช้งาน Playwright เป็นหลัก (เพราะ DrissionPage มีปัญหาและ Solver เป็น Playwright)
    - รองรับ Recaptcha Solver
    - Fallback ไป requests
    """
    safe_mkdir(folder)
    images_dir = os.path.join(folder, "images")
    files_dir = os.path.join(folder, "files")
    safe_mkdir(images_dir)
    safe_mkdir(files_dir)

    html = ""
    final_url = url
    downloaded_files = 0
    title = ""

    # NOTE: ปิด DrissionPage ไปเลยตาม requirement เพื่อใช้ Playwright + Solver ใหม่
        
    # 1) Playwright (Primary Method)
    try:
        log(job_id, "info", f"Scraping URL (Playwright): {url}")
        headless = bool(getattr(settings, "BROWSER_HEADLESS", False))
        slowmo = int(getattr(settings, "BROWSER_SLOWMO_MS", 0))

        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=headless,
                slow_mo=slowmo,
                args=["--start-maximized"]
            )
            ctx = browser.new_context(
                user_agent=USER_AGENT,
                locale="th-TH",
                viewport=None
            )
            pw_page = ctx.new_page()

            try:
                # 1) goto (ลอง 2 จังหวะ)
                try:
                    pw_page.goto(url, timeout=60_000, wait_until="domcontentloaded")
                except PWTimeoutError:
                    pw_page.goto(url, timeout=60_000, wait_until="load")

                # 2) เช็ค/แก้ captcha
                try:
                    has_captcha = pw_page.locator('iframe[title="reCAPTCHA"]').count() > 0
                    if has_captcha:
                        log(job_id, "warn", "Captcha detected! Attempting to solve with Playwright...")
                        solver = RecaptchaSolver(pw_page)
                        solved = solver.solveCaptcha(max_retries=3)
                        if solved:
                            log(job_id, "info", "Captcha Solved! Waiting for reload...")
                            try:
                                pw_page.wait_for_load_state("networkidle", timeout=10_000)
                            except PWTimeoutError:
                                pass
                            time.sleep(3)
                        else:
                            log(job_id, "error", "Failed to solve Captcha.")
                except Exception as e:
                    log(job_id, "warn", f"Error during captcha check: {e}")

                # 3) รอโหลดเพิ่ม
                try:
                    pw_page.wait_for_load_state("networkidle", timeout=10_000)
                except PWTimeoutError:
                    pass
                time.sleep(1.2)

                # 4) ✅ text_probe (ตำแหน่งถูกแล้ว)
                try:
                    text_probe = (pw_page.title() or "") + "\n" + (pw_page.content() or "")[:2000]
                    if is_probably_blocked(text_probe):
                        log(job_id, "warn", "Page looks blocked (captcha/denied/cloudflare).")
                except Exception:
                    pass

                # 5) เก็บผลลัพธ์
                final_url = pw_page.url
                html = pw_page.content() or ""
                title = pw_page.title() or ""

            finally:
                try:
                    ctx.close()
                except Exception:
                    pass
                try:
                    browser.close()
                except Exception:
                    pass

    except Exception as e:
        log(job_id, "warn", f"Playwright failed: {e}")


    # 2) requests fallback (ถ้า Playwright พัง หรือ html ว่างเปล่า)
    if not html.strip():
        try:
            log(job_id, "info", f"Scraping URL (requests fallback): {url}")
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if r.status_code == 200 and r.text.strip():
                final_url = r.url
                html = r.text
        except Exception as e:
            log(job_id, "error", f"requests fallback failed: {e}")

    # --- Process text/title ---
    if not title:
        title = extract_page_title(html)

    text = content_text_from_html(html)

    # ... (ส่วน download files/images ด้านล่างเหมือนเดิม ไม่ต้องแก้) ...
    
    content_path = os.path.join(folder, "content.txt")
    with open(content_path, "w", encoding="utf-8") as f:
        f.write(text or "")

    # --- Download files ---
    try:
        file_links = extract_file_links(html, final_url)
        downloaded_files = download_files_via_requests(
            file_links, files_dir, base_url=final_url, max_files=max_files
        )
    except Exception:
        downloaded_files = 0

    # --- Download images + write download meta ---
    images_download_meta_jsonl = os.path.join(images_dir, "images_download_meta.jsonl")

    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})

    img_items = extract_image_candidates(html, final_url)
    saved_imgs = download_images(
        session=sess,
        image_items=img_items,
        save_folder=images_dir,
        max_images=max_images,
        referer=final_url,
    )

    for it in saved_imgs:
        append_jsonl(images_download_meta_jsonl, {
            "type": "web_image_download",
            "page_url": final_url,
            "page_title": title,
            "source_url": it.get("source_url", ""),
            "saved_path": it.get("saved_path", ""),
            "sha1": it.get("sha1", ""),
            "width": it.get("width", 0),
            "height": it.get("height", 0),
            "alt_text": it.get("alt_text", ""),
            "title_text": it.get("title_text", ""),
            "nearby_text": it.get("nearby_text", ""),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    log(job_id, "info", f"Scrape finished: {final_url}", {
        "title": title,
        "content_len": len(text or ""),
        "images": len(saved_imgs),
        "files": downloaded_files,
    })

    return {
        "URL": final_url,
        "Title": title,
        "Folder": folder,
        "content_path": content_path,
        "images_dir": images_dir,
        "files_dir": files_dir,
        "images_download_meta_jsonl": images_download_meta_jsonl,
        "downloaded_images": len(saved_imgs),
        "downloaded_files": downloaded_files,
    }


def run_external_scrape(job_id: str, keyword: str, max_links: int = 5) -> Dict[str, Any]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_kw = re.sub(r"[^a-zA-Z0-9ก-๙_]+", "_", keyword)[:40]
    main_folder = os.path.join(settings.OUTPUT_BASE_DIR, f"data_{safe_kw}_{ts}")
    safe_mkdir(main_folder)

    urls = search_urls(job_id, keyword, max_links=max_links)
    log(job_id, "info", f"Search URLs => {len(urls)}", {"urls": urls})

    collected: List[Dict[str, Any]] = []
    for idx, u in enumerate(urls, start=1):
        site_folder = os.path.join(main_folder, f"{idx}_{safe_filename(urlparse(u).netloc, 40)}")
        try:
            res = scrape_single_url(job_id, u, site_folder)
            res["Source"] = "search_urls"
            collected.append(res)
        except Exception as e:
            log(job_id, "error", f"Failed to scrape {u}: {e}", {"traceback": traceback.format_exc()})

    csv_path = os.path.join(main_folder, "final_data.csv")
    if collected:
        pd.DataFrame(collected).to_csv(csv_path, index=False, encoding="utf-8-sig")

    return {
        "main_folder": main_folder,
        "csv_path": csv_path,
        "items": collected,
        "urls": urls,
    }
