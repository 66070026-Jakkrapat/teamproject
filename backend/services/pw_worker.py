"""
Standalone Playwright worker — runs in its own process to avoid
Uvicorn event-loop conflicts on Windows.

Usage:
    python -m backend.services.pw_worker google_search '{"keyword":"...", "max_results":3}'
    python -m backend.services.pw_worker scrape_url    '{"url":"...", "max_files":20}'
"""

import sys
import json
import os
import re
import time
import random
import hashlib
import unicodedata
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse, unquote, parse_qs

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)

# Base output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scrape_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────
def safe_filename(name: str, max_len: int = 60) -> str:
    name = unicodedata.normalize("NFKC", name or "")
    name = re.sub(r'[\\/*?"<>|:]', "", name)
    name = re.sub(r"\s+", " ", name).strip()
    name = name[:max_len].rstrip(" .").strip()
    return name if name else "site"


def _try_accept_consent(page):
    try:
        for txt in ["I agree", "Accept all", "Agree", "ยอมรับทั้งหมด", "ยอมรับ", "ยืนยัน", "ตกลง"]:
            btn = page.locator(f"button:has-text('{txt}')")
            if btn.count() > 0:
                btn.first.click(timeout=2000)
                page.wait_for_timeout(800)
                break
    except Exception:
        pass


def _wait_any(page, selectors, timeout_ms=45000):
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


def _norm_href(href):
    if not href:
        return ""
    if href.startswith("/url?"):
        return unquote(parse_qs(urlparse(href).query).get("q", [""])[0])
    return href


def _normalize_text(raw: str) -> str:
    """Clean up extracted text."""
    if not raw:
        return ""
    s = unicodedata.normalize("NFKC", raw)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[^\S\r\n]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    lines = []
    seen = set()
    for line in s.splitlines():
        line = line.strip()
        if len(line) < 3:
            continue
        key = re.sub(r"\s+", " ", line).lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return "\n".join(lines)

# นำเข้า readability เพิ่มเติม
try:
    from readability import Document
except ImportError:
    Document = None
    print("Warning: readability-lxml not installed. Falling back to basic extraction.", file=sys.stderr)

# ── BeautifulSoup Content Extraction (UPGRADED) ──────────────
JUNK_CLASSES = re.compile(
    r"header|footer|nav|menu|sidebar|widget|comment|social|share|"
    r"related|recommend|tag|popup|modal|banner|advert|sponsor|promo|"
    r"login|register|subscribe|newsletter|breadcrumb|pagination|"
    r"author|post-meta|meta-info|taboola|outbrain|disqus|"
    r"must-read|popular|trending|cookie|consent|aside|"
    r"relate|suggest|bottom|ads|line-it|fb-share|tw-share|"
    r"news-list|article-list|more-news|read-more",
    re.IGNORECASE
)

def _clean_dom(soup: BeautifulSoup):
    """Remove definitely non-article elements from the DOM."""
    for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe", 
                     "button", "nav", "footer", "header", "form", "aside", "menu", "dialog"]):
        tag.decompose()
        
    for tag in soup.find_all(True):
        if getattr(tag, "attrs", None) is None or not hasattr(tag, "get"):
            continue
            
        css_id = str(tag.get("id", "")).lower()
        css_class = tag.get("class", [])
        if isinstance(css_class, list):
            css_class = " ".join(css_class)
        css_class = str(css_class).lower()
            
        css_name = f"{css_id} {css_class}".strip()
        
        if css_name and JUNK_CLASSES.search(css_name):
            tag.decompose()
            continue
            
        style = str(tag.get("style", "")).lower()
        if "display: none" in style or "opacity: 0" in style or "visibility: hidden" in style:
            tag.decompose()

def _extract_main_container_bs4(soup: BeautifulSoup) -> str:
    """Fallback text density analysis if Readability fails."""
    for sel in ["article", "[itemprop='articleBody']", ".entry-content", ".post-content", ".article-content", ".detail-content"]:
        node = soup.select_one(sel)
        if node:
            text = node.get_text("\n", strip=True)
            if len(text) > 300:
                return text
                
    parent_scores = {}
    for p in soup.find_all(['p', 'div', 'span', 'h2', 'h3']):
        text = p.get_text(" ", strip=True)
        if len(text) < 30: 
            continue
            
        a_tags = p.find_all('a')
        a_text_len = sum(len(a.get_text(strip=True)) for a in a_tags)
        if len(text) > 0 and (a_text_len / len(text)) > 0.4:
            continue
            
        parent = p.parent
        if parent not in parent_scores:
            parent_scores[parent] = 0
        parent_scores[parent] += len(text)
        
        grandparent = parent.parent if parent else None
        if grandparent:
            if grandparent not in parent_scores:
                parent_scores[grandparent] = 0
            parent_scores[grandparent] += len(text) * 0.5

    if parent_scores:
        best_parent = sorted(parent_scores.items(), key=lambda x: x[1], reverse=True)[0][0]
        return best_parent.get_text("\n", strip=True)
        
    return soup.body.get_text("\n", strip=True) if soup.body else soup.get_text("\n", strip=True)

def _filter_lines(text: str) -> str:
    """Filter out boilerplate lines and stop at footer indicators."""
    if not text: return ""
    
    lines = [ln.strip() for ln in text.splitlines()]
    out = []
    seen = set()
    
    stop_words = [
        "เรื่องที่เกี่ยวข้อง", "บทความที่เกี่ยวข้อง", "อ่านเพิ่มเติม", "ข่าวล่าสุด", 
        "เรื่องเด่น", "แท็กที่เกี่ยวข้อง", "แสดงความคิดเห็น", "tags:", "credit:", "ที่มา:",
        "คุณอาจสนใจ", "ข่าวที่คุณอาจสนใจ", "บทความยอดนิยม", "ข่าวน่าสนใจ", "ข่าวแนะนำ",
        "sponsored", "บทความต่อไป", "แชร์ข่าว", "ติดตามเรา", "คลิกอ่านเพิ่มเติม",
        "ข่าวต่างประเทศ", "ข่าวสังคม", "ข่าวการเมือง", "ประเด็นฮิต" # เพิ่ม Stop words
    ]
    
    for ln in lines:
        if not ln:
            continue
            
        low = ln.lower()
        
        # Hard stop logic
        if len(low) < 60 and any(w in low for w in stop_words):
            if sum(len(x) for x in out) > 300: 
                break
                
        skip_phrases = ["หน้าแรก", "สมัครสมาชิก", "เข้าสู่ระบบ", "แชร์เรื่องนี้", "แชร์บทความ", "อ่านข่าวต่อ"]
        if any(w in low for w in skip_phrases) and len(low) < 40:
            continue
            
        if len(ln) < 10 and " " not in ln and not re.search(r'\d', ln):
            continue
            
        norm = re.sub(r'\s+', ' ', low)
        if norm in seen:
            continue
        seen.add(norm)
        
        out.append(ln)
        
    return "\n".join(out)

def content_main_text_from_html(html: str) -> str:
    if not html: return ""
    
    best_text = ""
    
    # 1. ลองใช้ Readability ก่อน (ถ้ามีการติดตั้งไว้)
    if Document:
        try:
            doc = Document(html)
            # ดึงเฉพาะ HTML ของเนื้อหาหลัก
            summary_html = doc.summary() 
            # เอา HTML ที่ได้มาแปลงเป็น text ด้วย BeautifulSoup อีกที
            soup_summary = BeautifulSoup(summary_html, "html.parser")
            _clean_dom(soup_summary) # Clean อีกรอบเผื่อมี JUNK หลงเหลือในเนื้อหาหลัก
            best_text = soup_summary.get_text("\n", strip=True)
        except Exception as e:
            print(f"Readability failed: {e}. Falling back to BS4.", file=sys.stderr)
            best_text = ""

    # 2. Fallback ไปใช้ BS4 Density Analysis ถ้า Readability ไม่ทำงาน หรือดึงเนื้อหามาได้น้อยเกินไป
    if not best_text or len(best_text) < 300:
        soup = BeautifulSoup(html, "html.parser")
        _clean_dom(soup)
        best_text = _extract_main_container_bs4(soup)
        
    # 3. Normalize & Filter Lines ชั้นสุดท้าย
    return _filter_lines(_normalize_text(best_text))


# ── Google Search ────────────────────────────────────────────
def google_search(keyword: str, max_results: int = 3) -> List[Dict[str, str]]:
    items, seen = [], set()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        ctx = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 720},
            locale="th-TH",
        )
        page = ctx.new_page()

        page.goto("https://www.google.com/", wait_until="domcontentloaded", timeout=60000)
        _try_accept_consent(page)

        box = page.locator("textarea[name='q'], input[name='q']").first
        box.click()
        box.type(keyword, delay=80)
        page.wait_for_timeout(500)
        page.keyboard.press("Enter")

        try:
            page.wait_for_load_state("domcontentloaded", timeout=60000)
        except Exception:
            pass
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except Exception:
            pass

        ok = _wait_any(page, ["#search a:has(h3)", "a:has(h3)", "a[data-ved]:has(h3)"], timeout_ms=45000)
        if not ok:
            browser.close()
            return items

        page.wait_for_timeout(1500)

        anchors = page.locator("#search a:has(h3)")
        if anchors.count() == 0:
            anchors = page.locator("a:has(h3)")

        for i in range(min(anchors.count(), max_results * 8)):
            try:
                a = anchors.nth(i)
                href = _norm_href(a.get_attribute("href") or "")
                title = ""
                try:
                    title = (a.locator("h3").first.inner_text() or "").strip()
                except Exception:
                    pass
                if href.startswith("http") and "google.com" not in urlparse(href).netloc:
                    key = href.split("#")[0].strip()
                    if key and key not in seen:
                        seen.add(key)
                        items.append({"url": key, "title": title or key})
                if len(items) >= max_results:
                    break
            except Exception:
                continue

        browser.close()

    return items


# ── Scrape URL ───────────────────────────────────────────────
def scrape_url(url: str, max_files: int = 20, save_folder: str = "") -> Dict[str, Any]:
    """Go to a URL, extract text content, download PDFs and images."""

    # Create save folder
    if not save_folder:
        domain = safe_filename(urlparse(url).netloc, 40)
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_folder = os.path.join(OUTPUT_DIR, f"{domain}_{ts}")
    os.makedirs(save_folder, exist_ok=True)

    files_dir = os.path.join(save_folder, "files")
    images_dir = os.path.join(save_folder, "images")
    os.makedirs(files_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    files_downloaded = []
    images_downloaded = []
    page_text = ""
    page_title = ""
    links_found = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        ctx = browser.new_context(
            accept_downloads=True,
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 720},
            locale="th-TH",
        )
        page = ctx.new_page()

        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        try:
            page.wait_for_load_state("networkidle", timeout=8000)
        except PWTimeoutError:
            pass
        page.wait_for_timeout(2000)

        # ── Extract title ────────────────────────────────
        try:
            page_title = page.title() or ""
        except Exception:
            page_title = ""

        # ── Extract HTML first for BS4 parsing ───────────
        try:
            html = page.content() or ""
        except Exception:
            html = ""

        # ── Extract text ─────────────────────────────────
        # Try BS4 main content extraction first
        page_text = content_main_text_from_html(html)

        # Fallback to visible inner text if BS4 fails
        if len(page_text) < 200:
            raw_text = ""
            for sel in ["article", "main", "[role='main']", ".entry-content",
                         ".post-content", ".article-content", ".content"]:
                try:
                    loc = page.locator(sel)
                    if loc.count() > 0:
                        raw_text = loc.first.inner_text(timeout=4000) or ""
                        if len(raw_text.strip()) > 200:
                            break
                except Exception:
                    continue

            if not raw_text.strip() or len(raw_text.strip()) < 200:
                try:
                    raw_text = page.inner_text("body") or ""
                except Exception:
                    raw_text = ""
            
            fallback_text = _normalize_text(raw_text)
            if len(fallback_text) > len(page_text):
                page_text = fallback_text

        # ── Find all links ───────────────────────────────
        try:
            all_links = page.eval_on_selector_all(
                "a[href]",
                "els => els.map(el => ({href: el.href, text: (el.textContent||'').trim().substring(0,120)}))"
            )
        except Exception:
            all_links = []

        for lnk in all_links:
            href = lnk.get("href", "")
            if not href:
                continue
            full = urljoin(url, href)
            links_found.append({"url": full, "text": lnk.get("text", "")})

        # ── Download PDFs ────────────────────────────────
        pdf_count = 0
        for lnk in links_found:
            if pdf_count >= max_files:
                break
            lurl = lnk["url"]
            if any(x in lurl.lower() for x in [".pdf", "getmedia", "download"]):
                fp = _download_file(lurl, files_dir, url)
                if fp:
                    files_downloaded.append(fp)
                    pdf_count += 1

        # ── Download images ──────────────────────────────
        try:
            img_srcs = page.eval_on_selector_all(
                "img[src]",
                """els => els.map(el => ({
                    src: el.src || el.dataset.src || '',
                    alt: (el.alt || '').trim(),
                    width: el.naturalWidth || el.width || 0,
                    height: el.naturalHeight || el.height || 0
                }))"""
            )
        except Exception:
            img_srcs = []

        img_count = 0
        seen_urls = set()
        for img in img_srcs:
            if img_count >= 20:
                break
            src = img.get("src", "").strip()
            if not src or src.startswith("data:"):
                continue
            full_src = urljoin(url, src)
            if full_src in seen_urls:
                continue
            # Skip icons/logos/tiny images
            w = img.get("width", 0) or 0
            h = img.get("height", 0) or 0
            if w > 0 and h > 0 and (w < 50 or h < 50):
                continue
            ignore_kw = ["logo", "icon", "sprite", "button", "spacer", "blank", "pixel", "avatar", "badge"]
            if any(k in full_src.lower() for k in ignore_kw):
                continue

            seen_urls.add(full_src)
            fp = _download_image(full_src, images_dir, url, img_count + 1)
            if fp:
                images_downloaded.append({
                    "path": fp,
                    "source_url": full_src,
                    "alt": img.get("alt", ""),
                })
                img_count += 1

        browser.close()

    # ── Save text content to file ────────────────────────
    content_path = os.path.join(save_folder, "content.txt")
    with open(content_path, "w", encoding="utf-8") as f:
        f.write(f"Title: {page_title}\n")
        f.write(f"URL: {url}\n")
        f.write(f"Scraped at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        f.write(page_text)

    # ── Save metadata ────────────────────────────────────
    meta = {
        "url": url,
        "title": page_title,
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "content_length": len(page_text),
        "files_downloaded": len(files_downloaded),
        "images_downloaded": len(images_downloaded),
    }
    with open(os.path.join(save_folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "success": True,
        "url": url,
        "title": page_title,
        "folder": save_folder,
        "content_path": content_path,
        "content_length": len(page_text),
        "page_text": page_text[:5000],
        "files": files_downloaded,
        "images": [im["path"] for im in images_downloaded],
        "images_detail": images_downloaded,
        "links_found": links_found[:50],
    }


def _download_file(url: str, save_dir: str, referer: str) -> Optional[str]:
    """Download a PDF or document file."""
    import httpx
    try:
        fn = os.path.basename(urlparse(url).path) or f"file_{random.randint(100,999)}.pdf"
        fn = re.sub(r'[^\w\-_\.]', '_', fn)
        fp = os.path.join(save_dir, fn)
        with httpx.Client(timeout=30, follow_redirects=True) as c:
            r = c.get(url, headers={"User-Agent": USER_AGENT, "Referer": referer})
            if r.status_code != 200:
                return None
            ct = (r.headers.get("content-type") or "").lower()
            if "text/html" in ct:
                return None
            if len(r.content) < 5000:
                return None
            with open(fp, "wb") as f:
                f.write(r.content)
        return fp
    except Exception:
        return None


def _download_image(url: str, save_dir: str, referer: str, idx: int) -> Optional[str]:
    """Download an image file."""
    import httpx
    try:
        # Determine extension from URL
        path = urlparse(url).path.lower()
        if ".png" in path:
            ext = "png"
        elif ".webp" in path:
            ext = "webp"
        elif ".gif" in path:
            ext = "gif"
        elif ".svg" in path:
            ext = "svg"
        else:
            ext = "jpg"

        fn = f"img_{idx:03d}_{random.randint(100,999)}.{ext}"
        fp = os.path.join(save_dir, fn)

        with httpx.Client(timeout=20, follow_redirects=True) as c:
            r = c.get(url, headers={
                "User-Agent": USER_AGENT,
                "Referer": referer,
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            })
            if r.status_code != 200:
                return None
            ct = (r.headers.get("content-type") or "").lower()
            if "text/html" in ct:
                return None
            if len(r.content) < 1000:
                return None
            with open(fp, "wb") as f:
                f.write(r.content)
        return fp
    except Exception:
        return None


# ── CLI entry ────────────────────────────────────────────────
if __name__ == "__main__":
    cmd = sys.argv[1]
    args = json.loads(sys.argv[2]) or {}

    try:
        if cmd == "google_search":
            result = google_search(args["keyword"], args.get("max_results", 3))
        elif cmd == "scrape_url":
            result = scrape_url(
                args["url"],
                max_files=args.get("max_files", 20),
                save_folder=args.get("save_folder", ""),
            )
        else:
            result = {"error": f"Unknown command: {cmd}"}
    except Exception as e:
        result = {"success": False, "error": str(e)}

    # Output JSON to stdout for the parent process to read
    print("__PW_RESULT__")
    print(json.dumps(result, ensure_ascii=False, default=str))