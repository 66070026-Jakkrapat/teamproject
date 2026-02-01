# backend/scraping/facebook_scraping.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

from backend.settings import settings
from backend.utils.job_manager import log

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

@dataclass
class FacebookSession:
    storage_state_path: str

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

def ensure_logged_in(job_id: str, headless: bool = True) -> FacebookSession:
    """
    Login แล้ว save storage_state ไว้ reuse session ครั้งถัดไป
    - ถ้ามี storage_state อยู่แล้ว จะลองเปิด Facebook แล้วตรวจว่าล็อกอินอยู่ไหม
    - ถ้าไม่อยู่ จะทำ login ใหม่ แล้ว save state
    """
    storage = settings.FB_STORAGE_STATE_PATH
    _ensure_dir(storage)

    if not settings.FB_EMAIL or not settings.FB_PASS:
        raise RuntimeError("FB_EMAIL/FB_PASS missing in .env")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(
            user_agent=USER_AGENT,
            storage_state=storage if os.path.exists(storage) else None
        )
        page = ctx.new_page()

        log(job_id, "info", "Opening facebook.com to verify session...")
        page.goto("https://www.facebook.com/", timeout=60_000)
        time.sleep(2)

        if _is_logged_in(page):
            log(job_id, "info", "Facebook session is already logged in (reuse).")
            ctx.storage_state(path=storage)
            ctx.close()
            browser.close()
            return FacebookSession(storage_state_path=storage)

        log(job_id, "warn", "Session not logged in. Performing login...")
        _do_login(job_id, page)

        ctx.storage_state(path=storage)
        log(job_id, "info", f"Saved storage_state: {storage}")

        ctx.close()
        browser.close()
        return FacebookSession(storage_state_path=storage)

def _is_logged_in(page) -> bool:
    # heuristic: presence of profile/menu or absence of login form
    html = page.content().lower()
    if "name=\"email\"" in html and "name=\"pass\"" in html:
        return False
    # sometimes FB loads minimal; try check for "home.php" redirect or "logout"
    if "logout" in html or "home.php" in page.url:
        return True
    # fallback: check for cookie presence via JS
    try:
        cookies = page.context.cookies()
        return any(c.get("name") in ("c_user", "xs") for c in cookies)
    except Exception:
        return False

def _do_login(job_id: str, page) -> None:
    page.goto("https://www.facebook.com/login", timeout=60_000)
    time.sleep(2)

    try:
        page.fill("input[name='email']", settings.FB_EMAIL, timeout=10_000)
        page.fill("input[name='pass']", settings.FB_PASS, timeout=10_000)
        page.click("button[name='login']", timeout=10_000)
    except PWTimeoutError:
        raise RuntimeError("Facebook login page selectors not found (page changed or blocked).")

    # wait navigation / checkpoint
    time.sleep(4)
    page.wait_for_load_state("networkidle", timeout=60_000)

    # If checkpoint/2FA, user must resolve manually (but we keep browser state)
    url = page.url.lower()
    if "checkpoint" in url or "two_factor" in url or "security" in url:
        log(job_id, "warn", f"Facebook requires extra verification at: {page.url}")
        # We do not hard-fail; allow user to do manual step if headless=False
        if page.context.browser.is_connected():
            pass

    if not _is_logged_in(page):
        raise RuntimeError("Facebook login failed (still not logged in). Consider headless=False for first login.")

def scrape_facebook_post_html(job_id: str, post_url: str, headless: bool = True) -> Dict[str, Any]:
    """
    เปิด post/page URL ด้วย session แล้วคืน html + final_url
    """
    sess = ensure_logged_in(job_id, headless=headless)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(user_agent=USER_AGENT, storage_state=sess.storage_state_path)
        page = ctx.new_page()

        log(job_id, "info", f"Opening FB URL: {post_url}")
        page.goto(post_url, timeout=60_000)
        page.wait_for_load_state("networkidle", timeout=60_000)
        time.sleep(2)

        html = page.content()
        final_url = page.url

        ctx.close()
        browser.close()

    return {"final_url": final_url, "html": html}

def scrape_facebook_group_search(job_id: str, group_url: str, keyword: str, headless: bool = True) -> Dict[str, Any]:
    """
    Example: search within group/page by keyword (basic). Returns html snapshot.
    """
    sess = ensure_logged_in(job_id, headless=headless)
    q = keyword.strip()
    search_url = f"{group_url.rstrip('/')}/search/?q={q}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(user_agent=USER_AGENT, storage_state=sess.storage_state_path)
        page = ctx.new_page()

        log(job_id, "info", f"Opening FB search: {search_url}")
        page.goto(search_url, timeout=60_000)
        page.wait_for_load_state("networkidle", timeout=60_000)
        time.sleep(2)

        html = page.content()
        final_url = page.url

        ctx.close()
        browser.close()

    return {"final_url": final_url, "html": html}
