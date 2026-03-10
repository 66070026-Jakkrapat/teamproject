"""
Web scraper service — calls pw_worker.py in a separate process
to avoid Uvicorn event-loop conflicts on Windows.
"""

import os
import sys
import re
import json
import time
import asyncio
import subprocess
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

from backend.database import AsyncSessionLocal
from backend.models import Document, Chunk
from backend.services.chunker import chunk_document


DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "scrape_output")
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path to the worker script
PW_WORKER = os.path.join(os.path.dirname(__file__), "pw_worker.py")
PYTHON_EXE = sys.executable


def _safe_folder_name(name: str, max_len: int = 40) -> str:
    name = re.sub(r'[\\/*?"<>|:]', '', name or '')
    name = re.sub(r'\s+', '_', name).strip('_')
    return name[:max_len] or "site"


def _run_pw_worker(command: str, args: dict) -> Any:
    """Run pw_worker.py in a subprocess and return the parsed JSON result."""
    # Force UTF-8 encoding in the child process
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONLEGACYWINDOWSSTDIO"] = "0"

    cmd = [PYTHON_EXE, "-u", PW_WORKER, command, json.dumps(args, ensure_ascii=False)]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        stdout_bytes, stderr_bytes = proc.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("⚠️ pw_worker timed out (300s)")
        return None
    except Exception as e:
        print(f"⚠️ pw_worker failed to start: {e}")
        return None

    # Decode output safely
    stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
    stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

    if proc.returncode != 0:
        print(f"⚠️ pw_worker [{command}] exit code {proc.returncode}")
        print(f"   stderr: {stderr[-500:]}")
        return None

    marker = "__PW_RESULT__"
    if marker in stdout:
        json_str = stdout.split(marker, 1)[1].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error: {e}")
            print(f"   json_str: {json_str[:300]}")
            return None
    else:
        print(f"⚠️ No result marker in worker output")
        print(f"   stdout: {stdout[:300]}")
        print(f"   stderr: {stderr[:300]}")
        return None



# ============================================================
# Async wrappers for FastAPI
# ============================================================
async def fetch_google_search_results(keyword: str, max_results: int = 3) -> List[Dict[str, str]]:
    """Search Google via Playwright subprocess."""
    result = await asyncio.to_thread(
        _run_pw_worker,
        "google_search",
        {"keyword": keyword, "max_results": max_results},
    )
    return result if isinstance(result, list) else []


async def scrape_url(
    url: str,
    download_pdfs: bool = True,
    download_images: bool = True,
    max_files: int = 20,
    save_folder: str = "",
) -> Dict[str, Any]:
    """Scrape a URL via Playwright subprocess."""
    result = await asyncio.to_thread(
        _run_pw_worker,
        "scrape_url",
        {"url": url, "max_files": max_files, "save_folder": save_folder},
    )
    if result is None:
        return {
            "success": False,
            "error": "Playwright worker failed — check terminal for details",
            "files": [],
            "images": [],
            "page_text": "",
            "links_found": [],
            "url": url,
        }
        
    # --- RAG Ingestion Block ---
    page_text = result.get("page_text", "").strip()
    if page_text:
        try:
            async with AsyncSessionLocal() as db:
                # 1. Save Document
                doc = Document(
                    filename=_safe_folder_name(urlparse(url).netloc) + ".txt",
                    doc_type="web_scrape",
                    source_url=url,
                    raw_text=page_text,
                    status="processing"
                )
                db.add(doc)
                await db.commit()
                await db.refresh(doc)
                
                # 2. Chunk and Embedding
                chunk_results = await chunk_document(page_text)
                for cr in chunk_results:
                    chunk = Chunk(
                        document_id=doc.id,
                        chunk_index=cr.chunk_index,
                        chunk_text=cr.text,
                        summary=cr.summary,
                        token_count=cr.token_count,
                        embedding=cr.embedding,
                        metadata_={
                            "start_char": cr.start_char,
                            "end_char": cr.end_char,
                        },
                    )
                    db.add(chunk)
                
                doc.status = "completed"
                await db.commit()
                
                result["rag_chunks_created"] = len(chunk_results)
                result["rag_document_id"] = doc.id
        except Exception as e:
            print(f"⚠️ Failed to ingest scraped content into RAG: {e}")
            result["rag_error"] = str(e)
            
    return result


async def scrape_by_keyword(
    keyword: str,
    max_sites: int = 3,
    max_files_per_site: int = 10,
):
    """Search Google for keyword, follow top N links, and scrape them (Async Generator)."""
    yield {"status": "searching", "message": f"🔍 กำลังค้นหา keyword '{keyword}' ใน Google..."}

    # Create a keyword-based output folder
    safe_kw = _safe_folder_name(keyword)
    ts = time.strftime("%Y%m%d_%H%M%S")
    keyword_folder = os.path.join(OUTPUT_DIR, f"{safe_kw}_{ts}")
    os.makedirs(keyword_folder, exist_ok=True)

    targets = await fetch_google_search_results(keyword, max_results=max_sites)

    all_results = []
    urls = [t["url"] for t in targets]

    if not urls:
        yield {
            "status": "done",
            "result": {
                "keyword": keyword,
                "output_folder": keyword_folder,
                "urls_scraped": 0,
                "urls": [],
                "total_files": 0,
                "total_images": 0,
                "files": [],
                "results": [{"success": False, "error": "ไม่พบผลลัพธ์จาก Google หรือเกิดข้อผิดพลาดในการค้นหา"}],
            }
        }
        return

    yield {"status": "found", "message": f"🌐 พบ {len(urls)} เว็บไซต์ กำลังเริ่มดึงข้อมูลเรียงตามลำดับ..."}

    for idx, target in enumerate(targets, 1):
        url = target["url"]
        domain = _safe_folder_name(urlparse(url).netloc)
        site_folder = os.path.join(keyword_folder, f"{idx:02d}_{domain}")

        yield {"status": "scraping", "message": f"🕷️ [เว็บ {idx}/{len(urls)}] กำลังดึงข้อความและรูปภาพจาก {domain} ..."}

        result = await scrape_url(url, max_files=max_files_per_site, save_folder=site_folder)
        result["source_url"] = url
        result["search_title"] = target.get("title", "")
        result["rank"] = idx
        
        if result.get("rag_document_id"):
            yield {"status": "ingesting", "message": f"🧠 [เว็บ {idx}/{len(urls)}] นำเข้าข้อมูล '{domain}' สู่ระบบ RAG เสร็จสมบูรณ์ (ได้ {result.get('rag_chunks_created')} chunks)"}
            
        all_results.append(result)

    total_files = []
    total_images = []
    for r in all_results:
        total_files.extend(r.get("files", []))
        total_images.extend(r.get("images", []))

    final_result = {
        "keyword": keyword,
        "output_folder": keyword_folder,
        "urls_scraped": len(urls),
        "urls": urls,
        "total_files": len(total_files),
        "total_images": len(total_images),
        "files": total_files,
        "results": all_results,
    }
    
    yield {"status": "done", "result": final_result}
