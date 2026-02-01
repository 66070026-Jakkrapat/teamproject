# backend/main.py
from __future__ import annotations

import os
import time
import uuid
import threading
import sys
import traceback
import asyncio
from typing import Any, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi import Response
from backend.vision.image_understanding import warmup_blip
from backend.settings import settings
from backend.utils.job_manager import (
    new_job, set_stage, log,
    get_status, get_logs, get_workflow_steps
)

# ✅ แนะนำ: set env Paddle ไว้ตรงนี้ด้วย (กันหลุด)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "true")

from backend.scraping.web_scraping import run_external_scrape
from backend.scraping.facebook_scraping import scrape_facebook_post_html
from backend.vision.image_understanding import caption_images_with_blip_and_translate
from backend.ocr.ocr_pipeline import process_folder_pdfs
from backend.rag.rag_store import RAGStore
from backend.rag.ingest import ingest_main_folder, ingest_text_blob
from backend.agent.agent_flow import AgenticRAG
from backend.report_utils.report import generate_report_md
from backend.evaluation.eval_runner import run_eval

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

UI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui"))

store = RAGStore(
    database_url=settings.DATABASE_URL,
    ollama_host=settings.OLLAMA_HOST,
    embed_model=settings.EMBED_MODEL,
    embed_dims=settings.EMBED_DIMS
)
agent = AgenticRAG(store=store)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ startup
    os.makedirs(settings.OUTPUT_BASE_DIR, exist_ok=True)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    await store.init_db()
    # ใน lifespan:
    warmup_blip(settings.BLIP_MODEL)
    yield
    # ✅ shutdown (ถ้าจะปิด engine ก็ทำได้ แต่ไม่จำเป็น)
    # await store.engine.dispose()

app = FastAPI(
    title="AI Agent: Scraping + OCR + Dual RAG + Tavily + Evaluation",
    description=(
        "Workflow:\n"
        "1) /pipeline/external/scrape\n"
        "2) caption images (BLIP + Argos)\n"
        "3) OCR PDFs\n"
        "4) ingest to RAG (namespace external)\n"
        "5) /ask (Agent) → structured/semantic → fallback Tavily\n"
        "6) /eval/run precision/recall@k\n\n"
        "UI Workflow: /ui/workflow\n"
    ),
    version="1.0.0",
    lifespan=lifespan,  # ✅ เปลี่ยนตรงนี้
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)



@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    if os.path.exists(os.path.join(UI_DIR, "favicon.ico")):
        return FileResponse(os.path.join(UI_DIR, "favicon.ico"))
    return Response(status_code=204)


@app.get("/docs", include_in_schema=False)
def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_ui_parameters={"persistAuthorization": True},
    )


@app.get("/ui/swagger_custom.js", include_in_schema=False)
def swagger_custom_js():
    return FileResponse(os.path.join(UI_DIR, "swagger_custom.js"))


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui")


@app.get("/ui", include_in_schema=False)
def ui_index():
    return FileResponse(os.path.join(UI_DIR, "index.html"))


@app.get("/ui/workflow", include_in_schema=False)
def ui_workflow():
    return FileResponse(os.path.join(UI_DIR, "workflow.html"))


@app.get("/ui/app.js", include_in_schema=False)
def ui_app_js():
    return FileResponse(os.path.join(UI_DIR, "app.js"))


@app.get("/ui/workflow.js", include_in_schema=False)
def ui_workflow_js():
    return FileResponse(os.path.join(UI_DIR, "workflow.js"))


@app.get("/ui/styles.css", include_in_schema=False)
def ui_styles():
    return FileResponse(os.path.join(UI_DIR, "styles.css"))

@app.get("/workflow/steps")
def workflow_steps():
    return {"steps": get_workflow_steps()}


@app.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    return get_status(job_id)


@app.get("/jobs/{job_id}/logs")
def job_logs(job_id: str, tail: int = 200):
    return {"job_id": job_id, "logs": get_logs(job_id, tail=tail)}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "api": {"host": settings.API_HOST, "port": settings.API_PORT},
        "paths": {"ui": "/ui", "workflow_ui": "/ui/workflow", "docs": "/docs"},
        "db": {"database_url": settings.DATABASE_URL},
        "models": {
            "ollama_host": settings.OLLAMA_HOST,
            "embed_model": settings.EMBED_MODEL,
            "embed_dims": settings.EMBED_DIMS,
            "blip_model": settings.BLIP_MODEL,
        },
        "output_dirs": {"output_base": settings.OUTPUT_BASE_DIR, "upload_dir": settings.UPLOAD_DIR},
    }


def _run_async(coro):
    return asyncio.run(coro)


def _external_pipeline_thread(job_id: str, keyword: str, amount: int):
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        set_stage(job_id, "web_scraping", f"Scraping keyword={keyword} amount={amount}")
        log(job_id, "info", "Starting external scrape...")
        scrape = run_external_scrape(job_id, keyword, max_links=amount)
        main_folder = scrape["main_folder"]
        items = scrape["items"]

        set_stage(job_id, "data_collecting", "Saved web content/images/pdfs", {"main_folder": main_folder, "items": items})
        log(job_id, "info", "External scrape complete", {"main_folder": main_folder})

        # captioning per site
        set_stage(job_id, "captioning", "Captioning images (BLIP + Argos)...")
        for it in items:
            site_folder = it["Folder"]
            content_path = os.path.join(site_folder, "content.txt")
            ctx = ""
            if os.path.exists(content_path):
                ctx = open(content_path, "r", encoding="utf-8", errors="ignore").read()

            images_dir = os.path.join(site_folder, "images")
            images_meta = os.path.join(images_dir, "images_meta.jsonl")
            images_under = os.path.join(site_folder, "images_understanding.jsonl")

            # ✅ เพิ่มบรรทัดนี้
            
            images_download_meta = os.path.join(images_dir, "images_download_meta.jsonl")

            res = caption_images_with_blip_and_translate(
                images_dir=images_dir,
                images_meta_jsonl=images_meta,
                images_understanding_jsonl=images_under,
                context_text=ctx,
                blip_model=settings.BLIP_MODEL,
                images_download_meta_jsonl=images_download_meta,  # ✅ เพิ่มบรรทัดนี้
                dedupe=True
            )

            log(job_id, "info", "Captioned images", {"site_folder": site_folder, **res})


        # OCR
        set_stage(job_id, "ocr", "OCR PDFs (PyMuPDF + PaddleOCR)...")
        out_root = os.path.join(main_folder, "ocr_results")
        os.makedirs(out_root, exist_ok=True)

        def pcb(p: dict):
            if p.get("stage") in ("file_start", "page_done", "file_done", "done"):
                log(job_id, "info", "OCR progress", p)

        process_folder_pdfs(files_dir=main_folder, out_root=out_root, progress_cb=pcb)

        # Ingest external (✅ FIX: use thread-local store/engine)
        set_stage(job_id, "ingest_external", "Ingest to External RAG...")

        store_thread = RAGStore(
            database_url=settings.DATABASE_URL,
            ollama_host=settings.OLLAMA_HOST,
            embed_model=settings.EMBED_MODEL,
            embed_dims=settings.EMBED_DIMS
        )

        async def _ingest():
            # กันพลาดกรณี extension/table ยังไม่พร้อม
            await store_thread.init_db()
            summary = await ingest_main_folder(store_thread, "external", main_folder)
            # ปิด engine ใน loop เดียวกัน
            await store_thread.engine.dispose()
            return summary

        ingest_summary = run_async_in_new_loop(_ingest())
        log(job_id, "info", "Ingest external done", ingest_summary)

        report_path = generate_report_md(main_folder, {
            "job_id": job_id,
            "keyword": keyword,
            "namespace": "external",
            "ingest": ingest_summary
        })
        log(job_id, "info", "Generated report.md", {"report_path": report_path})

        set_stage(job_id, "ready", "External pipeline completed ✅", {
            "main_folder": main_folder,
            "ingest": ingest_summary,
            "report_path": report_path
        })

    except Exception as e:
        tb = traceback.format_exc()
        msg = f"Pipeline failed: {type(e).__name__}: {repr(e)}"
        set_stage(job_id, "error", msg)
        log(job_id, "error", "Pipeline exception", {"error": repr(e), "type": type(e).__name__, "traceback": tb})


@app.post("/pipeline/external/scrape")
def start_external_scrape(payload: Dict[str, Any] = Body(...)):
    keyword = (payload.get("keyword") or "").strip()
    amount = int(payload.get("amount") or 5)
    if not keyword:
        return {"status": "bad_request", "message": "keyword required"}

    job_id = uuid.uuid4().hex[:12]
    new_job(job_id, kind="external_pipeline")
    log(job_id, "info", "Job created")

    t = threading.Thread(target=_external_pipeline_thread, args=(job_id, keyword, amount), daemon=True)
    t.start()

    return {"status": "ok", "job_id": job_id}


@app.post("/facebook/scrape_post")
def facebook_scrape_post(payload: Dict[str, Any] = Body(...)):
    url = (payload.get("url") or "").strip()
    headless = bool(payload.get("headless", True))
    if not url:
        return {"status": "bad_request", "message": "url required"}
    job_id = uuid.uuid4().hex[:12]
    new_job(job_id, kind="facebook_scrape")
    set_stage(job_id, "web_scraping", "Scraping Facebook with login session...")
    try:
        out = scrape_facebook_post_html(job_id, url, headless=headless)
        set_stage(job_id, "ready", "Facebook scrape done ✅", {"final_url": out["final_url"]})
        return {"status": "ok", "job_id": job_id, "final_url": out["final_url"], "html_len": len(out["html"])}
    except Exception as e:
        set_stage(job_id, "error", f"Facebook scrape failed: {e}")
        return {"status": "error", "job_id": job_id, "error": str(e)}


@app.post("/pipeline/internal/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), entity_hint: str = "internal_doc"):
    """
    ✅ แก้: เก็บไฟล์ลงโฟลเดอร์เฉพาะ job แล้ว OCR เฉพาะโฟลเดอร์นั้น
    """
    job_id = uuid.uuid4().hex[:12]
    new_job(job_id, kind="internal_pipeline")

    job_dir = os.path.join(settings.UPLOAD_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    set_stage(job_id, "data_collecting", "Uploading file...")
    save_path = os.path.join(job_dir, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    log(job_id, "info", "Saved upload", {"path": save_path})

    # OCR
    set_stage(job_id, "ocr", "OCR uploaded pdf...")
    out_root = os.path.join(job_dir, "ocr_results")
    os.makedirs(out_root, exist_ok=True)

    def pcb(p: dict):
        if p.get("stage") in ("file_start", "page_done", "file_done", "done"):
            log(job_id, "info", "OCR progress", p)

    # ✅ OCR เฉพาะ job_dir (ไม่สแกนทั้ง tmp_uploads/)
    process_folder_pdfs(files_dir=job_dir, out_root=out_root, progress_cb=pcb)

    # ingest internal from OCR output
    set_stage(job_id, "ingest_internal", "Ingesting to Internal RAG...")
    added_chunks = added_facts = 0

    from backend.utils.jsonl import iter_jsonl

    for root, _, files in os.walk(out_root):
        for fn in files:
            if fn == "docs.jsonl":
                docs = os.path.join(root, fn)
                for obj in iter_jsonl(docs) or []:
                    if obj.get("type") == "page_text":
                        r = await ingest_text_blob(
                            store, "internal",
                            obj.get("text") or "",
                            meta={
                                "source_type": "internal_pdf_page",
                                "source_path": save_path,
                                "page": int(obj.get("page") or 0),
                                "extract_method": obj.get("extract_method") or "",
                            },
                            entity_hint=entity_hint
                        )
                        added_chunks += r["chunks"]
                        added_facts += r["facts"]

    set_stage(job_id, "ready", "Internal ingest completed ✅", {
        "upload_path": save_path,
        "added_chunks": added_chunks,
        "added_facts": added_facts
    })
    return {"status": "ok", "job_id": job_id, "upload_path": save_path, "added_chunks": added_chunks, "added_facts": added_facts}


@app.post("/ask")
async def ask(payload: Dict[str, Any] = Body(...)):
    q = (payload.get("question") or "").strip()
    if not q:
        return {"status": "bad_request", "message": "question required"}
    top_k = int(payload.get("top_k") or settings.RAG_TOP_K)
    out = await agent.answer(q, top_k=top_k)
    return {"status": "ok", **out}


@app.get("/rag/preview")
async def rag_preview(namespace: str = "external", source_type: str = "", limit: int = 30):
    chunks = await store.preview_chunks(namespace=namespace, source_type=source_type, limit=limit)
    return {"status": "ok", "namespace": namespace, "count": len(chunks), "chunks": chunks}


@app.post("/rag/reset")
async def rag_reset():
    await store.reset_db()
    return {"status": "ok", "message": "db reset done"}


@app.post("/eval/run")
async def eval_run(payload: Dict[str, Any] = Body(...)):
    dataset_path = (payload.get("dataset_path") or "").strip()
    namespace = (payload.get("namespace") or "external").strip()
    k = int(payload.get("k") or 5)
    if not dataset_path or not os.path.exists(dataset_path):
        return {"status": "bad_request", "message": "dataset_path not found"}
    results = await run_eval(store, dataset_path, namespace, k=k)
    return {"status": "ok", "namespace": namespace, "k": k, "results": results}

def run_async_in_new_loop(coro):
    """Run async coroutine in a fresh event loop (safe for threads)."""
    return asyncio.run(coro)



if __name__ == "__main__":
    import uvicorn
    print(f"✅ Swagger: http://localhost:{settings.API_PORT}/docs")
    print(f"✅ UI:      http://localhost:{settings.API_PORT}/ui")
    print(f"✅ Flow UI: http://localhost:{settings.API_PORT}/ui/workflow")
    uvicorn.run("backend.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
