# api/index.py
# ─────────────────────────────────────────────────────────
# Thin Vercel proxy: serves UI files and forwards ALL API
# requests to the worker machine (your local computer).
#
# This file does NOT import backend.main or any heavy
# packages, so it fits within Vercel's 500 MB limit.
# ─────────────────────────────────────────────────────────
from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Config from env vars (set on Vercel dashboard) ───────
WORKER_BASE_URL = os.getenv("WORKER_BASE_URL", "").rstrip("/")
WORKER_SECRET = os.getenv("WORKER_SHARED_SECRET", "")
PROXY_TIMEOUT = int(os.getenv("WORKER_TIMEOUT_SEC", "180"))

UI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui"))

app = FastAPI(
    title="Thai Business Insight AI (Vercel Proxy)",
    description="Thin proxy that forwards requests to the worker machine.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────
def _worker_headers() -> dict[str, str]:
    h: dict[str, str] = {}
    if WORKER_SECRET:
        h["X-Worker-Secret"] = WORKER_SECRET
    return h


def _serve_ui_file(filename: str, content_type: str = "text/html") -> Response:
    path = os.path.join(UI_DIR, filename)
    if not os.path.exists(path):
        return Response(status_code=404, content="Not Found")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return Response(content=content, media_type=content_type)


async def _proxy_get(path: str, params: dict | None = None) -> Response:
    if not WORKER_BASE_URL:
        return JSONResponse({"error": "WORKER_BASE_URL not configured"}, status_code=503)
    try:
        async with httpx.AsyncClient(timeout=PROXY_TIMEOUT) as client:
            r = await client.get(
                f"{WORKER_BASE_URL}{path}",
                params=params,
                headers=_worker_headers(),
            )
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))
    except Exception as e:
        return JSONResponse({"error": f"Proxy error: {e}"}, status_code=502)


async def _proxy_post(path: str, body: Any = None, params: dict | None = None) -> Response:
    if not WORKER_BASE_URL:
        return JSONResponse({"error": "WORKER_BASE_URL not configured"}, status_code=503)
    try:
        async with httpx.AsyncClient(timeout=PROXY_TIMEOUT) as client:
            r = await client.post(
                f"{WORKER_BASE_URL}{path}",
                json=body,
                params=params,
                headers=_worker_headers(),
            )
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))
    except Exception as e:
        return JSONResponse({"error": f"Proxy error: {e}"}, status_code=502)


# ── UI routes ────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/UI_FORUSER")


@app.get("/UI_FORUSER", include_in_schema=False)
@app.get("/ui_foruser", include_in_schema=False)
def ui_foruser():
    return _serve_ui_file("ui_foruser.html")


@app.get("/ui", include_in_schema=False)
def ui_index():
    return _serve_ui_file("index.html")


@app.get("/ui/workflow", include_in_schema=False)
def ui_workflow():
    return _serve_ui_file("workflow.html")


@app.get("/ui/mlflow", include_in_schema=False)
@app.get("/mlflow_ui", include_in_schema=False)
def ui_mlflow():
    return _serve_ui_file("mlflow.html")


@app.get("/ui/app.js", include_in_schema=False)
def ui_app_js():
    return _serve_ui_file("app.js", "application/javascript")


@app.get("/ui/app_foruser.js", include_in_schema=False)
def ui_foruser_js():
    return _serve_ui_file("app_foruser.js", "application/javascript")


@app.get("/ui/workflow.js", include_in_schema=False)
def ui_workflow_js():
    return _serve_ui_file("workflow.js", "application/javascript")


@app.get("/ui/mlflow.js", include_in_schema=False)
def ui_mlflow_js():
    return _serve_ui_file("mlflow.js", "application/javascript")


@app.get("/ui/styles.css", include_in_schema=False)
def ui_styles():
    return _serve_ui_file("styles.css", "text/css")


@app.get("/ui/swagger_custom.js", include_in_schema=False)
def swagger_custom_js():
    return _serve_ui_file("swagger_custom.js", "application/javascript")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    path = os.path.join(UI_DIR, "favicon.ico")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return Response(content=f.read(), media_type="image/x-icon")
    return Response(status_code=204)


# ── Proxy: Health / Status ───────────────────────────────
@app.get("/health")
async def health():
    return await _proxy_get("/health")


@app.get("/worker/status")
async def worker_status():
    return await _proxy_get("/worker/status")


@app.get("/workflow/steps")
async def workflow_steps():
    return await _proxy_get("/workflow/steps")


@app.get("/documents")
async def documents():
    return await _proxy_get("/documents")


@app.get("/mlflow/summary")
async def mlflow_summary():
    return await _proxy_get("/mlflow/summary")


@app.get("/rag/preview")
async def rag_preview(namespace: str = "external", source_type: str = "", limit: int = 30):
    return await _proxy_get("/rag/preview", {"namespace": namespace, "source_type": source_type, "limit": limit})


# ── Proxy: Jobs ──────────────────────────────────────────
@app.get("/jobs/{job_id}/status")
async def job_status(job_id: str):
    return await _proxy_get(f"/jobs/{job_id}/status")


@app.get("/jobs/{job_id}/logs")
async def job_logs(job_id: str, tail: int = 200):
    return await _proxy_get(f"/jobs/{job_id}/logs", {"tail": tail})


# ── Proxy: Pipeline ──────────────────────────────────────
@app.post("/pipeline/external/scrape")
async def pipeline_external_scrape(request: Request):
    body = await request.json()
    return await _proxy_post("/pipeline/external/scrape", body)


@app.post("/pipeline/internal/upload_pdf")
async def pipeline_internal_upload_pdf(request: Request):
    """Proxy file upload to worker."""
    if not WORKER_BASE_URL:
        return JSONResponse({"error": "WORKER_BASE_URL not configured"}, status_code=503)
    try:
        form = await request.form()
        file = form.get("file")
        entity_hint = form.get("entity_hint", "internal_doc")
        if file is None:
            return JSONResponse({"error": "file required"}, status_code=400)
        file_bytes = await file.read()
        async with httpx.AsyncClient(timeout=PROXY_TIMEOUT) as client:
            r = await client.post(
                f"{WORKER_BASE_URL}/pipeline/internal/upload_pdf",
                params={"entity_hint": entity_hint},
                files={"file": (file.filename, file_bytes, file.content_type or "application/pdf")},
                headers=_worker_headers(),
            )
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))
    except Exception as e:
        return JSONResponse({"error": f"Proxy error: {e}"}, status_code=502)


# ── Proxy: Ask ───────────────────────────────────────────
@app.post("/ask")
async def ask(request: Request):
    body = await request.json()
    return await _proxy_post("/ask", body)


# ── Proxy: Facebook ──────────────────────────────────────
@app.post("/facebook/scrape_post")
async def facebook_scrape_post(request: Request):
    body = await request.json()
    return await _proxy_post("/facebook/scrape_post", body)


# ── Proxy: Eval ──────────────────────────────────────────
@app.post("/eval/run")
async def eval_run(request: Request):
    body = await request.json()
    return await _proxy_post("/eval/run", body)


# ── Proxy: RAG reset ─────────────────────────────────────
@app.post("/rag/reset")
async def rag_reset(request: Request):
    return await _proxy_post("/rag/reset", {})
