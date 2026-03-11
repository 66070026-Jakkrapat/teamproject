# api/index.py
# ─────────────────────────────────────────────────────────
# Vercel serverless entry point.
#
# ทำงานจริงบน Vercel — ใช้ OpenAI GPT API + RAG (ถ้ามี DB)
# Deploy version: 2026-03-11T14:52
# ─────────────────────────────────────────────────────────
from __future__ import annotations

import os
import sys
import traceback
from typing import Any

# ── Ensure project root is on sys.path ───────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Lightweight imports ──────────────────────────────────
import httpx
from fastapi import FastAPI, Request, Response, File, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Read env vars ────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")

UI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui"))

# ── FastAPI app ──────────────────────────────────────────
app = FastAPI(title="Thai Business Insight AI", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lazy-loaded singletons ───────────────────────────────
_llm = None
_store = None
_agent = None
_init_done = False
_init_error = ""


async def _ensure_init():
    """Lazy-init: ทำครั้งเดียวแล้ว cache"""
    global _llm, _store, _agent, _init_done, _init_error
    if _init_done:
        return
    _init_done = True

    # 1) LLM
    try:
        from backend.llm_client import create_llm, create_embedder
        _llm = create_llm()
    except Exception as e:
        _init_error = f"LLM init failed: {e}"
        traceback.print_exc()
        return

    # 2) RAG Store + Agent (ถ้ามี DB)
    if DATABASE_URL:
        try:
            from backend.settings import settings
            from backend.rag.rag_store import RAGStore
            from backend.agent.agent_flow import AgenticRAG

            embedder = create_embedder(dims=settings.EMBED_DIMS)
            _store = RAGStore(
                database_url=DATABASE_URL,
                embedder=embedder,
                embed_dims=settings.EMBED_DIMS,
            )
            await _store.init_db()
            _agent = AgenticRAG(store=_store)
        except Exception as e:
            _init_error = f"RAG init failed (continuing without): {e}"
            traceback.print_exc()
            # ไม่ fatal — ยังตอบ direct LLM ได้


# ── UI file serving ──────────────────────────────────────
def _serve_ui(filename: str, content_type: str = "text/html") -> Response:
    path = os.path.join(UI_DIR, filename)
    if not os.path.exists(path):
        return Response(status_code=404, content="Not Found")
    mode = "rb" if content_type.startswith("image") else "r"
    encoding = None if mode == "rb" else "utf-8"
    with open(path, mode, encoding=encoding) as f:
        data = f.read()
    return Response(content=data, media_type=content_type)


# ── UI Routes ────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/UI_FORUSER")


@app.get("/UI_FORUSER", include_in_schema=False)
@app.get("/ui_foruser", include_in_schema=False)
def ui_foruser():
    return _serve_ui("ui_foruser.html")


@app.get("/ui", include_in_schema=False)
def ui_index():
    return _serve_ui("index.html")


@app.get("/ui/workflow", include_in_schema=False)
def ui_workflow():
    return _serve_ui("workflow.html")


@app.get("/mlflow_ui", include_in_schema=False)
@app.get("/ui/mlflow", include_in_schema=False)
def ui_mlflow():
    return _serve_ui("mlflow.html")


@app.get("/ui/app.js", include_in_schema=False)
def ui_app_js():
    return _serve_ui("app.js", "application/javascript")


@app.get("/ui/app_foruser.js", include_in_schema=False)
def ui_foruser_js():
    return _serve_ui("app_foruser.js", "application/javascript")


@app.get("/ui/workflow.js", include_in_schema=False)
def ui_workflow_js():
    return _serve_ui("workflow.js", "application/javascript")


@app.get("/ui/mlflow.js", include_in_schema=False)
def ui_mlflow_js():
    return _serve_ui("mlflow.js", "application/javascript")


@app.get("/ui/styles.css", include_in_schema=False)
def ui_styles():
    return _serve_ui("styles.css", "text/css")


@app.get("/ui/swagger_custom.js", include_in_schema=False)
def swagger_js():
    return _serve_ui("swagger_custom.js", "application/javascript")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    path = os.path.join(UI_DIR, "favicon.ico")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return Response(content=f.read(), media_type="image/x-icon")
    return Response(status_code=204)


# ══════════════════════════════════════════════════════════
#  API Endpoints
# ══════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    await _ensure_init()
    return {
        "status": "ok",
        "llm_provider": LLM_PROVIDER,
        "has_openai_key": bool(OPENAI_API_KEY),
        "has_database": bool(DATABASE_URL),
        "agent_ready": _agent is not None,
        "init_error": _init_error or None,
    }


@app.get("/worker/status")
async def worker_status():
    await _ensure_init()
    return {
        "status": "ok",
        "mode": "direct_openai",
        "agent_ready": _agent is not None,
        "message": "Running on Vercel with OpenAI API",
    }


@app.get("/workflow/steps")
async def workflow_steps():
    return {"steps": []}


# ── /ask — หัวใจของระบบ ──────────────────────────────────
@app.post("/ask")
async def ask(request: Request):
    await _ensure_init()

    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"status": "bad_request", "message": "question required"}, 400)

    top_k = int(body.get("top_k") or 8)

    # ── Path A: Full Agent + RAG ──────────────────────────
    if _agent is not None:
        try:
            result = await _agent.answer(question, top_k=top_k)
            warnings = []
            for meta in result.get("chunks") or []:
                w = (meta or {}).get("warning")
                if w and w not in warnings:
                    warnings.append(w)
            if warnings:
                result["warnings"] = warnings
            return {"status": "ok", **result}
        except Exception as e:
            traceback.print_exc()
            # fall through to direct LLM

    # ── Path B: Direct LLM (ไม่มี RAG) ───────────────────
    if _llm is not None:
        try:
            prompt = (
                "คุณเป็นผู้เชี่ยวชาญด้านธุรกิจและเทรนด์ธุรกิจไทย\n"
                "ตอบคำถามต่อไปนี้เป็นภาษาไทย อย่างละเอียดและมีข้อมูลสนับสนุน\n"
                "ถ้าไม่ทราบ ให้ตอบว่าไม่ทราบ อย่าแต่งข้อมูลขึ้นมา\n\n"
                f"คำถาม: {question}"
            )
            answer = await _llm.generate(prompt)
            return {
                "status": "ok",
                "route": "direct_llm",
                "answer": answer,
                "chunks": [],
                "tavily_used": False,
            }
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "message": f"LLM call failed: {e}"}

    # ── Path C: Nothing works ─────────────────────────────
    return {
        "status": "error",
        "message": f"ระบบยังไม่พร้อมใช้งาน กรุณาตรวจสอบ OPENAI_API_KEY. Error: {_init_error}",
    }


# ── /documents ───────────────────────────────────────────
@app.get("/documents")
async def documents():
    await _ensure_init()
    if _store is None:
        return {"status": "ok", "documents": [], "count": 0}
    try:
        from backend.rag.rag_store import RAGChunk
        from sqlalchemy import select, func
        async with _store.SessionLocal() as db:
            stmt = (
                select(
                    RAGChunk.namespace,
                    RAGChunk.source_path,
                    RAGChunk.source_type,
                    func.count(RAGChunk.id).label("chunk_count"),
                )
                .group_by(RAGChunk.namespace, RAGChunk.source_path, RAGChunk.source_type)
                .limit(100)
            )
            result = await db.execute(stmt)
            rows = result.all()
        docs = []
        for ns, path, stype, count in rows:
            basename = os.path.basename(path) if path else stype or "unknown"
            docs.append({
                "namespace": ns or "",
                "source_path": path or "",
                "source_type": stype or "",
                "filename": basename,
                "type": (stype or "UNKNOWN").upper(),
                "chunk_count": count,
                "table_row_count": 0,
                "status": "completed",
                "updated_at": "",
            })
        return {"status": "ok", "documents": docs, "count": len(docs)}
    except Exception as e:
        traceback.print_exc()
        return {"status": "ok", "documents": [], "count": 0}


# ── /rag/preview ─────────────────────────────────────────
@app.get("/rag/preview")
async def rag_preview(namespace: str = "external", source_type: str = "", limit: int = 30):
    await _ensure_init()
    if _store is None:
        return {"status": "ok", "namespace": namespace, "count": 0, "chunks": []}
    try:
        chunks = await _store.preview_chunks(namespace=namespace, source_type=source_type, limit=limit)
        return {"status": "ok", "namespace": namespace, "count": len(chunks), "chunks": chunks}
    except Exception:
        return {"status": "ok", "namespace": namespace, "count": 0, "chunks": []}


# ── /mlflow/summary ──────────────────────────────────────
@app.get("/mlflow/summary")
async def mlflow_summary():
    return {"status": "ok", "enabled": False, "message": "MLflow not available on Vercel"}


# ── Jobs (not supported on Vercel serverless) ────────────
@app.get("/jobs/{job_id}/status")
async def job_status(job_id: str):
    return {"job_id": job_id, "stage": "unknown", "message": "Jobs not available in serverless mode"}


@app.get("/jobs/{job_id}/logs")
async def job_logs(job_id: str, tail: int = 200):
    return {"job_id": job_id, "logs": []}


# ── Pipeline endpoints (not available on Vercel) ─────────
@app.post("/pipeline/external/scrape")
async def pipeline_external_scrape():
    return JSONResponse(
        {"status": "error", "message": "Scraping ไม่สามารถทำบน Vercel ได้ กรุณาใช้ local server"},
        status_code=501,
    )


@app.post("/api/pdf_summary")
async def api_pdf_summary(file: UploadFile = File(...)):
    await _ensure_init()
    if _llm is None:
        return JSONResponse({"status": "error", "message": "LLM not initialized. Check OPENAI_API_KEY."}, 500)
    
    try:
        import pypdf
        import io
        
        content = await file.read()
        reader = pypdf.PdfReader(io.BytesIO(content))
        text = ""
        # Limit to 10 pages to avoid hitting token limits
        for i in range(min(10, len(reader.pages))):
            extracted = reader.pages[i].extract_text()
            if extracted:
                text += extracted + "\n"
        
        prompt = (
            f"คุณเป็นผู้เชี่ยวชาญด้านการวิเคราะห์ข้อมูล วิจารณ์และสรุปเอกสาร PDF ต่อไปนี้ให้กระชับ "
            f"ดึงประเด็นสำคัญที่สุดออกมา 3-5 ข้อ (เป็นภาษาไทย)\n\n"
            f"ชื่อไฟล์: {file.filename}\n\nเนื้อหาบางส่วน:\n{text[:15000]}"
        )
        answer = await _llm.generate(prompt)
        return {"status": "ok", "summary": answer, "filename": file.filename}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": f"Failed to parse PDF: {str(e)}"}, 500)


@app.post("/facebook/scrape_post")
async def facebook_scrape():
    return JSONResponse(
        {"status": "error", "message": "Facebook scraping not available on Vercel"},
        status_code=501,
    )


@app.post("/rag/reset")
async def rag_reset():
    await _ensure_init()
    if _store is None:
        return JSONResponse({"status": "error", "message": "No database configured"}, 501)
    try:
        await _store.reset_db()
        return {"status": "ok", "message": "db reset done"}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, 500)


@app.post("/eval/run")
async def eval_run():
    return JSONResponse(
        {"status": "error", "message": "Eval not available on Vercel"},
        status_code=501,
    )
