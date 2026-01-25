"""
backend/main.py

FastAPI รวมระบบ:
- /scrape -> scrape -> OCR queue -> (auto) postprocess -> image understanding -> ingest RAG
- /ocr/status/{job_id} -> ดูสถานะ OCR
- /pipeline/status/{job_id} -> ดูสถานะ pipeline ทั้งหมด
- /models -> list models from Ollama
- Serve UI: /ui, /ui/app.js, /ui/styles.css
"""

from __future__ import annotations

import os
import json
import time
import uuid
import importlib
import threading
from typing import Any, Dict, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import web_scraping
from image_understanding import understand_images

# ของคุณต้องมีจริง:
from rag_store import RAGStore, ingest_scrape_folder
from agent_flow import AgenticRAG, OllamaLLM
from report_utils import generate_report_md

# ----------------------------
# ENV (fix .env path ให้ชัวร์)
# ----------------------------
ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(ENV_PATH)

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")  # << สำคัญ

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
RAG_COLLECTION = os.getenv("RAG_COLLECTION", "rag_docs")

OUTPUT_BASE_DIR = getattr(web_scraping, "OUTPUT_BASE_DIR", os.path.join(os.getcwd(), "scraped_outputs"))

UI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui"))

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="AI Scraper + OCR + Agentic RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Global RAG + Agent
# ----------------------------
store = RAGStore(
    chroma_dir=CHROMA_DIR,
    collection_name=RAG_COLLECTION,
    ollama_host=OLLAMA_HOST,
    embed_model=OLLAMA_EMBED_MODEL,
)
llm = OllamaLLM(host=OLLAMA_HOST, model=OLLAMA_LLM_MODEL)
agent = AgenticRAG(store=store, llm=llm)

# ----------------------------
# Pipeline Status Registry
# ----------------------------
PIPELINE_LOCK = threading.Lock()
PIPELINE_STATUS: Dict[str, Dict[str, Any]] = {}  # job_id -> status dict


def set_stage(job_id: str, stage: str, message: str = "", extra: Optional[dict] = None) -> None:
    with PIPELINE_LOCK:
        st = PIPELINE_STATUS.setdefault(job_id, {})
        st["job_id"] = job_id
        st["stage"] = stage
        st["message"] = message
        st["updated_at"] = time.time()
        if extra:
            st.update(extra)


def get_stage(job_id: str) -> Dict[str, Any]:
    with PIPELINE_LOCK:
        return PIPELINE_STATUS.get(job_id, {"job_id": job_id, "stage": "unknown"})


# ----------------------------
# Ollama helpers
# ----------------------------
def ollama_models() -> Dict[str, Any]:
    try:
        import requests
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if r.status_code != 200:
            return {"reachable": False, "status_code": r.status_code, "models": []}
        data = r.json()
        models = [m.get("name") for m in (data.get("models") or []) if m.get("name")]
        return {"reachable": True, "models": models}
    except Exception as e:
        return {"reachable": False, "error": str(e), "models": []}


# ----------------------------
# Background: wait OCR then postprocess+ingest
# ----------------------------
def wait_ocr_done(job_id: str, timeout_s: int = 60 * 60) -> Dict[str, Any]:
    t0 = time.time()
    while True:
        with web_scraping.OCR_LOCK:
            st = web_scraping.OCR_STATUS.get(job_id)
        if not st:
            return {"status": "not_found"}
        status = st.get("status")
        if status in ("done", "error"):
            return st
        if time.time() - t0 > timeout_s:
            return {"status": "timeout", "last": st}
        time.sleep(1.0)


def run_postprocess_and_ingest(job_id: str) -> None:
    """
    ทำงานครบ:
    - รอ OCR done
    - image understanding (web images + pdf extracted images)
    - generate report.md
    - ingest เข้า Chroma (RAG)
    """
    set_stage(job_id, "waiting_ocr", "Waiting OCR job to finish...")
    ocr_st = wait_ocr_done(job_id)

    if ocr_st.get("status") == "error":
        set_stage(job_id, "ocr_error", "OCR failed", {"ocr": ocr_st})
        return

    if ocr_st.get("status") == "timeout":
        set_stage(job_id, "ocr_timeout", "OCR timeout", {"ocr": ocr_st})
        return

    main_folder = ocr_st.get("main_folder") or ""
    out_root = ocr_st.get("out_root") or ""

    if not main_folder or not os.path.isdir(main_folder):
        set_stage(job_id, "bad_folder", "main_folder not found", {"main_folder": main_folder})
        return

    # ---------- Vision step ----------
    models_info = ollama_models()
    vision_ok = models_info.get("reachable") and (OLLAMA_VISION_MODEL in (models_info.get("models") or []))

    if vision_ok:
        set_stage(job_id, "vision_web", "Running image understanding for web images...")
        # 1) web images per site
        for name in sorted(os.listdir(main_folder)):
            site_folder = os.path.join(main_folder, name)
            if not os.path.isdir(site_folder) or not (name[:1].isdigit() and "_" in name):
                continue
            images_dir = os.path.join(site_folder, "images")
            if not os.path.isdir(images_dir):
                continue

            ctx = ""
            content_path = os.path.join(site_folder, "content.txt")
            if os.path.exists(content_path):
                try:
                    with open(content_path, "r", encoding="utf-8") as f:
                        ctx = f.read()
                except Exception:
                    ctx = ""

            out_jsonl = os.path.join(site_folder, "images_understanding.jsonl")
            print(f"🧠 Vision(web): {name} -> {out_jsonl}", flush=True)
            understand_images(
                images_dir=images_dir,
                out_jsonl=out_jsonl,
                context_text=ctx,
                context_map=None,
                ollama_host=OLLAMA_HOST,
                model=OLLAMA_VISION_MODEL,
                max_images=10000,
                sleep_s=0.0,
                dedupe=True,
            )

        # 2) pdf extracted images understanding
        set_stage(job_id, "vision_pdf", "Running image understanding for PDF extracted images...")
        if out_root and os.path.isdir(out_root):
            for pdf_name in sorted(os.listdir(out_root)):
                pdf_dir = os.path.join(out_root, pdf_name)
                if not os.path.isdir(pdf_dir):
                    continue

                docs_jsonl = os.path.join(pdf_dir, "docs.jsonl")
                images_dir = os.path.join(pdf_dir, "images")
                if not os.path.exists(docs_jsonl) or not os.path.isdir(images_dir):
                    continue

                ctx_map = {}
                try:
                    with open(docs_jsonl, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            if obj.get("type") == "figure":
                                ipath = obj.get("image_path") or ""
                                hint = obj.get("caption_hint") or ""
                                if ipath and hint:
                                    ctx_map[ipath] = hint
                except Exception:
                    ctx_map = {}

                out_jsonl = os.path.join(pdf_dir, "pdf_images_understanding.jsonl")
                print(f"🧠 Vision(pdf): {pdf_name} -> {out_jsonl}", flush=True)
                understand_images(
                    images_dir=images_dir,
                    out_jsonl=out_jsonl,
                    context_text="",
                    context_map=ctx_map,
                    ollama_host=OLLAMA_HOST,
                    model=OLLAMA_VISION_MODEL,
                    max_images=10000,
                    sleep_s=0.0,
                    dedupe=True,
                )

        set_stage(job_id, "vision_done", "Vision completed")
    else:
        set_stage(job_id, "vision_skipped", f"Vision skipped (model not available): {OLLAMA_VISION_MODEL}",
                  {"available_models": models_info.get("models", [])})

    # ---------- Report ----------
    set_stage(job_id, "report", "Generating report.md...")
    scrape_result_stub = {
        "file_path": os.path.join(main_folder, "final_data.csv"),
        "ocr_job_id": job_id,
        "ocr_status_url": f"/ocr/status/{job_id}",
        "data": [],
    }
    report_path = generate_report_md(main_folder=main_folder, scrape_result=scrape_result_stub)

    # ---------- Ingest ----------
    set_stage(job_id, "ingesting", "Ingesting into RAG (ChromaDB)...")
    print(f"📦 Ingesting into Chroma: dir={CHROMA_DIR} collection={RAG_COLLECTION}", flush=True)
    ingest_summary = ingest_scrape_folder(store=store, main_folder=main_folder)

    set_stage(job_id, "ready", "Pipeline completed ✅", {
        "main_folder": main_folder,
        "ocr_out_root": out_root,
        "report_path": report_path,
        "ingest": ingest_summary,
        "chroma_dir": os.path.abspath(CHROMA_DIR),
        "collection": RAG_COLLECTION,
    })


# ----------------------------
# Preflight / Health
# ----------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    deps = {
        "playwright": "playwright.sync_api",
        "bs4": "bs4",
        "pandas": "pandas",
        "fitz": "fitz",
        "PIL": "PIL",
        "paddleocr": "paddleocr",
        "chromadb": "chromadb",
        "requests": "requests",
    }
    dep_ok = {}
    for k, mod in deps.items():
        try:
            importlib.import_module(mod)
            dep_ok[k] = True
        except Exception:
            dep_ok[k] = False

    models = ollama_models()
    return {
        "env_path": ENV_PATH,
        "deps": dep_ok,
        "ui": {
            "dir": UI_DIR,
            "exists": os.path.isdir(UI_DIR),
            "index": os.path.exists(os.path.join(UI_DIR, "index.html")),
            "app_js": os.path.exists(os.path.join(UI_DIR, "app.js")),
            "styles": os.path.exists(os.path.join(UI_DIR, "styles.css")),
        },
        "ollama": {
            "host": OLLAMA_HOST,
            "reachable": models.get("reachable"),
            "models": models.get("models"),
            "llm_model": OLLAMA_LLM_MODEL,
            "embed_model": OLLAMA_EMBED_MODEL,
            "vision_model": OLLAMA_VISION_MODEL,
            "vision_ok": (OLLAMA_VISION_MODEL in (models.get("models") or [])) if models.get("reachable") else False,
        },
        "rag": {
            "chroma_dir": os.path.abspath(CHROMA_DIR),
            "collection": RAG_COLLECTION,
            "exists": os.path.isdir(CHROMA_DIR),
        }
    }


@app.get("/models")
def models() -> Dict[str, Any]:
    return {
        "ollama_host": OLLAMA_HOST,
        **ollama_models()
    }


# ----------------------------
# UI Serving
# ----------------------------
@app.get("/")
def root():
    return RedirectResponse(url="/ui")


@app.get("/ui")
def ui_index():
    return FileResponse(os.path.join(UI_DIR, "index.html"))


@app.get("/ui/app.js")
def ui_app_js():
    return FileResponse(os.path.join(UI_DIR, "app.js"))


@app.get("/ui/styles.css")
def ui_styles():
    return FileResponse(os.path.join(UI_DIR, "styles.css"))


# ----------------------------
# Pipeline Endpoints
# ----------------------------
@app.get("/pipeline/status/{job_id}")
def pipeline_status(job_id: str) -> Dict[str, Any]:
    return get_stage(job_id)


# ----------------------------
# External: scrape -> OCR -> auto postprocess+ingest
# ----------------------------
@app.get("/scrape")
def scrape(keyword: str, amount: int = 5, auto_postprocess: int = 1) -> Dict[str, Any]:
    """
    เรียก scrape
    - auto_postprocess=1 จะรัน postprocess+ingest อัตโนมัติหลัง OCR done
    """
    set_stage("GLOBAL", "scrape_called", f"keyword={keyword} amount={amount}")

    print(f"🔔 /scrape keyword='{keyword}' amount={amount}", flush=True)
    set_stage("GLOBAL", "scraping", "Scraping in progress...")

    result = web_scraping.run_scraper_logic(keyword=keyword, max_links=amount)

    if result.get("status") != "success":
        return result

    job_id = result.get("ocr_job_id")
    main_folder = result.get("folder")

    set_stage(job_id, "ocr_queued", "OCR queued", {"main_folder": main_folder})

    # auto pipeline in background thread
    if auto_postprocess and job_id:
        t = threading.Thread(target=run_postprocess_and_ingest, args=(job_id,), daemon=True)
        t.start()
        print(f"🧩 Auto postprocess started (job_id={job_id})", flush=True)

    return result


@app.get("/ocr/status/{job_id}")
def ocr_status(job_id: str) -> Dict[str, Any]:
    with web_scraping.OCR_LOCK:
        st = web_scraping.OCR_STATUS.get(job_id)
    if not st:
        return {"status": "not_found", "job_id": job_id}
    return {"job_id": job_id, **st}


@app.post("/postprocess/{job_id}")
def postprocess(job_id: str) -> Dict[str, Any]:
    """
    กรณีอยากสั่งเอง
    """
    t = threading.Thread(target=run_postprocess_and_ingest, args=(job_id,), daemon=True)
    t.start()
    return {"status": "started", "job_id": job_id}


# ----------------------------
# Internal: upload pdf -> OCR -> ingest
# ----------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    base = os.path.join(OUTPUT_BASE_DIR, "internal_uploads", job_id)
    os.makedirs(base, exist_ok=True)

    raw_path = os.path.join(base, file.filename)
    with open(raw_path, "wb") as f:
        f.write(await file.read())

    set_stage(job_id, "upload_saved", "File saved", {"saved_path": raw_path})

    import ocr_pipeline
    out_root = os.path.join(base, "ocr_results")
    os.makedirs(out_root, exist_ok=True)

    set_stage(job_id, "ocr_running", "OCR running for uploaded PDF...")
    ocr_pipeline.process_folder_pdfs(files_dir=base, out_root=out_root, progress_cb=None)

    set_stage(job_id, "ingesting", "Ingesting upload into RAG...")
    ingest_summary = ingest_scrape_folder(store=store, main_folder=base)

    set_stage(job_id, "ready", "Upload pipeline completed ✅", {
        "saved_path": raw_path,
        "ocr_out_root": out_root,
        "ingest": ingest_summary,
        "chroma_dir": os.path.abspath(CHROMA_DIR),
        "collection": RAG_COLLECTION,
    })

    return get_stage(job_id)


# ----------------------------
# Ask (Agentic RAG)
# ----------------------------
@app.post("/ask")
def ask(payload: Dict[str, Any]) -> Dict[str, Any]:
    q = (payload.get("question") or "").strip()
    if not q:
        return {"status": "bad_request", "message": "question required"}
    top_k = int(payload.get("top_k") or 10)
    out = agent.answer(question=q, top_k=top_k)
    return {"status": "ok", **out}


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    print(f"\n✅ Swagger UI: http://localhost:{API_PORT}/docs")
    print(f"✅ UI:         http://localhost:{API_PORT}/ui")
    print(f"✅ Health:     http://localhost:{API_PORT}/health")
    print(f"✅ Models:     http://localhost:{API_PORT}/models\n")
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
