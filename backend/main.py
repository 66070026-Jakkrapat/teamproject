# backend/main.py
from __future__ import annotations

import os
import time
import uuid
import threading
import sys
import traceback
import asyncio
import json
import requests
import httpx
from typing import Any, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Body, Request, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi import Response
from backend.settings import settings
from backend.utils.job_manager import (
    new_job, set_stage, log,
    get_status, get_logs, get_workflow_steps
)

# ✅ set env Paddle ไว้ตรงนี้ (กันหลุด)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "true")

# --- Core imports (may fail on Vercel if DB packages are incomplete) ---
try:
    from backend.rag.rag_store import RAGStore, RAGChunk, RAGFact
    from backend.rag.ingest import ingest_main_folder, ingest_text_blob
    from backend.agent.agent_flow import AgenticRAG
    from backend.agent import prompts as agent_prompts
    from backend.report_utils.report import generate_report_md
    _CORE_OK = True
except Exception:
    _CORE_OK = False

# --- Heavy imports (may not be available on Vercel) ---
try:
    from backend.vision.image_understanding import warmup_blip, caption_images_with_blip_and_translate
    _VISION_OK = True
except Exception:
    _VISION_OK = False
    def warmup_blip(*a, **kw): pass
    def caption_images_with_blip_and_translate(*a, **kw): return {}

try:
    from backend.scraping.web_scraping import run_external_scrape
    from backend.scraping.facebook_scraping import scrape_facebook_post_html
    _SCRAPE_OK = True
except Exception:
    _SCRAPE_OK = False

try:
    from backend.ocr.ocr_pipeline import process_folder_pdfs
    _OCR_OK = True
except Exception:
    _OCR_OK = False

try:
    from backend.evaluation.eval_runner import run_eval
    _EVAL_OK = True
except Exception:
    _EVAL_OK = False

try:
    from backend.observability.mlflow_tracker import PromptSpec, mlflow_tracker
    _MLFLOW_OK = True
except Exception:
    _MLFLOW_OK = False
    # Provide a no-op tracker stub
    class _NoOpTracker:
        enabled = False
        client = None
        def start_run(self, *a, **kw):
            from contextlib import contextmanager
            @contextmanager
            def _noop(): yield
            return _noop()
        def log_params(self, *a, **kw): pass
        def log_metrics(self, *a, **kw): pass
        def log_dict(self, *a, **kw): pass
        def log_text(self, *a, **kw): pass
        def log_artifacts(self, *a, **kw): pass
        def log_directory(self, *a, **kw): pass
        def log_prompt_registry(self, *a, **kw): return {}
        def log_pipeline_snapshot(self, *a, **kw): pass
    mlflow_tracker = _NoOpTracker()
    class PromptSpec:
        def __init__(self, **kw): pass

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

UI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ui"))

# --- Lazy init: only create store/agent if core packages + DATABASE_URL are available ---
store = None
agent = None
if _CORE_OK and settings.DATABASE_URL:
    try:
        from backend.llm_client import create_embedder
        store = RAGStore(
            database_url=settings.DATABASE_URL,
            embedder=create_embedder(dims=settings.EMBED_DIMS),
            embed_dims=settings.EMBED_DIMS,
        )
        agent = AgenticRAG(store=store)
    except Exception:
        store = None
        agent = None



def _safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _summarize_scrape(scrape: Dict[str, Any]) -> Dict[str, Any]:
    items = scrape.get("items") or []
    return {
        "keyword": scrape.get("keyword", ""),
        "main_folder": scrape.get("main_folder", ""),
        "csv_path": scrape.get("csv_path", ""),
        "url_count": len(scrape.get("urls") or []),
        "item_count": len(items),
        "downloaded_images": sum(int(it.get("downloaded_images") or 0) for it in items),
        "downloaded_files": sum(int(it.get("downloaded_files") or 0) for it in items),
        "pdf_extracted": sum(int(it.get("pdf_extracted") or 0) for it in items),
    }


def _summarize_ocr(out_root: str) -> Dict[str, Any]:
    summary = {
        "ocr_docs_jsonl": 0,
        "ocr_pages": 0,
        "ocr_figures": 0,
        "ocr_text_layer_pages": 0,
        "ocr_paddle_pages": 0,
    }
    if not os.path.isdir(out_root):
        return summary

    for root, _, files in os.walk(out_root):
        for fn in files:
            if fn != "docs.jsonl":
                continue
            summary["ocr_docs_jsonl"] += 1
            docs_path = os.path.join(root, fn)
            with open(docs_path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") == "page_text":
                        summary["ocr_pages"] += 1
                        method = str(obj.get("extract_method") or "")
                        if method == "text_layer":
                            summary["ocr_text_layer_pages"] += 1
                        elif method.startswith("paddleocr"):
                            summary["ocr_paddle_pages"] += 1
                    elif obj.get("type") == "figure":
                        summary["ocr_figures"] += 1
    return summary


def _document_display_name(source_path: str, source_type: str) -> str:
    if not source_path:
        return source_type or "unknown"
    base = os.path.basename(source_path.rstrip("\\/"))
    if base.lower() in {"content.txt", "outline.json"}:
        parent = os.path.basename(os.path.dirname(source_path.rstrip("\\/")))
        return parent or base
    return base


def _document_type_label(source_type: str) -> str:
    source_type = (source_type or "").strip().lower()
    if "internal_pdf" in source_type:
        return "INTERNAL_PDF"
    if "web" in source_type:
        return "WEB_INGEST"
    if source_type:
        return source_type.upper()
    return "UNKNOWN"


async def _collect_documents_summary() -> list[Dict[str, Any]]:
    async with store.SessionLocal() as db:
        chunk_rows = (await db.execute(
            RAGChunk.__table__.select().order_by(RAGChunk.namespace, RAGChunk.source_path, RAGChunk.chunk_index)
        )).mappings().all()
        fact_rows = (await db.execute(
            RAGFact.__table__.select().order_by(RAGFact.namespace, RAGFact.source_path)
        )).mappings().all()

    docs: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for row in chunk_rows:
        key = (row["namespace"] or "", row["source_path"] or "", row["source_type"] or "")
        item = docs.setdefault(key, {
            "namespace": row["namespace"] or "",
            "source_path": row["source_path"] or "",
            "source_type": row["source_type"] or "",
            "filename": _document_display_name(row["source_path"] or "", row["source_type"] or ""),
            "type": _document_type_label(row["source_type"] or ""),
            "chunk_count": 0,
            "table_row_count": 0,
            "pages": set(),
            "status": "completed",
            "updated_at": "",
        })
        item["chunk_count"] += 1
        if int(row.get("page") or 0) > 0:
            item["pages"].add(int(row.get("page") or 0))
        source_path = row["source_path"] or ""
        if source_path and os.path.exists(source_path):
            ts = os.path.getmtime(source_path)
            iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
            if iso > item["updated_at"]:
                item["updated_at"] = iso

    for row in fact_rows:
        key = (row["namespace"] or "", row["source_path"] or "", row["source_type"] or "")
        item = docs.setdefault(key, {
            "namespace": row["namespace"] or "",
            "source_path": row["source_path"] or "",
            "source_type": row["source_type"] or "",
            "filename": _document_display_name(row["source_path"] or "", row["source_type"] or ""),
            "type": _document_type_label(row["source_type"] or ""),
            "chunk_count": 0,
            "table_row_count": 0,
            "pages": set(),
            "status": "completed",
            "updated_at": "",
        })
        item["table_row_count"] += 1

    out = []
    for item in docs.values():
        pages = sorted(item.pop("pages"))
        item["page_count"] = len(pages)
        out.append(item)

    out.sort(key=lambda x: (x.get("updated_at") or "", x.get("filename") or ""), reverse=True)
    return out


def _format_metric_value(value: Any, kind: str = "ratio") -> str:
    if value is None:
        return "--"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if kind == "percent":
        return f"{round(value * 100)}%"
    if kind == "ms":
        return f"{round(value)}ms"
    return f"{value:.2f}"


def _collect_mlflow_summary() -> Dict[str, Any]:
    if not mlflow_tracker.enabled or not mlflow_tracker.client:
        return {"enabled": False, "tracking_uri": settings.MLFLOW_TRACKING_URI, "message": "MLflow is disabled"}

    experiment_names = [
        settings.MLFLOW_EXPERIMENT,
        settings.MLFLOW_PROMPT_EXPERIMENT,
    ]
    experiments = []
    for name in experiment_names:
        if not name:
            continue
        exp = mlflow_tracker.client.get_experiment_by_name(name)
        if exp:
            experiments.append({"id": exp.experiment_id, "name": exp.name})

    if not experiments:
        return {"enabled": True, "tracking_uri": settings.MLFLOW_TRACKING_URI, "experiments": [], "runs": []}

    runs = []
    for exp in experiments:
        found = mlflow_tracker.client.search_runs(
            experiment_ids=[exp["id"]],
            max_results=12,
            order_by=["attributes.start_time DESC"],
        )
        for run in found:
            runs.append({
                "experiment": exp["name"],
                "run_name": run.data.tags.get("mlflow.runName", run.info.run_id),
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": int(run.info.start_time or 0),
                "metrics": dict(run.data.metrics or {}),
                "params": dict(run.data.params or {}),
                "tags": dict(run.data.tags or {}),
            })

    runs.sort(key=lambda x: x["start_time"], reverse=True)
    runs = runs[:20]

    preferred_cards = [
        ("Accuracy", ["mean_answer_relevance", "answer_relevance"], "percent"),
        ("Recall", ["mean_semantic_recall@k", "semantic_recall@k", "recall@k"], "ratio"),
        ("Faithfulness", ["mean_faithfulness", "faithfulness"], "percent"),
        ("Relevance", ["mean_context_precision", "semantic_context_precision"], "percent"),
    ]
    cards = []
    for label, candidates, kind in preferred_cards:
        value = None
        for run in runs:
            for key in candidates:
                if key in run["metrics"]:
                    value = run["metrics"][key]
                    break
            if value is not None:
                break
        cards.append({"label": label, "value": _format_metric_value(value, kind), "raw": value})

    line_keys = [
        ("Answer Relevance", "answer_relevance"),
        ("Faithfulness", "faithfulness"),
        ("Semantic Recall", "semantic_recall@k"),
    ]
    line_series = []
    recent_reversed = list(reversed(runs[:8]))
    for label, key in line_keys:
        values = []
        for run in recent_reversed:
            metric = run["metrics"].get(key)
            if metric is not None:
                values.append({"x": run["run_name"][:18], "y": float(metric)})
        if values:
            line_series.append({"label": label, "values": values})

    bar_metrics = []
    aggregate_keys = [
        ("Added Chunks", "ingest_added_chunks"),
        ("Added Facts", "ingest_added_facts"),
        ("Scraped URLs", "scrape_url_count"),
        ("OCR Pages", "ocr_pages"),
    ]
    for label, key in aggregate_keys:
        value = None
        for run in runs:
            if key in run["metrics"]:
                value = run["metrics"][key]
                break
        if value is not None:
            bar_metrics.append({"label": label, "value": float(value)})

    return {
        "enabled": True,
        "tracking_uri": settings.MLFLOW_TRACKING_URI,
        "experiments": experiments,
        "cards": cards,
        "line_series": line_series,
        "bar_metrics": bar_metrics,
        "runs": runs[:10],
    }


def _sync_prompt_registry() -> None:
    prompts = [
        PromptSpec(
            name="router_prompt",
            text=agent_prompts.ROUTER_PROMPT,
            tool_name="router",
            description="Route questions into semantic or structured retrieval.",
        ),
        PromptSpec(
            name="synth_prompt",
            text=agent_prompts.SYNTH_PROMPT,
            tool_name="answer_synthesizer",
            description="Grounded answer synthesis from retrieved context.",
        ),
        PromptSpec(
            name="tavily_prompt",
            text=agent_prompts.TAVILY_PROMPT,
            tool_name="web_fallback",
            description="Fallback web answer synthesis from Tavily snippets.",
        ),
    ]
    mlflow_tracker.log_prompt_registry(prompts)


def _register_pipeline_snapshot() -> None:
    manifest = {
        "pipeline_name": settings.MLFLOW_PIPELINE_NAME,
        "api_entrypoint": "backend.main:app",
        "ui_routes": ["/UI_FORUSER", "/ui", "/ui/workflow"],
        "components": {
            "scraping": "backend.scraping.web_scraping",
            "ocr": "backend.ocr.ocr_pipeline",
            "rag_store": "backend.rag.rag_store",
            "agent": "backend.agent.agent_flow",
            "evaluation": "backend.evaluation.eval_runner",
        },
        "models": {
            "embed_model": settings.EMBED_MODEL,
            "llm_model": settings.OLLAMA_LLM_MODEL,
            "blip_model": settings.BLIP_MODEL,
        },
    }
    files = [
        os.path.join(os.path.dirname(__file__), "main.py"),
        os.path.join(os.path.dirname(__file__), "agent", "prompts.py"),
        os.path.join(os.path.dirname(__file__), "evaluation", "eval_runner.py"),
    ]
    mlflow_tracker.log_pipeline_snapshot(manifest, files=files)


def _worker_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if settings.WORKER_SHARED_SECRET:
        headers["X-Worker-Secret"] = settings.WORKER_SHARED_SECRET
    return headers


def _remote_job_id(job_id: str) -> str:
    return f"remote:{job_id}"


def _split_remote_job_id(job_id: str) -> tuple[bool, str]:
    if job_id.startswith("remote:"):
        return True, job_id.split(":", 1)[1]
    return False, job_id


def _require_worker_secret(request: Request) -> None:
    if not settings.WORKER_SHARED_SECRET:
        return
    given = request.headers.get("X-Worker-Secret", "")
    if given != settings.WORKER_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="invalid worker secret")


def _should_proxy_to_worker() -> bool:
    return bool(settings.HYBRID_WORKER_ENABLED and settings.WORKER_BASE_URL)


def _proxy_worker_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{settings.WORKER_BASE_URL}{path}"
    response = requests.post(
        url,
        json=payload,
        headers=_worker_headers(),
        timeout=settings.WORKER_TIMEOUT_SEC,
    )
    response.raise_for_status()
    return response.json()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ startup
    os.makedirs(settings.OUTPUT_BASE_DIR, exist_ok=True)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    if store:
        await store.init_db()
    _sync_prompt_registry()
    if settings.MLFLOW_REGISTER_PIPELINE:
        _register_pipeline_snapshot()
    # Warmup BLIP model (skip on Vercel where vision modules are unavailable)
    if _VISION_OK:
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
    return RedirectResponse(url="/UI_FORUSER")


@app.get("/ui", include_in_schema=False)
def ui_index():
    return FileResponse(os.path.join(UI_DIR, "index.html"))


@app.get("/UI_FORUSER", include_in_schema=False)
@app.get("/ui_foruser", include_in_schema=False)
def ui_foruser():
    return FileResponse(os.path.join(UI_DIR, "ui_foruser.html"))


@app.get("/ui/workflow", include_in_schema=False)
def ui_workflow():
    return FileResponse(os.path.join(UI_DIR, "workflow.html"))


@app.get("/ui/mlflow", include_in_schema=False)
@app.get("/mlflow_ui", include_in_schema=False)
def ui_mlflow():
    return FileResponse(os.path.join(UI_DIR, "mlflow.html"))


@app.get("/ui/app.js", include_in_schema=False)
def ui_app_js():
    return FileResponse(os.path.join(UI_DIR, "app.js"))


@app.get("/ui/app_foruser.js", include_in_schema=False)
def ui_foruser_js():
    return FileResponse(os.path.join(UI_DIR, "app_foruser.js"))


@app.get("/ui/workflow.js", include_in_schema=False)
def ui_workflow_js():
    return FileResponse(os.path.join(UI_DIR, "workflow.js"))


@app.get("/ui/mlflow.js", include_in_schema=False)
def ui_mlflow_js():
    return FileResponse(os.path.join(UI_DIR, "mlflow.js"))


@app.get("/ui/styles.css", include_in_schema=False)
def ui_styles():
    return FileResponse(os.path.join(UI_DIR, "styles.css"))

@app.get("/workflow/steps")
def workflow_steps():
    return {"steps": get_workflow_steps()}


@app.get("/jobs/{job_id}/status")
def job_status(job_id: str):
    is_remote, actual_job_id = _split_remote_job_id(job_id)
    if is_remote and settings.WORKER_BASE_URL:
        response = requests.get(
            f"{settings.WORKER_BASE_URL}/jobs/{actual_job_id}/status",
            headers=_worker_headers(),
            timeout=settings.WORKER_TIMEOUT_SEC,
        )
        response.raise_for_status()
        return response.json()
    return get_status(job_id)


@app.get("/jobs/{job_id}/logs")
def job_logs(job_id: str, tail: int = 200):
    is_remote, actual_job_id = _split_remote_job_id(job_id)
    if is_remote and settings.WORKER_BASE_URL:
        response = requests.get(
            f"{settings.WORKER_BASE_URL}/jobs/{actual_job_id}/logs",
            headers=_worker_headers(),
            params={"tail": tail},
            timeout=settings.WORKER_TIMEOUT_SEC,
        )
        response.raise_for_status()
        return response.json()
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


@app.get("/documents")
async def documents():
    docs = await _collect_documents_summary()
    return {"status": "ok", "documents": docs, "count": len(docs)}


@app.get("/mlflow/summary")
def mlflow_summary():
    return {"status": "ok", **_collect_mlflow_summary()}


@app.get("/worker/health")
def worker_health(request: Request):
    _require_worker_secret(request)
    return {
        "status": "ok",
        "mode": "worker",
        "hybrid_proxy_enabled": settings.HYBRID_WORKER_ENABLED,
        "worker_base_url": settings.WORKER_BASE_URL,
    }


@app.get("/worker/status")
def worker_status():
    data: Dict[str, Any] = {
        "hybrid_worker_enabled": settings.HYBRID_WORKER_ENABLED,
        "worker_base_url": settings.WORKER_BASE_URL,
        "worker_secret_configured": bool(settings.WORKER_SHARED_SECRET),
    }
    if not settings.WORKER_BASE_URL:
        data["reachable"] = False
        data["message"] = "WORKER_BASE_URL is not configured"
        return data

    try:
        response = requests.get(
            f"{settings.WORKER_BASE_URL}/worker/health",
            headers=_worker_headers(),
            timeout=min(settings.WORKER_TIMEOUT_SEC, 20),
        )
        response.raise_for_status()
        data["reachable"] = True
        data["worker"] = response.json()
    except Exception as e:
        data["reachable"] = False
        data["error"] = str(e)
    return data


def _run_async(coro):
    return asyncio.run(coro)


def _external_pipeline_impl(job_id: str, keyword: str, amount: int, fixed_sources: list[str] | None = None):
    try:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        source_mode = "fixed_sources" if fixed_sources else "search"
        with mlflow_tracker.start_run(
            run_name=f"external-pipeline-{job_id}",
            tags={"kind": "external_pipeline", "job_id": job_id, "source_mode": source_mode},
        ):
            mlflow_tracker.log_params(
                {
                    "job_id": job_id,
                    "keyword": keyword,
                    "amount": amount,
                    "source_mode": source_mode,
                    "fixed_source_count": len(fixed_sources or []),
                    "embed_model": settings.EMBED_MODEL,
                    "llm_model": settings.OLLAMA_LLM_MODEL,
                    "blip_model": settings.BLIP_MODEL,
                }
            )

            set_stage(job_id, "web_scraping", f"Scraping keyword={keyword} amount={amount} mode={source_mode}")
            log(job_id, "info", "Starting external scrape...")
            scrape = run_external_scrape(job_id, keyword, max_links=amount, fixed_sources=fixed_sources)
            main_folder = scrape["main_folder"]
            items = scrape["items"]

            scrape_summary = _summarize_scrape(scrape)
            mlflow_tracker.log_dict(scrape_summary, "scrape/summary.json")
            mlflow_tracker.log_metrics(
                {
                    "scrape_item_count": scrape_summary["item_count"],
                    "scrape_url_count": scrape_summary["url_count"],
                    "scrape_downloaded_images": scrape_summary["downloaded_images"],
                    "scrape_downloaded_files": scrape_summary["downloaded_files"],
                    "scrape_pdf_extracted": scrape_summary["pdf_extracted"],
                }
            )

            set_stage(job_id, "data_collecting", "Saved web content/images/pdfs", {"main_folder": main_folder, "items": items})
            log(job_id, "info", "External scrape complete", {"main_folder": main_folder})

            set_stage(job_id, "captioning", "Captioning images (BLIP + Argos)...")
            caption_stats = []
            for it in items:
                site_folder = it["Folder"]
                content_path = os.path.join(site_folder, "content.txt")
                ctx = ""
                if os.path.exists(content_path):
                    with open(content_path, "r", encoding="utf-8", errors="ignore") as handle:
                        ctx = handle.read()

                images_dir = os.path.join(site_folder, "images")
                images_meta = os.path.join(images_dir, "images_meta.jsonl")
                images_under = os.path.join(site_folder, "images_understanding.jsonl")
                images_download_meta = os.path.join(images_dir, "images_download_meta.jsonl")

                res = caption_images_with_blip_and_translate(
                    images_dir=images_dir,
                    images_meta_jsonl=images_meta,
                    images_understanding_jsonl=images_under,
                    context_text=ctx,
                    blip_model=settings.BLIP_MODEL,
                    images_download_meta_jsonl=images_download_meta,
                    dedupe=True,
                )
                caption_stats.append({"site_folder": site_folder, **res})
                log(job_id, "info", "Captioned images", {"site_folder": site_folder, **res})

            if caption_stats:
                mlflow_tracker.log_dict({"items": caption_stats}, "captioning/results.json")

            set_stage(job_id, "ocr", "OCR PDFs (PyMuPDF + PaddleOCR)...")
            out_root = os.path.join(main_folder, "ocr_results")
            os.makedirs(out_root, exist_ok=True)

            def pcb(p: dict):
                if p.get("stage") in ("file_start", "page_done", "file_done", "done"):
                    log(job_id, "info", "OCR progress", p)

            process_folder_pdfs(files_dir=main_folder, out_root=out_root, progress_cb=pcb)
            ocr_summary = _summarize_ocr(out_root)
            mlflow_tracker.log_dict(ocr_summary, "ocr/summary.json")
            mlflow_tracker.log_metrics(ocr_summary)

            set_stage(job_id, "ingest_external", "Ingest to External RAG...")
            store_thread = RAGStore(
                database_url=settings.DATABASE_URL,
                embed_dims=settings.EMBED_DIMS,
            )

            async def _ingest():
                await store_thread.init_db()
                summary = await ingest_main_folder(store_thread, "external", main_folder)
                await store_thread.engine.dispose()
                return summary

            ingest_summary = run_async_in_new_loop(_ingest())
            mlflow_tracker.log_dict(ingest_summary, "ingest/summary.json")
            mlflow_tracker.log_metrics(
                {
                    "ingest_added_chunks": ingest_summary.get("added_chunks", 0),
                    "ingest_added_facts": ingest_summary.get("added_facts", 0),
                    "ingest_skipped": ingest_summary.get("skipped", 0),
                }
            )
            log(job_id, "info", "Ingest external done", ingest_summary)

            report_path = generate_report_md(
                main_folder,
                {
                    "job_id": job_id,
                    "keyword": keyword,
                    "namespace": "external",
                    "ingest": ingest_summary,
                },
            )
            mlflow_tracker.log_artifacts([report_path], artifact_path="reports")
            if os.path.exists(scrape_summary["csv_path"]):
                mlflow_tracker.log_artifacts([scrape_summary["csv_path"]], artifact_path="scrape")
            mlflow_tracker.log_text(
                _safe_json({"scrape": scrape_summary, "ocr": ocr_summary, "ingest": ingest_summary}),
                "pipeline/run_summary.json",
            )

            log(job_id, "info", "Generated report.md", {"report_path": report_path})
            set_stage(
                job_id,
                "ready",
                "External pipeline completed โ…",
                {
                    "main_folder": main_folder,
                    "ingest": ingest_summary,
                    "report_path": report_path,
                },
            )

    except Exception as e:
        tb = traceback.format_exc()
        msg = f"Pipeline failed: {type(e).__name__}: {repr(e)}"
        set_stage(job_id, "error", msg)
        log(job_id, "error", "Pipeline exception", {"error": repr(e), "type": type(e).__name__, "traceback": tb})


def _external_pipeline_thread(job_id: str, keyword: str, amount: int, fixed_sources: list[str] | None = None):
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    return _external_pipeline_impl(job_id, keyword, amount, fixed_sources)


@app.post("/pipeline/external/scrape")
def start_external_scrape(payload: Dict[str, Any] = Body(...)):
    keyword = (payload.get("keyword") or "").strip()
    amount = int(payload.get("amount") or 5)
    fixed_sources = payload.get("fixed_sources") or []
    if isinstance(fixed_sources, str):
        fixed_sources = [line.strip() for line in fixed_sources.splitlines() if line.strip()]
    elif isinstance(fixed_sources, list):
        fixed_sources = [str(x).strip() for x in fixed_sources if str(x).strip()]
    else:
        fixed_sources = []
    if not keyword:
        return {"status": "bad_request", "message": "keyword required"}
    if _should_proxy_to_worker():
        remote = _proxy_worker_json(
            "/worker/pipeline/external/scrape",
            {"keyword": keyword, "amount": amount, "fixed_sources": fixed_sources},
        )
        remote["job_id"] = _remote_job_id(str(remote.get("job_id", "")))
        remote["execution"] = "remote_worker"
        remote["worker_base_url"] = settings.WORKER_BASE_URL
        return remote

    job_id = uuid.uuid4().hex[:12]
    new_job(job_id, kind="external_pipeline")
    log(job_id, "info", "Job created")

    t = threading.Thread(target=_external_pipeline_thread, args=(job_id, keyword, amount, fixed_sources), daemon=True)
    t.start()

    return {"status": "ok", "job_id": job_id, "mode": "fixed_sources" if fixed_sources else "search", "fixed_sources": fixed_sources}


@app.post("/worker/pipeline/external/scrape")
def worker_start_external_scrape(request: Request, payload: Dict[str, Any] = Body(...)):
    _require_worker_secret(request)
    keyword = (payload.get("keyword") or "").strip()
    amount = int(payload.get("amount") or 5)
    fixed_sources = payload.get("fixed_sources") or []
    if isinstance(fixed_sources, str):
        fixed_sources = [line.strip() for line in fixed_sources.splitlines() if line.strip()]
    elif isinstance(fixed_sources, list):
        fixed_sources = [str(x).strip() for x in fixed_sources if str(x).strip()]
    else:
        fixed_sources = []
    if not keyword:
        return {"status": "bad_request", "message": "keyword required"}

    job_id = uuid.uuid4().hex[:12]
    new_job(job_id, kind="external_pipeline")
    log(job_id, "info", "Worker job created")

    t = threading.Thread(target=_external_pipeline_thread, args=(job_id, keyword, amount, fixed_sources), daemon=True)
    t.start()

    return {
        "status": "ok",
        "job_id": job_id,
        "mode": "fixed_sources" if fixed_sources else "search",
        "fixed_sources": fixed_sources,
        "execution": "worker_local",
    }


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
    if _should_proxy_to_worker():
        file_bytes = await file.read()
        async with httpx.AsyncClient(timeout=settings.WORKER_TIMEOUT_SEC) as client:
            response = await client.post(
                f"{settings.WORKER_BASE_URL}/worker/pipeline/internal/upload_pdf",
                headers=_worker_headers(),
                params={"entity_hint": entity_hint},
                files={"file": (file.filename, file_bytes, file.content_type or "application/pdf")},
            )
            response.raise_for_status()
            remote = response.json()
            remote["job_id"] = _remote_job_id(str(remote.get("job_id", "")))
            remote["execution"] = "remote_worker"
            remote["worker_base_url"] = settings.WORKER_BASE_URL
            return remote

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


@app.post("/worker/pipeline/internal/upload_pdf")
async def worker_upload_pdf(request: Request, file: UploadFile = File(...), entity_hint: str = "internal_doc"):
    _require_worker_secret(request)
    original_hybrid = settings.HYBRID_WORKER_ENABLED
    settings.HYBRID_WORKER_ENABLED = False
    try:
        return await upload_pdf(file=file, entity_hint=entity_hint)
    finally:
        settings.HYBRID_WORKER_ENABLED = original_hybrid


@app.post("/ask")
async def ask(payload: Dict[str, Any] = Body(...)):
    q = (payload.get("question") or "").strip()
    if not q:
        return {"status": "bad_request", "message": "question required"}
    top_k = int(payload.get("top_k") or settings.RAG_TOP_K)
    try:
        out = await agent.answer(q, top_k=top_k)
    except Exception as e:
        traceback.print_exc()
        return {
            "status": "error",
            "message": "ask pipeline failed",
            "error": str(e),
        }
    warnings = []
    for meta in out.get("chunks") or []:
        warning = (meta or {}).get("warning")
        if warning and warning not in warnings:
            warnings.append(warning)
    if warnings:
        out["warnings"] = warnings
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
    results = await run_eval(store, dataset_path, namespace, k=k, agent=agent)
    return {"status": "ok", "namespace": namespace, "k": k, "results": results}

def run_async_in_new_loop(coro):
    """Run async coroutine in a fresh event loop (safe for threads)."""
    return asyncio.run(coro)



if __name__ == "__main__":
    import uvicorn
    print(f"Root:       http://localhost:{settings.API_PORT}/")
    print(f"UI_FORUSER: http://localhost:{settings.API_PORT}/UI_FORUSER")
    print(f"UI:         http://localhost:{settings.API_PORT}/ui")
    print(f"Flow UI:    http://localhost:{settings.API_PORT}/ui/workflow")
    print(f"Swagger:    http://localhost:{settings.API_PORT}/docs")
    uvicorn.run("backend.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
