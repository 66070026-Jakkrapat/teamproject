# backend/utils/job_manager.py
from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

WORKFLOW_STEPS = [
    {"key": "idle", "label": "Idle"},
    {"key": "web_scraping", "label": "Web Scraping"},
    {"key": "data_collecting", "label": "Data Collecting"},
    {"key": "captioning", "label": "Image Captioning"},
    {"key": "ocr", "label": "OCR Processing"},
    {"key": "ingest_external", "label": "Ingest External RAG"},
    {"key": "ingest_internal", "label": "Ingest Internal RAG"},
    {"key": "ready", "label": "Ready"},
    {"key": "error", "label": "Error"},
]

_LOCK = threading.Lock()
_STATUS: Dict[str, Dict[str, Any]] = {}
_LOGS: Dict[str, List[Dict[str, Any]]] = {}

def new_job(job_id: str, kind: str) -> None:
    with _LOCK:
        _STATUS[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "stage": "idle",
            "message": "",
            "updated_at": time.time(),
            "result": {},
        }
        _LOGS[job_id] = []

def set_stage(job_id: str, stage: str, message: str = "", extra: Optional[dict] = None) -> None:
    with _LOCK:
        st = _STATUS.setdefault(job_id, {"job_id": job_id})
        st["stage"] = stage
        st["message"] = message
        st["updated_at"] = time.time()
        if extra:
            st.update(extra)

def log(job_id: str, level: str, message: str, data: Optional[dict] = None) -> None:
    with _LOCK:
        _LOGS.setdefault(job_id, [])
        _LOGS[job_id].append({
            "ts": time.time(),
            "level": level,
            "message": message,
            "data": data or {},
        })
        _LOGS[job_id] = _LOGS[job_id][-5000:]  # cap

def get_status(job_id: str) -> Dict[str, Any]:
    with _LOCK:
        return dict(_STATUS.get(job_id, {"job_id": job_id, "stage": "not_found"}))

def get_logs(job_id: str, tail: int = 200) -> List[Dict[str, Any]]:
    with _LOCK:
        return list(_LOGS.get(job_id, []))[-tail:]

def get_workflow_steps() -> List[Dict[str, Any]]:
    return WORKFLOW_STEPS[:]
