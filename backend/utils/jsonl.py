# backend/utils/jsonl.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable

def safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    safe_mkdir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    safe_mkdir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def iter_jsonl(path: str):
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue
