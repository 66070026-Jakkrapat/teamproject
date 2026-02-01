# backend/report_utils/__init__.py

from __future__ import annotations

import os
import json
from typing import Any, Dict, List

def _count_jsonl(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def _list_files(dir_path: str, exts: tuple = ()) -> List[str]:
    if not dir_path or not os.path.isdir(dir_path):
        return []
    out = []
    for fn in os.listdir(dir_path):
        p = os.path.join(dir_path, fn)
        if os.path.isfile(p):
            if not exts or fn.lower().endswith(exts):
                out.append(p)
    return sorted(out)

def generate_report_md(main_folder: str, scrape_result: Dict[str, Any]) -> str:
    """
    สร้าง report.md เพื่อ audit โฟลเดอร์ scrape ว่ามีอะไรบ้าง และช่วย debug pipeline
    """
    report_path = os.path.join(main_folder, "report.md")

    lines: List[str] = []
    lines.append("# Scrape Audit Report\n\n")
    lines.append(f"- Main folder: `{main_folder}`\n")
    if scrape_result.get("file_path"):
        lines.append(f"- CSV: `{scrape_result['file_path']}`\n")
    if scrape_result.get("ocr_job_id"):
        lines.append(f"- OCR Job: `{scrape_result['ocr_job_id']}`\n")
    if scrape_result.get("ocr_status_url"):
        lines.append(f"- OCR Status URL: `http://localhost:8000{scrape_result['ocr_status_url']}`\n")

    lines.append("\n---\n\n## Sites\n\n")
    lines.append("| # | Source | URL | images | images_meta | images_understanding | files | site_folder |\n")
    lines.append("|---|--------|-----|--------|------------|---------------------|-------|------------|\n")

    collected = scrape_result.get("data") or []
    for i, row in enumerate(collected, start=1):
        folder = row.get("Folder") or ""
        img_dir = os.path.join(folder, "images")
        file_dir = os.path.join(folder, "files")

        imgs = _list_files(img_dir, (".jpg", ".jpeg", ".png", ".webp"))
        files = _list_files(file_dir, (".pdf", ".xlsx", ".xls", ".docx", ".pptx", ".zip", ".bin"))

        meta_jsonl = os.path.join(img_dir, "images_meta.jsonl")
        under_jsonl = os.path.join(folder, "images_understanding.jsonl")

        meta_lines = _count_jsonl(meta_jsonl)
        under_lines = _count_jsonl(under_jsonl)

        url = row.get("URL") or ""
        url_md = f"[link]({url})" if url.startswith("http") else url

        lines.append(
            f"| {i} | {row.get('Source','')} | {url_md} | {len(imgs)} | {meta_lines} | {under_lines} | {len(files)} | `{folder}` |\n"
        )

    ocr_root = os.path.join(main_folder, "ocr_results")
    lines.append("\n---\n\n## OCR Outputs\n\n")
    lines.append(f"- ocr_results: `{ocr_root}`\n")
    if os.path.isdir(ocr_root):
        pdf_dirs = [d for d in sorted(os.listdir(ocr_root)) if os.path.isdir(os.path.join(ocr_root, d))]
        lines.append(f"- pdf folders: {len(pdf_dirs)}\n")
        for d in pdf_dirs[:80]:
            p = os.path.join(ocr_root, d)
            docs_jsonl = os.path.join(p, "docs.jsonl")
            img_jsonl = os.path.join(p, "images.jsonl")
            lines.append(
                f"  - `{d}` docs.jsonl={os.path.exists(docs_jsonl)} images.jsonl={os.path.exists(img_jsonl)}\n"
            )
    else:
        lines.append("- (ยังไม่พบ ocr_results)\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return report_path
