# backend/report_utils/report.py
from __future__ import annotations

import os
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

def generate_report_md(main_folder: str, summary: Dict[str, Any]) -> str:
    report_path = os.path.join(main_folder, "report.md")
    lines: List[str] = []
    lines.append("# Pipeline Report\n\n")
    lines.append(f"- main_folder: `{main_folder}`\n")
    for k, v in summary.items():
        lines.append(f"- {k}: `{v}`\n")

    lines.append("\n## Folders\n")
    for name in sorted(os.listdir(main_folder)):
        p = os.path.join(main_folder, name)
        if os.path.isdir(p):
            lines.append(f"- `{name}/`\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return report_path
