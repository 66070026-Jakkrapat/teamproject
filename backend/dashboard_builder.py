"""
dashboard_builder.py

Fallback dashboard builder (rule-based) เผื่อ LLM output schema ไม่ตรง
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

def build_dashboard_json(question: str, context: str) -> Dict[str, Any]:
    # พยายามดึงตัวเลขพื้นฐานแบบง่าย (MVP fallback)
    # สำหรับ 100% ใช้ LLM จะดีกว่า แต่อันนี้กันพัง
    nums = re.findall(r"(\d[\d,]*(?:\.\d+)?)", context or "")
    highlights = []
    if "รายได้" in context:
        highlights.append("พบคำว่า 'รายได้' ในเอกสาร")
    if "กำไร" in context:
        highlights.append("พบคำว่า 'กำไร' ในเอกสาร")
    if not highlights:
        highlights.append("สรุปจากบริบทที่ดึงมาได้")

    return {
        "title": f"Dashboard: {question[:60]}",
        "highlights": highlights[:8],
        "metrics": [],
        "tables": [
            {
                "name": "Extracted Numbers (sample)",
                "columns": ["value"],
                "rows": [[n] for n in nums[:30]],
            }
        ],
    }
