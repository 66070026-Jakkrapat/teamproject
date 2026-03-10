"""Typhoon OCR service for document layout analysis and table extraction."""

import base64
import httpx
import json
from typing import Dict, List, Any
from backend.config import get_settings

settings = get_settings()


class TyphoonOCRService:
    """Calls Typhoon OCR (Vision API) to extract text and tables from documents."""

    def __init__(self):
        self.api_key = settings.TYPHOON_API_KEY
        self.endpoint = settings.TYPHOON_OCR_ENDPOINT
        self.model = settings.TYPHOON_OCR_MODEL

    async def extract_from_image(self, image_bytes: bytes, mime_type: str = "image/png") -> Dict[str, Any]:
        """Extract text and tables from a single image."""
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a document OCR specialist. Extract ALL text and tables from the image. "
                        "Return the result as JSON with two keys:\n"
                        '- "text_blocks": list of strings (each paragraph/section as a separate entry)\n'
                        '- "tables": list of objects, each with "headers" (list of strings) and '
                        '"rows" (list of lists of strings)\n'
                        "Preserve the original Thai language text exactly. "
                        "Do not translate or summarize."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Extract all text and tables from this document image. Return as JSON."
                        }
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(self.endpoint, json=payload, headers=headers)
            response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Try to parse JSON from the response
        return self._parse_ocr_response(content)

    async def extract_from_pdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extract from PDF by converting pages to images first (via Typhoon)."""
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a document OCR specialist. Extract ALL text and tables from this PDF. "
                        "Return the result as JSON with two keys:\n"
                        '- "text_blocks": list of strings (each paragraph/section)\n'
                        '- "tables": list of objects with "headers" and "rows"\n'
                        "Preserve Thai language text exactly."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:application/pdf;base64,{b64_pdf}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Extract all text and tables from this PDF document. Return as JSON."
                        }
                    ]
                }
            ],
            "max_tokens": 8192,
            "temperature": 0.1,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(self.endpoint, json=payload, headers=headers)
            response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return self._parse_ocr_response(content)

    def _parse_ocr_response(self, content: str) -> Dict[str, Any]:
        """Parse the OCR response, handling both JSON and plain text."""
        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        if "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Fallback: return raw text as a single block
        return {
            "text_blocks": [content],
            "tables": []
        }


# Singleton
ocr_service = TyphoonOCRService()
