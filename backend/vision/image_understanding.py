# backend/vision/image_understanding.py
from __future__ import annotations

import os
import io
import time
import json
import hashlib
import traceback
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.utils.text import normalize_text
from backend.utils.jsonl import write_jsonl, iter_jsonl

# ----------------------------
# Optional deps flags
# ----------------------------
_BLIP_OK = False
_ARGOS_OK = False

# ----------------------------
# Try import BLIP (robust way)
# ----------------------------
try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    _BLIP_OK = True
except Exception:
    _BLIP_OK = False

# ----------------------------
# Try import Argos
# ----------------------------
try:
    import argostranslate.translate
    import argostranslate.package
    _ARGOS_OK = True
except Exception:
    _ARGOS_OK = False

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".webp")


# ----------------------------
# Utils
# ----------------------------
def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def list_images(images_dir: str) -> List[str]:
    out: List[str] = []
    if not images_dir or not os.path.isdir(images_dir):
        return out
    for root, _, files in os.walk(images_dir):
        for fn in files:
            if fn.lower().endswith(SUPPORTED_EXTS):
                out.append(os.path.join(root, fn))
    return sorted(out)


def warmup_blip(model_name: str) -> None:
    """
    โหลด BLIP ไว้ล่วงหน้า (ลดหน่วงรอบแรก)
    """
    if not _BLIP_OK:
        print("[BLIP] Warmup skipped (_BLIP_OK=False)")
        return
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[BLIP] Warming up {model_name} on {device} ...")
        _ = BlipProcessor.from_pretrained(model_name)
        m = BlipForConditionalGeneration.from_pretrained(model_name)
        _ = m.to(device)
        print("[BLIP] Warmup success.")
    except Exception as e:
        print(f"[BLIP Error] Warmup failed: {e}")
        traceback.print_exc()


def compute_relevance_score(context_text: str, caption: str) -> float:
    if not context_text or not caption:
        return 0.0
    t1 = normalize_text(context_text)[:7000]
    t2 = normalize_text(caption)[:600]
    try:
        vec = TfidfVectorizer(stop_words=None)
        X = vec.fit_transform([t1, t2])
        score = float(cosine_similarity(X[0], X[1])[0][0])
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def guess_image_type(caption_en: str) -> str:
    c = (caption_en or "").lower()
    if any(k in c for k in ["chart", "graph", "plot", "bar chart", "line chart", "pie chart"]):
        return "chart"
    if any(k in c for k in ["table", "spreadsheet"]):
        return "table"
    if any(k in c for k in ["document", "paper", "report", "invoice", "slide", "presentation"]):
        return "document"
    if "logo" in c:
        return "logo"
    if any(k in c for k in ["person", "people", "man", "woman", "portrait", "face"]):
        return "people"
    if "infographic" in c:
        return "infographic"
    return "other"


# ----------------------------
# Argos helpers
# ----------------------------
def ensure_argos_langpair(from_code: str = "en", to_code: str = "th") -> bool:
    """
    Best-effort: ถ้าไม่มี package en->th จะพยายามติดตั้งให้
    (ต้องมีอินเทอร์เน็ตในครั้งแรก)
    """
    if not _ARGOS_OK:
        return False

    try:
        installed = argostranslate.package.get_installed_packages()
        if any(p.from_code == from_code and p.to_code == to_code for p in installed):
            return True

        print(f"[Argos] Language package {from_code}->{to_code} not installed. Trying to install (best-effort)...")
        try:
            argostranslate.package.update_package_index()
            available = argostranslate.package.get_available_packages()
            target = None
            for p in available:
                if p.from_code == from_code and p.to_code == to_code:
                    target = p
                    break
            if not target:
                print("[Argos] No available package found in index.")
                return False
            download_path = target.download()
            argostranslate.package.install_from_path(download_path)
            print("[Argos] Installed language package successfully.")
            return True
        except Exception as e:
            print(f"[Argos] Auto-install failed: {e}")
            print("[Argos Hint] If offline, run once with internet: `argospm install en th`")
            return False
    except Exception:
        return False


def translate_en_to_th(text: str, from_code: str = "en", to_code: str = "th") -> str:
    if not _ARGOS_OK:
        return ""
    if not text:
        return ""

    # ensure package exists (best-effort)
    _ = ensure_argos_langpair(from_code, to_code)

    try:
        out = argostranslate.translate.translate(text, from_code, to_code)
        return (out or "").strip()
    except Exception as e:
        print(f"[Argos Translation Error] {e}")
        return ""


# ----------------------------
# Download meta mapper
# ----------------------------
def _load_download_meta_map(images_download_meta_jsonl: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    คืน map โดย key=sha1 (ถ้ามี) และ key=saved_path (fallback)
    """
    mp: Dict[str, Dict[str, Any]] = {}
    if not images_download_meta_jsonl or not os.path.exists(images_download_meta_jsonl):
        return mp

    for obj in iter_jsonl(images_download_meta_jsonl) or []:
        if not isinstance(obj, dict):
            continue
        sha1 = (obj.get("sha1") or "").strip()
        sp = (obj.get("saved_path") or "").strip()
        if sha1:
            mp[f"sha1:{sha1}"] = obj
        if sp:
            mp[f"path:{os.path.abspath(sp)}"] = obj
    return mp


def _fallback_caption_from_meta(base: Dict[str, Any]) -> str:
    """
    ถ้า BLIP caption ว่าง ให้ดึงข้อความจาก meta ที่ scrape มา:
    - alt_text / title_text / nearby_text
    """
    if not base:
        return ""
    fallback = " ".join([
        (base.get("alt_text") or "").strip(),
        (base.get("title_text") or "").strip(),
        (base.get("nearby_text") or "").strip(),
    ]).strip()
    return normalize_text(fallback)[:300]


# ----------------------------
# BLIP captioner (robust)
# ----------------------------
class BLIPCaptioner:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if (_BLIP_OK and torch.cuda.is_available()) else "cpu"
        self._processor = None
        self._model = None

    def _lazy_init(self) -> None:
        if self._processor is not None and self._model is not None:
            return
        if not _BLIP_OK:
            return

        try:
            print(f"[BLIP] Loading processor/model: {self.model_name} on {self.device}")
            self._processor = BlipProcessor.from_pretrained(self.model_name)
            self._model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
            print("[BLIP] Loaded successfully.")
        except Exception as e:
            print(f"[BLIP Error] Failed to load BLIP model: {e}")
            traceback.print_exc()
            self._processor = None
            self._model = None

    def caption_en(self, image: Image.Image) -> str:
        if not _BLIP_OK:
            return ""
        self._lazy_init()
        if self._processor is None or self._model is None:
            return ""

        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            # ป้องกันภาพใหญ่มากจนโมเดล fail/ช้าเกิน
            if max(image.size) > 1600:
                image = image.copy()
                image.thumbnail((1600, 1600))

            inputs = self._processor(images=image, return_tensors="pt")

            # บาง key ไม่ใช่ tensor -> อย่าฝืน .to()
            for k in list(inputs.keys()):
                try:
                    inputs[k] = inputs[k].to(self.device)
                except Exception:
                    pass

            with torch.no_grad():
                out_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=30,
                    num_beams=3,
                )

            cap = self._processor.decode(out_ids[0], skip_special_tokens=True)
            return normalize_text(cap)
        except Exception as e:
            print(f"[BLIP Inference Error] {e}")
            return ""


# ----------------------------
# Main function
# ----------------------------
def caption_images_with_blip_and_translate(
    images_dir: str,
    images_meta_jsonl: str,
    images_understanding_jsonl: str,
    context_text: str,
    blip_model: str,
    images_download_meta_jsonl: Optional[str] = None,
    dedupe: bool = True
) -> Dict[str, Any]:
    """
    สร้าง caption_en ด้วย BLIP และ caption_th ด้วย Argos
    - กัน duplicate ด้วย sha1
    - ✅ ถ้า duplicate: copy caption จากต้นฉบับแทนการปล่อยว่าง
    - ✅ ถ้า BLIP ได้ caption ว่าง: fallback จาก alt/title/nearby_text
    - เขียน 2 ไฟล์:
      1) images_meta.jsonl (ไว้ดู/อ้างอิง)
      2) images_understanding.jsonl (ไว้ ingest เข้า RAG)
    """

    context_text = context_text or ""

    print(f"[Captioning] Starting for folder: {images_dir} using model: {blip_model}")
    captioner = BLIPCaptioner(model_name=blip_model)
    imgs = list_images(images_dir)
    print(f"[Captioning] Found {len(imgs)} images to process.")

    download_meta_map = _load_download_meta_map(images_download_meta_jsonl)

    # maps for dedupe copy
    first_path_by_sha1: Dict[str, str] = {}
    caption_cache_by_sha1: Dict[str, Tuple[str, str, str, float, bool, str, str]] = {}
    # (cap_en, cap_th, image_type, score, keep, caption_status, caption_error)

    meta_rows: List[Dict[str, Any]] = []
    under_rows: List[Dict[str, Any]] = []

    processed = 0
    captioned = 0
    duplicates_skipped = 0

    for i, ipath in enumerate(imgs):
        apath = os.path.abspath(ipath)
        try:
            b = open(ipath, "rb").read()
        except Exception as e:
            print(f"[Captioning Error] Could not read file {ipath}: {e}")
            continue

        h = sha1_bytes(b)
        is_dup = bool(dedupe and (h in first_path_by_sha1))
        dup_of_sha1 = h if is_dup else ""

        base = download_meta_map.get(f"sha1:{h}") or download_meta_map.get(f"path:{apath}") or {}

        if is_dup:
            duplicates_skipped += 1
            cap_en, cap_th, image_type, score, keep, caption_status, caption_error = caption_cache_by_sha1.get(
                h, ("", "", "other", 0.0, False, "fail", "duplicate_without_cache")
            )
        else:
            print(f"[Captioning] Processing image {i+1}/{len(imgs)}: {ipath}")
            first_path_by_sha1[h] = ipath

            try:
                im = Image.open(io.BytesIO(b)).convert("RGB")
            except Exception as e:
                print(f"[Captioning Error] Invalid image data {ipath}: {e}")
                continue

            # 1) BLIP caption
            cap_en = (captioner.caption_en(im) or "").strip()

            # 2) Fallback ถ้า BLIP ว่าง: ใช้ alt/title/nearby_text
            used_fallback = False
            if not cap_en:
                cap_en = _fallback_caption_from_meta(base)
                used_fallback = bool(cap_en)

            # 3) Translate TH (เฉพาะเมื่อมี cap_en)
            cap_th = translate_en_to_th(cap_en, "en", "th") if cap_en else ""

            # 4) status/error
            if cap_en:
                caption_status = "ok"
                caption_error = "used_meta_fallback" if used_fallback else ""
                print(f"  -> EN: {cap_en}")
                if cap_th:
                    print(f"  -> TH: {cap_th}")
            else:
                caption_status = "fail"
                caption_error = "empty_caption_after_fallback"
                print("  -> Failed to generate caption (BLIP empty + fallback empty).")

            image_type = guess_image_type(cap_en)
            score = compute_relevance_score(context_text, cap_en or cap_th)
            keep = (score >= 0.12) or (image_type in ("chart", "table", "document", "infographic"))

            # cache ไว้ให้ duplicate reuse
            caption_cache_by_sha1[h] = (cap_en, cap_th, image_type, float(score), bool(keep), caption_status, caption_error)

        if cap_en:
            captioned += 1

        row = {
            "type": "web_image",
            "page_url": (base.get("page_url") or base.get("page") or "").strip(),
            "page_title": (base.get("page_title") or "").strip(),
            "source_url": (base.get("source_url") or base.get("img_url") or "").strip(),
            "saved_path": ipath,
            "sha1": h,
            "width": int(base.get("width") or 0),
            "height": int(base.get("height") or 0),
            "alt_text": (base.get("alt_text") or base.get("alt") or "").strip(),
            "title_text": (base.get("title_text") or base.get("title") or "").strip(),
            "nearby_text": (base.get("nearby_text") or "").strip(),

            "caption_en": cap_en,
            "caption_th": cap_th,
            "image_type": image_type,
            "relevance_score": float(score),
            "keep_for_rag": bool(keep),

            # ✅ debug fields (ช่วยหาสาเหตุ caption ว่าง)
            "caption_status": caption_status,
            "caption_error": caption_error,

            "is_duplicate": bool(is_dup),
            "duplicate_of_sha1": dup_of_sha1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        meta_rows.append(row)

        under_rows.append({
            "type": "image_understanding",
            "image_path": ipath,
            "sha1": h,
            "caption_en": cap_en,
            "caption_th": cap_th,
            "image_type": image_type,
            "relevance_score": float(score),
            "keep_for_rag": bool(keep),

            "caption_status": caption_status,
            "caption_error": caption_error,

            "is_duplicate": bool(is_dup),
            "duplicate_of_sha1": dup_of_sha1,
            "model": blip_model if _BLIP_OK else "blip_not_available",
            "translator": "argos" if _ARGOS_OK else "argos_not_available",
            "page_url": (base.get("page_url") or base.get("page") or "").strip(),
            "source_url": (base.get("source_url") or base.get("img_url") or "").strip(),
        })

        processed += 1

    # เขียนไฟล์ผลลัพธ์
    write_jsonl(images_meta_jsonl, meta_rows)
    write_jsonl(images_understanding_jsonl, under_rows)

    print(f"[Captioning] Finished. Processed: {processed}, Captioned: {captioned}, Duplicates_skipped: {duplicates_skipped}")

    # ✅ คืน key ให้ตรงกับ log ที่คุณเห็น
    return {
        "images_total": len(imgs),
        "processed": processed,
        "captioned": captioned,
        "duplicates_skipped": duplicates_skipped,
        "blip_ok": _BLIP_OK,
        "argos_ok": _ARGOS_OK,
        "images_meta_jsonl": images_meta_jsonl,
        "images_understanding_jsonl": images_understanding_jsonl,
    }
