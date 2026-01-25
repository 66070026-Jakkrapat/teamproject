"""
rag_store.py

งานหลัก:
- สร้าง/โหลด ChromaDB
- ทำ embeddings ผ่าน Ollama (/api/embeddings) โดยใช้ nomic-embed-text
- Ingest:
  - Web: content.txt, images_meta.jsonl, images_understanding.jsonl
  - OCR: ocr_results/**/docs.jsonl, ocr_results/**/images.jsonl
  - Internal upload: PDF -> OCR -> docs.jsonl แล้ว ingest
- Retrieval: query -> top-k chunks พร้อม metadata สำหรับ citation

ออกแบบให้ "ครบ" และ "ทนทาน" (ไม่พังง่าย):
- chunking แบบยืดหยุ่น
- dedupe ด้วย sha1(text+source)
"""

from __future__ import annotations

import os
import re
import json
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
import requests


# ----------------------------
# Helpers
# ----------------------------

def sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()

def safe_read_text(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    for enc in ("utf-8", "utf-8-sig", "cp874", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            pass
    return ""

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

def normalize_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"[^\S\r\n]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    """
    chunk แบบง่ายแต่ใช้งานจริงได้:
    - ตัดด้วยย่อหน้า/บรรทัดก่อน
    - รวมจนถึง chunk_size
    - overlap เพื่อคงบริบท
    """
    text = normalize_text(text)
    if not text:
        return []

    paras = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    buf = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p

    if buf:
        chunks.append(buf)

    # add overlap (rough)
    if overlap > 0 and len(chunks) > 1:
        out: List[str] = []
        prev = ""
        for c in chunks:
            if prev:
                out.append((prev[-overlap:] + "\n" + c).strip())
            else:
                out.append(c)
            prev = c
        return out

    return chunks


# ----------------------------
# Ollama Embedding Client
# ----------------------------

class OllamaEmbedder:
    def __init__(self, host: str, model: str, timeout_s: int = 60):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Ollama embeddings endpoint: POST /api/embeddings
        payload: {"model": "...", "prompt": "text"}
        """
        out: List[List[float]] = []
        for t in texts:
            payload = {"model": self.model, "prompt": t}
            r = requests.post(f"{self.host}/api/embeddings", json=payload, timeout=self.timeout_s)
            if r.status_code != 200:
                raise RuntimeError(f"Ollama embeddings failed: {r.status_code} {r.text[:200]}")
            data = r.json()
            emb = data.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError("Ollama embeddings invalid response")
            out.append([float(x) for x in emb])
        return out


# ----------------------------
# RAG Store
# ----------------------------

@dataclass
class RetrievedChunk:
    text: str
    score: float
    meta: Dict[str, Any]

class RAGStore:
    def __init__(
        self,
        chroma_dir: str,
        collection_name: str,
        ollama_host: str,
        embed_model: str,
    ):
        os.makedirs(chroma_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.col = self.client.get_or_create_collection(name=collection_name)
        self.embedder = OllamaEmbedder(host=ollama_host, model=embed_model)

    def upsert_texts(self, texts: List[str], metas: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> int:
        assert len(texts) == len(metas)
        if not texts:
            return 0

        # dedupe by id = sha1(text+source_key)
        gen_ids: List[str] = []
        for i, (t, m) in enumerate(zip(texts, metas)):
            source_key = f"{m.get('source_type','')}|{m.get('source_url','')}|{m.get('source_path','')}|{m.get('page','')}"
            gen_ids.append(sha1_text(source_key + "|" + t))

        if ids is None:
            ids = gen_ids

        # embeddings
        embs = self.embedder.embed(texts)

        self.col.upsert(
            ids=ids,
            documents=texts,
            metadatas=metas,
            embeddings=embs,
        )
        return len(texts)

    def query(self, question: str, top_k: int = 8) -> List[RetrievedChunk]:
        q_emb = self.embedder.embed([question])[0]
        res = self.col.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out: List[RetrievedChunk] = []
        for t, m, d in zip(docs, metas, dists):
            # chroma distance smaller is closer; convert to a similarity-ish score
            score = float(1.0 / (1.0 + float(d))) if d is not None else 0.0
            out.append(RetrievedChunk(text=t or "", score=score, meta=m or {}))
        return out


# ----------------------------
# Ingestion (External folder)
# ----------------------------

def ingest_scrape_folder(store: RAGStore, main_folder: str) -> Dict[str, Any]:
    """
    ingest โฟลเดอร์งาน scraped_outputs/data_xxx/
    อ่าน:
    - site_folder/content.txt
    - site_folder/images/images_meta.jsonl (caption+relevance baseline)
    - site_folder/images_understanding.jsonl (structured vision)
    - main_folder/ocr_results/**/docs.jsonl
    - main_folder/ocr_results/**/images.jsonl
    """
    added = 0
    skipped = 0

    def add_doc(text: str, meta: Dict[str, Any]) -> None:
        nonlocal added, skipped
        text = normalize_text(text)
        if len(text) < 30:
            skipped += 1
            return
        chunks = chunk_text(text)
        if not chunks:
            skipped += 1
            return
        metas = []
        for idx, c in enumerate(chunks, start=1):
            m = dict(meta)
            m["chunk_index"] = idx
            m["chunk_total"] = len(chunks)
            metas.append(m)
        added += store.upsert_texts(chunks, metas)

    # 1) Sites
    for name in sorted(os.listdir(main_folder)):
        site_folder = os.path.join(main_folder, name)
        if not os.path.isdir(site_folder) or not (name[:1].isdigit() and "_" in name):
            continue

        content_path = os.path.join(site_folder, "content.txt")
        content = safe_read_text(content_path)
        if content:
            add_doc(content, {
                "source_type": "web_page",
                "source_path": content_path,
                "site_folder": site_folder,
            })

        # baseline image meta
        meta_path = os.path.join(site_folder, "images", "images_meta.jsonl")
        for obj in iter_jsonl(meta_path):
            cap = (obj.get("caption") or "").strip()
            src_url = (obj.get("source_url") or "").strip()
            score = float(obj.get("relevance_score") or 0.0)
            if cap:
                add_doc(cap, {
                    "source_type": "web_image_caption",
                    "source_url": src_url,
                    "source_path": obj.get("saved_path") or "",
                    "relevance_score": score,
                    "site_folder": site_folder,
                })

        # structured image understanding
        structured_path = os.path.join(site_folder, "images_understanding.jsonl")
        for obj in iter_jsonl(structured_path):
            # store combined string for retrieval
            cap_th = (obj.get("caption_th") or "").strip()
            cap_en = (obj.get("caption_en") or "").strip()
            labels = obj.get("labels") or []
            image_type = (obj.get("image_type") or "other").strip()
            keep = bool(obj.get("keep_for_rag"))
            rel = float(obj.get("relevance_score") or 0.0)
            text = "\n".join([
                f"caption_th: {cap_th}",
                f"caption_en: {cap_en}",
                f"image_type: {image_type}",
                f"labels: {', '.join(labels) if isinstance(labels, list) else ''}",
                f"keep_for_rag: {keep}",
                f"relevance_score: {rel}",
            ]).strip()
            if (cap_th or cap_en) and keep:
                add_doc(text, {
                    "source_type": "web_image_understanding",
                    "source_path": obj.get("image_path") or "",
                    "site_folder": site_folder,
                    "relevance_score": rel,
                    "image_type": image_type,
                })

    # 2) OCR outputs
    ocr_root = os.path.join(main_folder, "ocr_results")
    if os.path.isdir(ocr_root):
        for pdf_name in sorted(os.listdir(ocr_root)):
            pdf_dir = os.path.join(ocr_root, pdf_name)
            if not os.path.isdir(pdf_dir):
                continue

            docs_jsonl = os.path.join(pdf_dir, "docs.jsonl")
            for obj in iter_jsonl(docs_jsonl):
                if obj.get("type") == "page_text":
                    add_doc(obj.get("text") or "", {
                        "source_type": "pdf_page",
                        "source_path": obj.get("source_pdf") or "",
                        "pdf_sha1": obj.get("pdf_sha1") or "",
                        "page": obj.get("page") or 0,
                        "extract_method": obj.get("extract_method") or "",
                    })
                elif obj.get("type") == "figure":
                    # OCR of embedded images in PDF
                    img_ocr = obj.get("image_ocr") or ""
                    if img_ocr and len(img_ocr) >= 10:
                        add_doc(img_ocr, {
                            "source_type": "pdf_figure_ocr",
                            "source_path": obj.get("image_path") or "",
                            "pdf_sha1": obj.get("pdf_sha1") or "",
                            "page": obj.get("page") or 0,
                        })

            img_jsonl = os.path.join(pdf_dir, "images.jsonl")  # (ถ้ามี) รูปจากเว็บ OCR
            for obj in iter_jsonl(img_jsonl):
                if obj.get("type") == "web_image":
                    add_doc(obj.get("text") or "", {
                        "source_type": "web_image_ocr",
                        "source_path": obj.get("image_path") or "",
                    })

            # structured pdf image understanding (optional)
            pdf_under = os.path.join(pdf_dir, "pdf_images_understanding.jsonl")
            for obj in iter_jsonl(pdf_under):
                cap_th = (obj.get("caption_th") or "").strip()
                cap_en = (obj.get("caption_en") or "").strip()
                keep = bool(obj.get("keep_for_rag"))
                rel = float(obj.get("relevance_score") or 0.0)
                if keep and (cap_th or cap_en):
                    add_doc(f"{cap_th}\n{cap_en}", {
                        "source_type": "pdf_image_understanding",
                        "source_path": obj.get("image_path") or "",
                        "pdf_sha1": obj.get("sha1") or "",
                        "relevance_score": rel,
                    })

    return {"added_chunks": added, "skipped": skipped, "folder": main_folder}
