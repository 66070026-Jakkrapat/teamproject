# backend/rag/rag_store.py
from __future__ import annotations

import re
import hashlib
import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy import String, Integer, Float, Text, select, insert, delete, Index
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, mapped_column, Mapped, sessionmaker
from pgvector.sqlalchemy import Vector
from sqlalchemy import text as sql_text

from backend.settings import settings

Base = declarative_base()


def sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


# ============================================================
# Embedder
# ============================================================
class OllamaEmbedder:
    def __init__(self, host: str, model: str, timeout_s: int = 60, max_concurrency: int = 4):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_concurrency = max(1, int(max_concurrency))

    async def embed_one(self, text: str) -> List[float]:
        payload = {"model": self.model, "prompt": text}
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_s)) as client:
            r = await client.post(f"{self.host}/api/embeddings", json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama embeddings failed: {r.status_code} {r.text[:200]}")
        data = r.json()
        emb = data.get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError("Ollama embeddings invalid response")
        return [float(x) for x in emb]

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _one(t: str) -> List[float]:
            async with sem:
                return await self.embed_one(t)

        return await asyncio.gather(*[_one(t) for t in texts])


# ============================================================
# DB Models
# ============================================================
class RAGChunk(Base):
    __tablename__ = "rag_chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(32), index=True)
    source_type: Mapped[str] = mapped_column(String(64), index=True)
    source_url: Mapped[str] = mapped_column(Text, default="")
    source_path: Mapped[str] = mapped_column(Text, default="")
    page: Mapped[int] = mapped_column(Integer, default=0)
    chunk_index: Mapped[int] = mapped_column(Integer, default=1)
    chunk_total: Mapped[int] = mapped_column(Integer, default=1)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Any] = mapped_column(Vector(int(settings.EMBED_DIMS)))
    score_hint: Mapped[float] = mapped_column(Float, default=0.0)


Index("ix_rag_chunks_namespace_source", RAGChunk.namespace, RAGChunk.source_type)


class RAGFact(Base):
    __tablename__ = "rag_facts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(32), index=True)
    entity: Mapped[str] = mapped_column(String(256), index=True)
    key: Mapped[str] = mapped_column(String(128), index=True)
    value: Mapped[str] = mapped_column(String(512))
    unit: Mapped[str] = mapped_column(String(64), default="")
    year: Mapped[int] = mapped_column(Integer, default=0)
    source_type: Mapped[str] = mapped_column(String(64), default="")
    source_path: Mapped[str] = mapped_column(Text, default="")
    page: Mapped[int] = mapped_column(Integer, default=0)
    evidence_text: Mapped[str] = mapped_column(Text, default="")


Index("ix_rag_facts_namespace_key", RAGFact.namespace, RAGFact.key)


@dataclass
class RetrievedChunk:
    text: str
    score: float
    meta: Dict[str, Any]


# ============================================================
# RAGStore
# ============================================================
class RAGStore:
    def __init__(self, database_url: str, ollama_host: str, embed_model: str, embed_dims: int = 768):
        self.database_url = database_url
        self.engine = create_async_engine(database_url, echo=False, future=True)
        self.SessionLocal = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        self.embedder = OllamaEmbedder(host=ollama_host, model=embed_model)
        self.embed_dims = embed_dims

    def _zero_embedding(self) -> List[float]:
        return [0.0] * int(self.embed_dims)

    @staticmethod
    def _normalize_ws(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).strip()

    @staticmethod
    def _tokenize_search_terms(text: str) -> List[str]:
        raw = re.findall(r"[\w\u0E00-\u0E7F]{2,}", (text or "").lower(), flags=re.UNICODE)
        seen = set()
        out: List[str] = []
        stopwords = {
            "และ", "หรือ", "ของ", "ที่", "ใน", "จาก", "กับ", "เป็น", "ให้", "ได้", "มาก", "แล้ว",
            "what", "when", "where", "which", "with", "have", "this", "that", "จาก", "ค่ะ", "ครับ",
        }
        for token in raw:
            if token in stopwords or token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    def _fallback_chunk_by_size(self, text: str, target_chars: int = 900, min_chunk_chars: int = 100) -> List[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            cleaned = self._normalize_ws(text)
            return [cleaned] if len(cleaned) >= min_chunk_chars else []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        for sentence in sentences:
            s = self._normalize_ws(sentence)
            if not s:
                continue
            projected = current_len + len(s) + (1 if current else 0)
            if current and projected > target_chars:
                chunk = " ".join(current).strip()
                if chunk:
                    chunks.append(chunk)
                current = [s]
                current_len = len(s)
            else:
                current.append(s)
                current_len = projected

        if current:
            chunk = " ".join(current).strip()
            if chunk:
                chunks.append(chunk)

        merged: List[str] = []
        for chunk in chunks:
            if merged and len(chunk) < min_chunk_chars:
                merged[-1] = (merged[-1] + " " + chunk).strip()
            else:
                merged.append(chunk)
        return merged

    @staticmethod
    def _lexical_score(question_tokens: List[str], text: str) -> float:
        hay = (text or "").lower()
        if not hay:
            return 0.0
        score = 0.0
        for token in question_tokens:
            count = hay.count(token)
            if count:
                score += min(3.0, 1.0 + math.log1p(count))
        if question_tokens:
            coverage = sum(1 for token in question_tokens if token in hay) / len(question_tokens)
            score += coverage * 3.0
        return score

    async def _query_lexical(self, namespace: str, question: str, top_k: int, min_score: float) -> List[RetrievedChunk]:
        tokens = self._tokenize_search_terms(question)
        async with self.SessionLocal() as db:
            stmt = (
                select(RAGChunk)
                .where(RAGChunk.namespace == namespace)
                .order_by(RAGChunk.source_path, RAGChunk.page, RAGChunk.chunk_index)
                .limit(max(top_k * 40, 200))
            )
            res = await db.execute(stmt)
            rows = [r[0] for r in res.all()]

        scored: List[RetrievedChunk] = []
        for ch in rows:
            score = self._lexical_score(tokens, ch.text or "")
            if score < min_score:
                continue
            scored.append(RetrievedChunk(
                text=ch.text or "",
                score=score,
                meta={
                    "namespace": ch.namespace,
                    "source_type": ch.source_type,
                    "source_url": ch.source_url,
                    "source_path": ch.source_path,
                    "page": ch.page,
                    "chunk_index": ch.chunk_index,
                    "chunk_total": ch.chunk_total,
                    "score": score,
                    "retrieval_mode": "lexical_fallback",
                    "text_preview": (ch.text or "")[:360],
                    "warning": f"Semantic embeddings unavailable for query; used lexical fallback on stored chunks in namespace '{namespace}'.",
                }
            ))

        scored.sort(key=lambda x: (-float(x.score or 0.0), x.meta.get("source_path", ""), x.meta.get("chunk_index", 0)))
        return scored[:top_k]

    async def init_db(self) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.run_sync(Base.metadata.create_all)

    async def reset_db(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

    # ============================================================
    # ✅ Semantic Chunking
    # ============================================================
    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """คำนวณ cosine similarity ระหว่าง 2 vectors"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-9)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        แบ่งประโยคสำหรับภาษาไทยและอังกฤษ
        ใช้ newline และ punctuation เป็น boundary
        """
        # normalize newlines ก่อน
        text = re.sub(r"\r\n|\r", "\n", text)

        # แบ่งที่ newline หรือหลัง . ! ? ที่ตามด้วย space/newline
        parts = re.split(r"(?<=[.!?])\s+|\n+", text)

        sentences = [p.strip() for p in parts if p.strip()]
        return sentences

    async def semantic_chunk(
        self,
        text: str,
        similarity_threshold: float = 0.85,
        min_chunk_chars: int = 100,
    ) -> List[str]:
        """
        แบ่ง text เป็น chunks ตามความหมาย

        วิธีการ:
        1. แบ่งเป็นประโยคก่อน
        2. embed ทุกประโยคพร้อมกัน
        3. เปรียบ cosine similarity ระหว่างประโยคติดกัน
        4. ถ้า similarity < threshold = เปลี่ยนเรื่อง → ตัด chunk ตรงนั้น

        Args:
            text: เนื้อหาทั้งหมด
            similarity_threshold: ค่า similarity ต่ำกว่านี้ = เปลี่ยนเรื่อง (0.0-1.0)
                                  - ค่าสูง (0.90+) = ตัดถี่ ได้ chunk เล็กลง
                                  - ค่าต่ำ (0.75-) = ตัดห่าง ได้ chunk ใหญ่ขึ้น
            min_chunk_chars: chunk ที่สั้นกว่านี้จะถูก merge กับ chunk ก่อนหน้า

        Returns:
            List[str]: chunks ที่แบ่งตามความหมาย
        """
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)

        # ถ้ามีแค่ประโยคเดียว ไม่ต้อง embed
        if len(sentences) <= 1:
            return [s for s in sentences if len(s) >= min_chunk_chars]

        # embed ทุกประโยคพร้อมกัน (ใช้ embed_many เพื่อ concurrency)
        try:
            embeddings = await self.embedder.embed_many(sentences)
        except Exception:
            return self._fallback_chunk_by_size(text, min_chunk_chars=min_chunk_chars)

        # วน compare similarity ระหว่างประโยคติดกัน
        chunks: List[str] = []
        current_sentences: List[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])

            if sim < similarity_threshold:
                # similarity ต่ำ = เปลี่ยนเรื่อง → ปิด chunk เก่า
                chunk_text = " ".join(current_sentences).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                # เริ่ม chunk ใหม่
                current_sentences = [sentences[i]]
            else:
                # ยังเรื่องเดียวกัน → สะสมใน chunk ปัจจุบัน
                current_sentences.append(sentences[i])

        # chunk สุดท้าย
        if current_sentences:
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(chunk_text)

        # merge chunk ที่สั้นเกินไปเข้ากับ chunk ก่อนหน้า
        merged: List[str] = []
        for chunk in chunks:
            if merged and len(chunk) < min_chunk_chars:
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)

        return [c for c in merged if c.strip()]

    # ============================================================
    # Upsert
    # ============================================================
    async def upsert_chunks(self, namespace: str, chunks: List[str], metas: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        embedding_warning = ""
        try:
            embs = await self.embedder.embed_many(chunks)
        except Exception as e:
            embs = [self._zero_embedding() for _ in chunks]
            embedding_warning = str(e)
        rows = []
        for t, m, e in zip(chunks, metas, embs):
            source_key = (
                f"{namespace}|{m.get('source_type','')}|{m.get('source_url','')}|"
                f"{m.get('source_path','')}|{m.get('page','')}|{m.get('chunk_index','')}"
            )
            rid = sha1_text(source_key + "|" + t)
            rows.append({
                "id": rid,
                "namespace": namespace,
                "source_type": m.get("source_type", ""),
                "source_url": m.get("source_url", ""),
                "source_path": m.get("source_path", ""),
                "page": int(m.get("page") or 0),
                "chunk_index": int(m.get("chunk_index") or 1),
                "chunk_total": int(m.get("chunk_total") or 1),
                "text": t,
                "embedding": e,
                "score_hint": float(m.get("relevance_score") or 0.0) if not embedding_warning else -1.0,
            })

        async with self.SessionLocal() as db:
            ids = [r["id"] for r in rows]
            await db.execute(delete(RAGChunk).where(RAGChunk.id.in_(ids)))
            await db.execute(insert(RAGChunk), rows)
            await db.commit()
        return len(rows)

    async def upsert_facts(self, namespace: str, facts: List[Dict[str, Any]], meta: Dict[str, Any]) -> int:
        if not facts:
            return 0
        rows = []
        for f in facts:
            key = (
                f"{namespace}|{f.get('entity','')}|{f.get('key','')}|"
                f"{f.get('value','')}|{f.get('year',0)}|"
                f"{meta.get('source_path','')}|{meta.get('page',0)}"
            )
            rid = sha1_text(key)
            rows.append({
                "id": rid,
                "namespace": namespace,
                "entity": f.get("entity", "unknown"),
                "key": f.get("key", ""),
                "value": str(f.get("value", "")),
                "unit": f.get("unit", ""),
                "year": int(f.get("year") or 0),
                "source_type": meta.get("source_type", ""),
                "source_path": meta.get("source_path", ""),
                "page": int(meta.get("page") or 0),
                "evidence_text": f.get("evidence_text", ""),
            })

        async with self.SessionLocal() as db:
            ids = [r["id"] for r in rows]
            await db.execute(delete(RAGFact).where(RAGFact.id.in_(ids)))
            await db.execute(insert(RAGFact), rows)
            await db.commit()
        return len(rows)

    # ============================================================
    # ✅ Ingest จาก content.txt ด้วย Semantic Chunking
    # ============================================================
    async def ingest_from_content_txt(
        self,
        namespace: str,
        content_path: str,
        source_url: str = "",
        source_type: str = "web_scrape",
        similarity_threshold: float = 0.85,
        min_chunk_chars: int = 100,
    ) -> Dict[str, Any]:
        """
        อ่าน content.txt → semantic_chunk() → upsert เข้า DB

        Args:
            namespace: ชื่อ namespace ใน DB
            content_path: path ของ content.txt
            source_url: URL ต้นทาง
            source_type: ประเภทของ source
            similarity_threshold: ค่า threshold สำหรับ semantic chunking
            min_chunk_chars: ความยาวขั้นต่ำของแต่ละ chunk

        Returns:
            dict: { "count": int, "chunks_preview": List[str] }
        """
        path = Path(content_path)
        if not path.exists():
            raise FileNotFoundError(f"content.txt not found: {content_path}")

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return {"count": 0, "chunks_preview": []}

        # ✅ Semantic chunking
        chunks = await self.semantic_chunk(
            text=text,
            similarity_threshold=similarity_threshold,
            min_chunk_chars=min_chunk_chars,
        )

        if not chunks:
            return {"count": 0, "chunks_preview": []}

        metas = [
            {
                "source_type": source_type,
                "source_url": source_url,
                "source_path": str(path),
                "page": 0,
                "chunk_index": idx + 1,
                "chunk_total": len(chunks),
            }
            for idx in range(len(chunks))
        ]

        count = await self.upsert_chunks(namespace, chunks, metas)

        return {
            "count": count,
            # preview 100 chars แรกของแต่ละ chunk เพื่อ debug
            "chunks_preview": [c[:100] + "..." if len(c) > 100 else c for c in chunks],
        }

    async def ingest_auto(
        self,
        namespace: str,
        folder: str,
        source_url: str = "",
        source_type: str = "web_scrape",
        similarity_threshold: float = 0.85,
        min_chunk_chars: int = 100,
    ) -> Dict[str, Any]:
        """
        ✅ Smart ingest: อ่านทั้ง content.txt และ pdf_texts/*.txt แล้ว semantic chunk

        Args:
            namespace: ชื่อ namespace ใน DB
            folder: path ของ folder ที่ scraper สร้าง
            source_url: URL ต้นทาง
            source_type: ประเภทของ source
            similarity_threshold: ค่า threshold สำหรับ semantic chunking
            min_chunk_chars: ความยาวขั้นต่ำของแต่ละ chunk

        Returns:
            dict: {
                "total_count": int,          — chunk ทั้งหมดที่ upsert
                "content_count": int,        — chunk จาก content.txt
                "pdf_count": int,            — chunk จาก PDF ทั้งหมด
                "pdf_files": List[str],      — รายชื่อ PDF ที่ ingest
            }
        """
        folder_path = Path(folder)
        total_count = 0
        content_count = 0
        pdf_count = 0
        pdf_files_ingested: List[str] = []

        # ── 1) content.txt (เนื้อหาจากเว็บ) ─────────────────────────
        content_txt = folder_path / "content.txt"
        if content_txt.exists() and content_txt.stat().st_size > 0:
            result = await self.ingest_from_content_txt(
                namespace=namespace,
                content_path=str(content_txt),
                source_url=source_url,
                source_type=source_type,
                similarity_threshold=similarity_threshold,
                min_chunk_chars=min_chunk_chars,
            )
            content_count = result["count"]
            total_count += content_count

        # ── 2) pdf_texts/*.txt (text ที่ extract จาก PDF) ────────────
        pdf_texts_dir = folder_path / "pdf_texts"
        if pdf_texts_dir.exists():
            txt_files = sorted(pdf_texts_dir.glob("*.txt"))

            for txt_file in txt_files:
                if txt_file.stat().st_size == 0:
                    continue
                try:
                    result = await self.ingest_from_content_txt(
                        namespace=namespace,
                        content_path=str(txt_file),
                        source_url=source_url,
                        # ✅ บอกว่ามาจาก PDF เพื่อแยก metadata ได้ชัดเจน
                        source_type="pdf",
                        similarity_threshold=similarity_threshold,
                        min_chunk_chars=min_chunk_chars,
                    )
                    pdf_count += result["count"]
                    total_count += result["count"]
                    pdf_files_ingested.append(txt_file.name)
                except Exception as e:
                    print(f"[ingest_auto] skip {txt_file.name}: {e}")

        return {
            "total_count": total_count,
            "content_count": content_count,
            "pdf_count": pdf_count,
            "pdf_files": pdf_files_ingested,
        }

    # ============================================================
    # Query
    # ============================================================
    async def query_semantic(
        self,
        namespace: str,
        question: str,
        top_k: int = 15,
        min_score: float = 0.01,
    ) -> List[RetrievedChunk]:
        """
        ค้นหา chunks ที่ใกล้เคียงกับ question มากที่สุด

        Args:
            namespace: ชื่อ namespace ใน DB
            question: คำถามที่ต้องการค้นหา
            top_k: จำนวน chunks สูงสุดที่จะ return (default 15)
            min_score: ตัด chunk ที่ score ต่ำกว่านี้ทิ้ง
        """
        try:
            q_emb = await self.embedder.embed_one(question)
        except Exception:
            return await self._query_lexical(namespace, question, top_k=top_k, min_score=min_score)
        dist_expr = RAGChunk.embedding.l2_distance(q_emb)

        async with self.SessionLocal() as db:
            stmt = (
                select(RAGChunk, dist_expr.label("dist"))
                .where(RAGChunk.namespace == namespace, RAGChunk.score_hint >= 0.0)
                .order_by(dist_expr)
                .limit(top_k)
            )
            res = await db.execute(stmt)
            rows = res.all()
        if not rows:
            return await self._query_lexical(namespace, question, top_k=top_k, min_score=min_score)

        out: List[RetrievedChunk] = []
        for row in rows:
            ch: RAGChunk = row[0]
            dist = float(row[1]) if row[1] is not None else 999.0
            score = float(1.0 / (1.0 + dist))

            if score < min_score:
                continue

            out.append(RetrievedChunk(
                text=ch.text or "",
                score=score,
                meta={
                    "namespace": ch.namespace,
                    "source_type": ch.source_type,
                    "source_url": ch.source_url,
                    "source_path": ch.source_path,
                    "page": ch.page,
                    "chunk_index": ch.chunk_index,
                    "chunk_total": ch.chunk_total,
                    "score": score,
                    "retrieval_mode": "semantic",
                    "text_preview": (ch.text or "")[:360],
                }
            ))

        # ✅ เรียงตาม chunk_index เพื่อให้ลำดับตรงกับบทความต้นทาง
        out.sort(key=lambda x: (x.meta.get("source_path", ""), x.meta.get("chunk_index", 0)))
        return out

    async def query_structured(self, namespace: str, key: str, limit: int = 30, source_contains: str = "") -> List[Dict[str, Any]]:
        async with self.SessionLocal() as db:
            stmt = select(RAGFact).where(RAGFact.namespace == namespace, RAGFact.key == key)
            if source_contains:
                stmt = stmt.where(RAGFact.source_path.contains(source_contains))
            stmt = stmt.order_by(RAGFact.year.asc(), RAGFact.entity.asc()).limit(limit)
            res = await db.execute(stmt)
            facts = [r[0] for r in res.all()]
        return [{
            "namespace": f.namespace, "entity": f.entity, "key": f.key,
            "value": f.value, "unit": f.unit, "year": f.year,
            "source_path": f.source_path, "page": f.page, "evidence_text": f.evidence_text,
        } for f in facts]

    async def get_source_chunks(
        self,
        namespace: str,
        source_url: str = "",
        source_path: str = "",
        limit: int = 50,
    ) -> List[RetrievedChunk]:
        if not source_url and not source_path:
            return []

        async with self.SessionLocal() as db:
            stmt = select(RAGChunk).where(RAGChunk.namespace == namespace)
            if source_url:
                stmt = stmt.where(RAGChunk.source_url == source_url)
            if source_path:
                stmt = stmt.where(RAGChunk.source_path == source_path)
            stmt = stmt.order_by(RAGChunk.chunk_index).limit(limit)
            res = await db.execute(stmt)
            rows = [r[0] for r in res.all()]

        return [
            RetrievedChunk(
                text=c.text or "",
                score=float(c.score_hint or 0.0),
                meta={
                    "namespace": c.namespace,
                    "source_type": c.source_type,
                    "source_url": c.source_url,
                    "source_path": c.source_path,
                    "page": c.page,
                    "chunk_index": c.chunk_index,
                    "chunk_total": c.chunk_total,
                    "score": float(c.score_hint or 0.0),
                    "text_preview": (c.text or "")[:360],
                },
            )
            for c in rows
        ]

    async def preview_chunks(self, namespace: str, source_type: str = "", limit: int = 30) -> List[Dict[str, Any]]:
        async with self.SessionLocal() as db:
            stmt = select(RAGChunk).where(RAGChunk.namespace == namespace)
            if source_type:
                stmt = stmt.where(RAGChunk.source_type == source_type)
            stmt = stmt.order_by(RAGChunk.source_type, RAGChunk.source_path, RAGChunk.page).limit(limit)
            res = await db.execute(stmt)
            rows = [r[0] for r in res.all()]
        return [{
            "id": c.id, "namespace": c.namespace, "source_type": c.source_type,
            "source_path": c.source_path, "page": c.page,
            "chunk_index": c.chunk_index, "chunk_total": c.chunk_total,
            "text_preview": (c.text or "")[:260],
        } for c in rows]
