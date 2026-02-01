# backend/rag/rag_store.py
from __future__ import annotations

import hashlib
import asyncio
from dataclasses import dataclass
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


class OllamaEmbedder:
    def __init__(self, host: str, model: str, timeout_s: int = 60, max_concurrency: int = 4):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.max_concurrency = max(1, int(max_concurrency))

    async def embed_one(self, text: str) -> List[float]:
        payload = {"model": self.model, "prompt": text}
        timeout = httpx.Timeout(self.timeout_s)
        async with httpx.AsyncClient(timeout=timeout) as client:
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


class RAGChunk(Base):
    __tablename__ = "rag_chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(32), index=True)     # external / internal
    source_type: Mapped[str] = mapped_column(String(64), index=True)
    source_url: Mapped[str] = mapped_column(Text, default="")
    source_path: Mapped[str] = mapped_column(Text, default="")
    page: Mapped[int] = mapped_column(Integer, default=0)
    chunk_index: Mapped[int] = mapped_column(Integer, default=1)
    chunk_total: Mapped[int] = mapped_column(Integer, default=1)
    text: Mapped[str] = mapped_column(Text)

    # ✅ dim ตาม settings (ถ้าเปลี่ยน dim ต้อง reset/migrate DB)
    embedding: Mapped[Any] = mapped_column(Vector(int(settings.EMBED_DIMS)))
    score_hint: Mapped[float] = mapped_column(Float, default=0.0)


Index("ix_rag_chunks_namespace_source", RAGChunk.namespace, RAGChunk.source_type)


class RAGFact(Base):
    __tablename__ = "rag_facts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(32), index=True)     # external / internal
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


class RAGStore:
    def __init__(self, database_url: str, ollama_host: str, embed_model: str, embed_dims: int = 768):
        self.database_url = database_url
        self.engine = create_async_engine(database_url, echo=False, future=True)
        self.SessionLocal = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)
        self.embedder = OllamaEmbedder(host=ollama_host, model=embed_model)
        self.embed_dims = embed_dims

    async def init_db(self) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.run_sync(Base.metadata.create_all)

    async def reset_db(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

    async def upsert_chunks(self, namespace: str, chunks: List[str], metas: List[Dict[str, Any]]) -> int:
        if not chunks:
            return 0

        embs = await self.embedder.embed_many(chunks)

        rows = []
        for t, m, e in zip(chunks, metas, embs):
            source_key = f"{namespace}|{m.get('source_type','')}|{m.get('source_url','')}|{m.get('source_path','')}|{m.get('page','')}|{m.get('chunk_index','')}"
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
                "score_hint": float(m.get("relevance_score") or 0.0),
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
            key = f"{namespace}|{f.get('entity','')}|{f.get('key','')}|{f.get('value','')}|{f.get('year',0)}|{meta.get('source_path','')}|{meta.get('page',0)}"
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

    async def query_semantic(self, namespace: str, question: str, top_k: int = 8) -> List[RetrievedChunk]:
        q_emb = await self.embedder.embed_one(question)

        dist_expr = RAGChunk.embedding.l2_distance(q_emb)

        async with self.SessionLocal() as db:
            stmt = (
                select(RAGChunk, dist_expr.label("dist"))
                .where(RAGChunk.namespace == namespace)
                .order_by(dist_expr)  # ✅ ชัวร์กว่า order_by("dist")
                .limit(top_k)
            )
            res = await db.execute(stmt)
            rows = res.all()

        out: List[RetrievedChunk] = []
        for row in rows:
            ch: RAGChunk = row[0]
            dist = float(row[1]) if row[1] is not None else 999.0
            score = float(1.0 / (1.0 + dist))
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
                }
            ))
        return out

    async def query_structured(self, namespace: str, key: str, limit: int = 30) -> List[Dict[str, Any]]:
        async with self.SessionLocal() as db:
            stmt = select(RAGFact).where(RAGFact.namespace == namespace, RAGFact.key == key).limit(limit)
            res = await db.execute(stmt)
            facts = [r[0] for r in res.all()]

        out = []
        for f in facts:
            out.append({
                "namespace": f.namespace,
                "entity": f.entity,
                "key": f.key,
                "value": f.value,
                "unit": f.unit,
                "year": f.year,
                "source_path": f.source_path,
                "page": f.page,
                "evidence_text": f.evidence_text,
            })
        return out

    async def preview_chunks(self, namespace: str, source_type: str = "", limit: int = 30) -> List[Dict[str, Any]]:
        async with self.SessionLocal() as db:
            stmt = select(RAGChunk).where(RAGChunk.namespace == namespace)
            if source_type:
                stmt = stmt.where(RAGChunk.source_type == source_type)
            stmt = stmt.order_by(RAGChunk.source_type, RAGChunk.source_path, RAGChunk.page).limit(limit)
            res = await db.execute(stmt)
            rows = [r[0] for r in res.all()]

        return [{
            "id": c.id,
            "namespace": c.namespace,
            "source_type": c.source_type,
            "source_path": c.source_path,
            "page": c.page,
            "chunk_index": c.chunk_index,
            "chunk_total": c.chunk_total,
            "text_preview": (c.text or "")[:260],
        } for c in rows]
