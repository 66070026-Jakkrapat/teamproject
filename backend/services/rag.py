"""Hybrid RAG engine: Vector Search + Text-to-SQL."""

import json
import httpx
from typing import List, Dict, Any, Optional
from sqlalchemy import text as sql_text, select
from sqlalchemy.ext.asyncio import AsyncSession
from backend.config import get_settings
from backend.services.embedding import get_embedding
from backend.models import Chunk, Document

settings = get_settings()


async def vector_search(
    query: str,
    session: AsyncSession,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Search chunks by semantic similarity using pgvector cosine distance."""
    query_embedding = await get_embedding(query)

    stmt = (
        select(Chunk, Document.filename, Chunk.embedding.cosine_distance(query_embedding).label("distance"))
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.embedding.is_not(None))
        .order_by(Chunk.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    result = await session.execute(stmt)
    rows = result.all()

    return [
        {
            "chunk_id": chunk.id,
            "text": chunk.chunk_text,
            "summary": chunk.summary,
            "chunk_index": chunk.chunk_index,
            "document_id": chunk.document_id,
            "filename": filename,
            "similarity": 1.0 - float(distance),
        }
        for chunk, filename, distance in rows
    ]


async def generate_sql_from_query(question: str, session: AsyncSession) -> str:
    """Use LLM to generate SQL from a natural language question.

    The SQL targets the structured_data table which stores table data in JSONB.
    """
    # Get sample schema info
    sample_result = await session.execute(
        sql_text("""
            SELECT table_name, headers::text
            FROM structured_data
            LIMIT 10
        """)
    )
    tables_info = sample_result.fetchall()

    schema_desc = "Available tables in structured_data:\n"
    for row in tables_info:
        schema_desc += f"- table_name: '{row[0]}', columns: {row[1]}\n"

    if not tables_info:
        schema_desc += "(No structured data tables available yet)\n"

    prompt = (
        "You are a SQL expert. Generate a PostgreSQL query to answer the user's question.\n"
        "The data is stored in a table called 'structured_data' with columns:\n"
        "- id (integer), document_id (integer), table_name (text), headers (jsonb), "
        "row_data (jsonb), row_index (integer)\n"
        "The row_data column contains JSONB with keys matching the headers.\n\n"
        f"{schema_desc}\n"
        f"Question: {question}\n\n"
        "Return ONLY the SQL query, no explanation. Use proper JSONB operators (->>, ->).\n"
        "SQL:"
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.OLLAMA_HOST}/api/generate",
                json={
                    "model": settings.OLLAMA_LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 500}
                }
            )
            response.raise_for_status()
            sql = response.json().get("response", "").strip()

            # Clean up the SQL (remove markdown fences if present)
            if sql.startswith("```"):
                sql = sql.split("```")[1]
                if sql.startswith("sql"):
                    sql = sql[3:]
                sql = sql.strip()

            return sql
    except Exception as e:
        print(f"⚠️ SQL generation failed: {e}")
        return ""


async def execute_text_to_sql(
    question: str,
    session: AsyncSession,
) -> Dict[str, Any]:
    """Generate and execute SQL from natural language question."""
    sql = await generate_sql_from_query(question, session)

    if not sql:
        return {"success": False, "error": "Could not generate SQL", "sql": "", "results": []}

    try:
        # Safety: only allow SELECT statements
        sql_upper = sql.upper().strip()
        if not sql_upper.startswith("SELECT"):
            return {"success": False, "error": "Only SELECT queries are allowed", "sql": sql, "results": []}

        result = await session.execute(sql_text(sql))
        rows = result.fetchall()
        columns = list(result.keys()) if result.keys() else []

        return {
            "success": True,
            "sql": sql,
            "columns": columns,
            "results": [dict(zip(columns, row)) for row in rows],
            "row_count": len(rows),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "sql": sql, "results": []}
