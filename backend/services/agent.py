"""Agent Orchestrator: decides whether to use Vector Search or Text-to-SQL."""

import httpx
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from backend.config import get_settings
from backend.services.rag import vector_search, execute_text_to_sql

settings = get_settings()


async def classify_query_intent(question: str) -> str:
    """Use LLM to classify whether a question needs vector search or SQL.

    Returns: 'vector', 'sql', or 'hybrid'
    """
    prompt = (
        "Classify the following question into one category:\n"
        "- 'sql': if it asks for specific numbers, statistics, comparisons, "
        "rankings, or data that would be in a table (e.g., revenue, profit, growth rate)\n"
        "- 'vector': if it asks for explanations, concepts, summaries, "
        "or general information from documents\n"
        "- 'hybrid': if it needs both tabular data AND contextual explanation\n\n"
        f"Question: {question}\n\n"
        "Reply with ONLY one word: sql, vector, or hybrid"
    )

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{settings.OLLAMA_HOST}/api/generate",
                json={
                    "model": settings.OLLAMA_LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 10}
                }
            )
            response.raise_for_status()
            intent = response.json().get("response", "").strip().lower()

            if "sql" in intent:
                return "sql"
            elif "hybrid" in intent:
                return "hybrid"
            else:
                return "vector"
    except Exception:
        return "vector"  # Default to vector search


async def generate_final_answer(
    question: str,
    context: str,
    method: str,
) -> str:
    """Generate a final answer using the retrieved context."""
    prompt = (
        "คุณคือผู้ช่วย AI ที่เชี่ยวชาญในการวิเคราะห์เอกสารและตอบคำถาม\n\n"
        "กฎสำคัญ:\n"
        "1. ตอบเป็นภาษาไทยเท่านั้น\n"
        "2. อ่านบริบทด้านล่างอย่างละเอียดทุก chunk แล้วดึงข้อมูลที่ตรงกับคำถามมาตอบ\n"
        "3. ถ้าในบริบทมีตัวเลข สถิติ ปี พ.ศ./ค.ศ. เปอร์เซ็นต์ ชื่อองค์กร ให้อ้างอิงตัวเลขเหล่านั้นในคำตอบด้วย\n"
        "4. ตอบให้เป็นหัวข้อย่อยหรือข้อๆ พร้อมรายละเอียด\n"
        "5. ห้ามแต่งข้อมูลเอง ให้ใช้เฉพาะข้อมูลจากบริบทที่ให้มาเท่านั้น\n"
        "6. ถ้าบริบทมีข้อมูลที่เกี่ยวข้องแม้เพียงบางส่วน ให้ตอบจากส่วนนั้น ห้ามบอกว่าไม่มีข้อมูล\n"
        "7. ข้ามข้อมูลที่ไม่เกี่ยวข้องกับคำถาม\n\n"
        f"=== บริบท ===\n{context[:8000]}\n=== จบบริบท ===\n\n"
        f"คำถาม: {question}\n\n"
        f"คำตอบ (ภาษาไทย อ้างอิงตัวเลขและสถิติจากบริบท):"
    )

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{settings.OLLAMA_HOST}/api/generate",
                json={
                    "model": settings.OLLAMA_LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 2000}
                }
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
    except Exception as e:
        print(f"⚠️ LLM Generation error details: {e.__class__.__name__}: {str(e)}")
        return f"เกิดข้อผิดพลาดในการสร้างคำตอบ: {str(e)}"


async def agent_query(
    question: str,
    session: AsyncSession,
) -> Dict[str, Any]:
    """Main agent entry point: classify intent → retrieve → answer.

    Returns dict with: answer, method, sources, sql_info (if applicable)
    """
    # Step 1: Classify intent
    intent = await classify_query_intent(question)

    context = ""
    sources = []
    sql_info = None

    # Step 2: Retrieve based on intent
    if intent in ("vector", "hybrid"):
        vector_results = await vector_search(question, session, top_k=10)
        for r in vector_results:
            context += f"[Document: {r['filename']}, Chunk {r['chunk_index']}]\n"
            if r.get("summary"):
                context += f"Summary: {r['summary']}\n"
            context += f"{r['text']}\n\n"
            sources.append({
                "type": "vector",
                "filename": r["filename"],
                "chunk_index": r["chunk_index"],
                "similarity": r["similarity"],
            })

    if intent in ("sql", "hybrid"):
        sql_result = await execute_text_to_sql(question, session)
        sql_info = sql_result
        if sql_result["success"] and sql_result["results"]:
            context += "\n[SQL Query Results]\n"
            context += f"SQL: {sql_result['sql']}\n"
            for row in sql_result["results"][:20]:
                context += str(row) + "\n"
            sources.append({
                "type": "sql",
                "sql": sql_result["sql"],
                "row_count": sql_result["row_count"],
            })

    # Fallback: if SQL-only returned nothing, try vector search
    if not context.strip() and intent == "sql":
        print("⚠️ SQL returned no results, falling back to vector search...")
        intent = "hybrid"  # update method to reflect fallback
        vector_results = await vector_search(question, session, top_k=10)
        for r in vector_results:
            context += f"[Document: {r['filename']}, Chunk {r['chunk_index']}]\n"
            if r.get("summary"):
                context += f"Summary: {r['summary']}\n"
            context += f"{r['text']}\n\n"
            sources.append({
                "type": "vector",
                "filename": r["filename"],
                "chunk_index": r["chunk_index"],
                "similarity": r["similarity"],
            })

    # Step 3: Generate answer
    if not context.strip():
        answer = "ไม่พบข้อมูลที่เกี่ยวข้องในระบบ กรุณาลองถามคำถามอื่นหรืออัปโหลดเอกสารเพิ่มเติม"
    else:
        answer = await generate_final_answer(question, context, intent)

    return {
        "answer": answer,
        "method": intent,
        "sources": sources,
        "sql_info": sql_info,
    }
