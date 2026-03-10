"""Diagnose what chunks are actually being retrieved for a Pet Parent query."""
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(__file__))

async def diagnose():
    from backend.database import AsyncSessionLocal
    from backend.services.rag import vector_search
    from sqlalchemy import text

    lines = []

    async with AsyncSessionLocal() as session:
        # 1. Check all documents in DB
        result = await session.execute(text("SELECT id, filename, status, length(raw_text) as text_len FROM documents"))
        docs = result.fetchall()
        lines.append("=== DOCUMENTS IN DB ===")
        for d in docs:
            lines.append(f"  ID={d[0]} | {d[1]} | status={d[2]} | text_len={d[3]}")

        # 2. Check chunk counts per document
        result = await session.execute(text("SELECT document_id, count(*), count(embedding) FROM chunks GROUP BY document_id"))
        chunks_info = result.fetchall()
        lines.append("\n=== CHUNKS PER DOCUMENT ===")
        for c in chunks_info:
            lines.append(f"  doc_id={c[0]} | total_chunks={c[1]} | with_embedding={c[2]}")

        # 3. Vector search for Pet Parent
        lines.append("\n=== VECTOR SEARCH: Pet Parent ===")
        results = await vector_search("ในปี 2025 SME ควรปรับตัวรับมือกับเทรนด์ Pet Parent อย่างไรบ้าง", session, top_k=5)
        for i, r in enumerate(results):
            lines.append(f"\n--- Result {i+1} (sim={r['similarity']:.4f}) ---")
            lines.append(f"  File: {r['filename']} | Chunk: {r['chunk_index']}")
            lines.append(f"  Text (first 300 chars): {r['text'][:300]}")

    with open("diagnose_result.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Saved to diagnose_result.txt")

asyncio.run(diagnose())
