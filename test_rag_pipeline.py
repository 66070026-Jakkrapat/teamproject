"""Test the improved RAG pipeline with top_k=10 and new prompt."""
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(__file__))

async def test_improved():
    from backend.database import AsyncSessionLocal
    from backend.services.agent import agent_query

    async with AsyncSessionLocal() as session:
        result = await agent_query(
            "ในปี 2025 SME ควรปรับตัวรับมือกับเทรนด์ Pet Parent อย่างไรบ้าง ขอตัวอย่างไอเดียธุรกิจ",
            session
        )

        lines = []
        lines.append(f"Method: {result['method']}")
        lines.append(f"Sources ({len(result['sources'])}):")
        for s in result['sources']:
            lines.append(f"  - {s.get('filename')} chunk {s.get('chunk_index')} sim={s.get('similarity', 0):.4f}")
        lines.append(f"\nAnswer:\n{result['answer']}")

        with open("test_improved_result.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("Saved to test_improved_result.txt")

asyncio.run(test_improved())
