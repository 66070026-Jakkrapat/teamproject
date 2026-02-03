import asyncio
from backend.settings import settings
from backend.rag.rag_store import RAGStore

async def main():
    store = RAGStore(
        database_url=settings.DATABASE_URL,
        ollama_host=settings.OLLAMA_HOST,
        embed_model=settings.EMBED_MODEL,
        embed_dims=settings.EMBED_DIMS,
    )
    await store.reset_db()
    print("✅ RAG DB reset complete (dropped & recreated tables).")

if __name__ == "__main__":
    asyncio.run(main())
