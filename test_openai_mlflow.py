import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from backend.services.agent import agent_query
from backend.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=1,
    max_overflow=2,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def run_test():
    print("🚀 Starting OpenAI + MLflow Test...")
    print(f"Tracking URI: {settings.MLFLOW_TRACKING_URI}")
    
    async with AsyncSessionLocal() as session:
        # A simple question to trigger the LLM
        question = "ช่วยสรุปให้ฟังหน่อยว่า GDP คืออะไรสั้นๆ"
        print(f"\nQuestion: {question}")
        
        try:
            result = await agent_query(question, session)
            print("\n✅ Answer Received:")
            print(result["answer"][:200] + "...\n" if len(result["answer"]) > 200 else result["answer"])
            print(f"Method used: {result['method']}")
            print("\n🎉 Test completed! Please check your DagsHub MLflow dashboard to see the latest run.")
        except Exception as e:
            print(f"\n❌ Error during test: {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_test())
