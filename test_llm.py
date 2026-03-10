"""Test Ollama LLM generation directly."""
import asyncio
import httpx

async def test_llm():
    print("Testing Ollama LLM generation...")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.1:8b",
                    "prompt": "Say hello in Thai, one sentence only.",
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 100}
                }
            )
            print(f"Status: {response.status_code}")
            data = response.json()
            print(f"Response: {data.get('response', 'NO RESPONSE KEY')}")
            print(f"Done: {data.get('done', 'NO DONE KEY')}")
    except Exception as e:
        print(f"ERROR: {e.__class__.__name__}: {e}")

asyncio.run(test_llm())
