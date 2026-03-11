# backend/llm_client.py
"""
Unified LLM and Embedder abstraction.
Supports both OpenAI API and Ollama (local).
"""
from __future__ import annotations

import asyncio
from typing import List, Protocol

import httpx


# ── Protocols ────────────────────────────────────────────
class LLM(Protocol):
    async def generate(self, prompt: str) -> str: ...


class Embedder(Protocol):
    async def embed_one(self, text: str) -> List[float]: ...
    async def embed_many(self, texts: List[str]) -> List[List[float]]: ...


# ── OpenAI ───────────────────────────────────────────────
class OpenAILLM:
    """Chat-completion via OpenAI-compatible API."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", timeout_s: int = 120):
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    async def generate(self, prompt: str) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key, timeout=self.timeout_s)
        resp = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()


class OpenAIEmbedder:
    """Embeddings via OpenAI-compatible API."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small",
                 dims: int = 768, timeout_s: int = 60, max_concurrency: int = 4):
        self.api_key = api_key
        self.model = model
        self.dims = dims
        self.timeout_s = timeout_s
        self.max_concurrency = max(1, int(max_concurrency))

    async def embed_one(self, text: str) -> List[float]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key, timeout=self.timeout_s)
        resp = await client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dims,
        )
        return [float(x) for x in resp.data[0].embedding]

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key, timeout=self.timeout_s)
        # OpenAI supports batch embedding natively
        resp = await client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dims,
        )
        # sort by index to ensure correct order
        sorted_data = sorted(resp.data, key=lambda d: d.index)
        return [[float(x) for x in d.embedding] for d in sorted_data]


# ── Ollama ───────────────────────────────────────────────
class OllamaLLM:
    def __init__(self, host: str, model: str, timeout_s: int = 120):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    async def generate(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        timeout = httpx.Timeout(self.timeout_s)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{self.host}/api/generate", json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"Ollama generate failed: {r.status_code} {r.text[:200]}")
        data = r.json()
        return (data.get("response") or "").strip()


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


# ── Factory Functions ────────────────────────────────────
def create_llm() -> LLM:
    """Create an LLM instance based on LLM_PROVIDER setting."""
    from backend.settings import settings
    if settings.LLM_PROVIDER == "ollama":
        return OllamaLLM(host=settings.OLLAMA_HOST, model=settings.OLLAMA_LLM_MODEL)
    # default: openai
    return OpenAILLM(api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_MODEL)


def create_embedder(dims: int | None = None) -> Embedder:
    """Create an Embedder instance based on LLM_PROVIDER setting."""
    from backend.settings import settings
    actual_dims = dims or settings.EMBED_DIMS
    if settings.LLM_PROVIDER == "ollama":
        return OllamaEmbedder(host=settings.OLLAMA_HOST, model=settings.EMBED_MODEL)
    # default: openai
    return OpenAIEmbedder(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_EMBED_MODEL,
        dims=actual_dims,
    )
