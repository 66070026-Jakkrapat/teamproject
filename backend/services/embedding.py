"""Embedding service using Ollama's nomic-embed-text model."""

import httpx
from typing import List
from backend.config import get_settings

settings = get_settings()


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector for a single text."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{settings.OLLAMA_HOST}/api/embeddings",
            json={
                "model": settings.EMBED_MODEL,
                "prompt": text,
            }
        )
        response.raise_for_status()
        return response.json()["embedding"]


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for a batch of texts."""
    embeddings = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        for text in texts:
            response = await client.post(
                f"{settings.OLLAMA_HOST}/api/embeddings",
                json={
                    "model": settings.EMBED_MODEL,
                    "prompt": text,
                }
            )
            response.raise_for_status()
            embeddings.append(response.json()["embedding"])
    return embeddings
