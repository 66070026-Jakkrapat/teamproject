"""Semantic + LLM chunking pipeline.

1. Semantic Split: Split text by semantic similarity boundaries
2. LLM Enrichment: Generate a summary for each chunk to prevent context loss
"""

import re
import httpx
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
from backend.config import get_settings
from backend.services.embedding import get_embedding, get_embeddings_batch

settings = get_settings()


@dataclass
class ChunkResult:
    """Result of chunking a document."""
    text: str
    summary: str = ""
    embedding: List[float] = field(default_factory=list)
    chunk_index: int = 0
    token_count: int = 0
    start_char: int = 0
    end_char: int = 0


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using Thai and English delimiters."""
    # Split on Thai sentence endings, periods, newlines
    parts = re.split(r'(?<=[。\.\!\?\n])\s*|(?<=\n)\s*', text)
    sentences = [s.strip() for s in parts if s.strip()]

    # If too few sentences, also split on double spaces
    if len(sentences) < 3:
        new_sentences = []
        for s in sentences:
            sub = re.split(r'\s{2,}', s)
            new_sentences.extend([x.strip() for x in sub if x.strip()])
        sentences = new_sentences

    return sentences


def _cosine_distance(a: List[float], b: List[float]) -> float:
    """Calculate cosine distance between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    cos_sim = np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-10)
    return 1.0 - cos_sim


async def semantic_split(
    text: str,
    threshold: float = 0.5,
    min_chunk_size: int = 100,
    max_chunk_size: int = 2000,
) -> List[str]:
    """Split text into chunks based on semantic similarity boundaries.

    Embeds each sentence, finds points of high semantic distance,
    and splits at those boundaries.
    """
    sentences = _split_into_sentences(text)

    if len(sentences) <= 1:
        return [text] if text.strip() else []

    # Get embeddings for all sentences
    embeddings = await get_embeddings_batch(sentences)

    # Calculate cosine distances between consecutive sentences
    distances = []
    for i in range(len(embeddings) - 1):
        dist = _cosine_distance(embeddings[i], embeddings[i + 1])
        distances.append(dist)

    # Find split points where distance exceeds threshold
    if distances:
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        dynamic_threshold = mean_dist + std_dist * threshold
    else:
        dynamic_threshold = threshold

    chunks = []
    current_chunk_sentences = [sentences[0]]
    current_length = len(sentences[0])

    for i, dist in enumerate(distances):
        next_sentence = sentences[i + 1]
        next_length = current_length + len(next_sentence)

        # Split if: semantic boundary detected AND chunk is large enough
        #        OR: chunk exceeds max size
        should_split = (
            (dist > dynamic_threshold and current_length >= min_chunk_size) or
            next_length > max_chunk_size
        )

        if should_split:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [next_sentence]
            current_length = len(next_sentence)
        else:
            current_chunk_sentences.append(next_sentence)
            current_length = next_length

    # Add the last chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks


async def generate_chunk_summary(chunk_text: str) -> str:
    """Use Ollama LLM to generate a brief summary for a chunk."""
    prompt = (
        "สรุปข้อความต่อไปนี้ให้สั้นกระชับใน 1-2 ประโยค เป็นภาษาไทย:\n\n"
        f"{chunk_text[:1500]}\n\n"
        "สรุป:"
    )

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.OLLAMA_HOST}/api/generate",
                json={
                    "model": settings.OLLAMA_LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 150}
                }
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
    except Exception as e:
        print(f"⚠️ LLM summary generation failed: {e}")
        return ""


async def chunk_document(
    text: str,
    semantic_threshold: float = 0.5,
    generate_summaries: bool = True,
) -> List[ChunkResult]:
    """Full chunking pipeline: semantic split + LLM enrichment + embeddings.

    Args:
        text: The cleaned document text
        semantic_threshold: Sensitivity for semantic splitting (higher = fewer splits)
        generate_summaries: Whether to generate LLM summaries for each chunk

    Returns:
        List of ChunkResult objects with text, summary, and embedding
    """
    # Step 1: Semantic split
    chunk_texts = await semantic_split(text, threshold=semantic_threshold)

    if not chunk_texts:
        return []

    results = []
    char_offset = 0

    for i, chunk_text in enumerate(chunk_texts):
        # Step 2: LLM enrichment (generate summary)
        summary = ""
        if generate_summaries:
            summary = await generate_chunk_summary(chunk_text)

        # Step 3: Create embedding (combine text + summary for richer embedding)
        embed_input = f"{summary}\n{chunk_text}" if summary else chunk_text
        embedding = await get_embedding(embed_input)

        result = ChunkResult(
            text=chunk_text,
            summary=summary,
            embedding=embedding,
            chunk_index=i,
            token_count=len(chunk_text.split()),
            start_char=char_offset,
            end_char=char_offset + len(chunk_text),
        )
        results.append(result)
        char_offset += len(chunk_text) + 1  # +1 for the space between chunks

    return results
