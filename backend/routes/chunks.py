"""Chunk visualization API routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models import Document, Chunk

router = APIRouter()


@router.get("/{doc_id}")
async def get_chunks_for_visualization(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get chunks with full metadata for the Chunk Visualizer UI.

    Returns chunk text, summary, token count, position info, and
    inter-chunk similarity for boundary visualization.
    """
    # Verify document exists
    doc_result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Get all chunks ordered
    chunks_result = await db.execute(
        select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.chunk_index)
    )
    chunks = chunks_result.scalars().all()

    # Calculate inter-chunk similarities (for boundary visualization)
    import numpy as np

    chunk_data = []
    for i, chunk in enumerate(chunks):
        similarity_to_next = None
        similarity_to_prev = None

        if chunk.embedding is not None and i < len(chunks) - 1 and chunks[i + 1].embedding is not None:
            a = np.array(chunk.embedding)
            b = np.array(chunks[i + 1].embedding)
            cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
            similarity_to_next = cos_sim

        if chunk.embedding is not None and i > 0 and chunks[i - 1].embedding is not None:
            a = np.array(chunk.embedding)
            b = np.array(chunks[i - 1].embedding)
            cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
            similarity_to_prev = cos_sim

        chunk_data.append({
            "id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "chunk_text": chunk.chunk_text,
            "summary": chunk.summary,
            "token_count": chunk.token_count,
            "metadata": chunk.metadata_,
            "similarity_to_next": similarity_to_next,
            "similarity_to_prev": similarity_to_prev,
        })

    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "total_chunks": len(chunks),
        "chunks": chunk_data,
    }
