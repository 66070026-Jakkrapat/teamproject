"""Document management API routes."""

import os
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from backend.database import get_db
from backend.models import Document, Chunk, StructuredData
from backend.services.ocr import ocr_service
from backend.services.thai_cleaner import clean_thai_text
from backend.services.chunker import chunk_document

router = APIRouter()

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a PDF or image, run OCR → clean → chunk → store pipeline."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Determine file type
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ("pdf", "png", "jpg", "jpeg", "webp", "tiff"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    file_bytes = await file.read()

    # Save file locally
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        f.write(file_bytes)

    # Create document record
    doc = Document(filename=file.filename, doc_type=ext, status="processing")
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    try:
        # Step 1: OCR
        if ext == "pdf":
            ocr_result = await ocr_service.extract_from_pdf(file_bytes)
        else:
            mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                        "webp": "image/webp", "tiff": "image/tiff"}
            ocr_result = await ocr_service.extract_from_image(file_bytes, mime_map.get(ext, "image/png"))

        text_blocks = ocr_result.get("text_blocks", [])
        tables = ocr_result.get("tables", [])

        # Step 2: Clean Thai text
        raw_text = "\n\n".join(text_blocks)
        cleaned_text = clean_thai_text(raw_text)
        doc.raw_text = cleaned_text

        # Step 3: Store structured table data
        for i, table in enumerate(tables):
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            for j, row in enumerate(rows):
                row_dict = dict(zip(headers, row)) if headers else {"data": row}
                sd = StructuredData(
                    document_id=doc.id,
                    table_name=f"{file.filename}_table_{i}",
                    headers=headers,
                    row_data=row_dict,
                    row_index=j,
                )
                db.add(sd)

        # Step 4: Semantic chunking + LLM summaries
        if cleaned_text.strip():
            chunk_results = await chunk_document(cleaned_text)
            for cr in chunk_results:
                chunk = Chunk(
                    document_id=doc.id,
                    chunk_index=cr.chunk_index,
                    chunk_text=cr.text,
                    summary=cr.summary,
                    token_count=cr.token_count,
                    embedding=cr.embedding,
                    metadata_={
                        "start_char": cr.start_char,
                        "end_char": cr.end_char,
                    },
                )
                db.add(chunk)

        doc.status = "completed"
        await db.commit()

        return {
            "id": doc.id,
            "filename": doc.filename,
            "status": "completed",
            "text_blocks": len(text_blocks),
            "tables_extracted": len(tables),
            "chunks_created": len(chunk_results) if cleaned_text.strip() else 0,
        }

    except Exception as e:
        doc.status = "failed"
        doc.error_message = str(e)
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("")
async def list_documents(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """List all documents with chunk/table counts."""
    result = await db.execute(
        select(Document).order_by(Document.created_at.desc()).offset(skip).limit(limit)
    )
    docs = result.scalars().all()

    items = []
    for doc in docs:
        # Count chunks
        chunk_count = await db.execute(
            select(func.count(Chunk.id)).where(Chunk.document_id == doc.id)
        )
        # Count structured rows
        table_count = await db.execute(
            select(func.count(StructuredData.id)).where(StructuredData.document_id == doc.id)
        )

        items.append({
            "id": doc.id,
            "filename": doc.filename,
            "source_url": doc.source_url,
            "doc_type": doc.doc_type,
            "status": doc.status,
            "chunk_count": chunk_count.scalar() or 0,
            "table_row_count": table_count.scalar() or 0,
            "created_at": doc.created_at.isoformat() if doc.created_at else None,
        })

    return {"documents": items, "total": len(items)}


@router.get("/{doc_id}")
async def get_document(doc_id: int, db: AsyncSession = Depends(get_db)):
    """Get document details with its chunks and tables."""
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Get chunks
    chunks_result = await db.execute(
        select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.chunk_index)
    )
    chunks = chunks_result.scalars().all()

    # Get structured data
    sd_result = await db.execute(
        select(StructuredData).where(StructuredData.document_id == doc_id).order_by(StructuredData.row_index)
    )
    structured = sd_result.scalars().all()

    return {
        "id": doc.id,
        "filename": doc.filename,
        "source_url": doc.source_url,
        "doc_type": doc.doc_type,
        "status": doc.status,
        "raw_text": doc.raw_text,
        "error_message": doc.error_message,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "chunks": [
            {
                "id": c.id,
                "chunk_index": c.chunk_index,
                "chunk_text": c.chunk_text,
                "summary": c.summary,
                "token_count": c.token_count,
                "metadata": c.metadata_,
            }
            for c in chunks
        ],
        "structured_data": [
            {
                "id": s.id,
                "table_name": s.table_name,
                "headers": s.headers,
                "row_data": s.row_data,
                "row_index": s.row_index,
            }
            for s in structured
        ],
    }


@router.delete("/{doc_id}")
async def delete_document(doc_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a document and all associated data."""
    result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    await db.delete(doc)
    await db.commit()
    return {"status": "deleted", "id": doc_id}
