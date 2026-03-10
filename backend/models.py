"""SQLAlchemy models for documents, chunks, and structured data."""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, JSON, Index
)
from sqlalchemy.orm import DeclarativeBase, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class Document(Base):
    """Represents an uploaded or scraped document."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(500), nullable=False)
    source_url = Column(String(2000), nullable=True)
    raw_text = Column(Text, nullable=True)
    doc_type = Column(String(50), default="pdf")  # pdf, image, html
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow,
                        onupdate=datetime.utcnow)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    structured_data = relationship("StructuredData", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    """A semantic chunk of a document with its embedding."""
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)  # LLM-generated summary
    token_count = Column(Integer, nullable=True)
    embedding = Column(Vector(768), nullable=True)  # nomic-embed-text dimension
    metadata_ = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
    )


class StructuredData(Base):
    """Long-format table data extracted from documents for Text-to-SQL."""
    __tablename__ = "structured_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    table_name = Column(String(500), nullable=True)
    headers = Column(JSON, nullable=True)   # List of column names
    row_data = Column(JSON, nullable=False)  # JSONB row in long format
    row_index = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="structured_data")

    __table_args__ = (
        Index("ix_structured_data_document_id", "document_id"),
    )
