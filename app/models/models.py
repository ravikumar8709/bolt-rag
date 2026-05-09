from sqlalchemy import Column, Integer, Text, ForeignKey, DateTime, String,Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db import Base

class PDFMetadata(Base):
    __tablename__ = "pdf_metadata"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    path = Column(String, nullable=True)
    source = Column(String)
    total_pages = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    embedding_dim = Column(Integer, nullable=False, default=384)

    # Two separate relationships
    pdf_chunks = relationship("PDFChunk", back_populates="pdf")
    document_chunks = relationship("DocumentChunk", back_populates="pdf_metadata")


class PDFChunk(Base):
    __tablename__ = "pdf_chunks"

    id = Column(Integer, primary_key=True, index=True)
    pdf_id = Column(Integer, ForeignKey("pdf_metadata.id"))
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    embedding_dim = Column(Integer, nullable=False, default=384)
    created_at = Column(DateTime, default=datetime.utcnow)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    pdf = relationship("PDFMetadata", back_populates="pdf_chunks")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, index=True)

    source = Column(String, nullable=False)
    text = Column(String, nullable=False)
    page_number = Column(Integer, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    embedding_dim = Column(Integer, nullable=False, default=384)
    pdf_metadata_id = Column(Integer, ForeignKey("pdf_metadata.id"))

    pdf_metadata = relationship("PDFMetadata", back_populates="document_chunks")


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    question = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    status = Column(Boolean, nullable=False)
    comments = Column(String, default="", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)