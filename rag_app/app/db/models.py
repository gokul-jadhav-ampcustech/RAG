"""
ORM models for the RAG pipeline.
"""
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, Boolean, Index
from sqlalchemy.orm import relationship

from app.config import get_settings
from app.db.database import Base

settings = get_settings()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(settings.embedding_dimension), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("Document", back_populates="chunks")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    title = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )
    user = relationship("User", back_populates="conversations")


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")


class LongTermMemory(Base):
    """
    Production-ready long-term memory with importance scoring,
    deduplication support, conflict tracking, and user isolation.
    """
    __tablename__ = "long_term_memories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="SET NULL"), nullable=True)
    source_message_id = Column(Integer, nullable=True)

    # Classification
    memory_type = Column(String(50), nullable=False, index=True)
    # preference | project | goal | technical_context | decision | fact | instruction | profile

    # Content
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)

    # Scoring
    importance_score = Column(Float, default=0.5, nullable=False)
    confidence_score = Column(Float, default=0.5, nullable=False)

    # Vector for semantic search
    embedding = Column(Vector(settings.embedding_dimension), nullable=True)

    # Lifecycle
    status = Column(String(20), default="active", nullable=False, index=True)
    # active | superseded | archived | deleted
    superseded_by_id = Column(Integer, ForeignKey("long_term_memories.id"), nullable=True)

    # Usage tracking
    access_count = Column(Integer, default=0, nullable=False)
    last_accessed_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=True)

    # Metadata (JSON string)
    meta = Column(Text, nullable=True)

    user = relationship("User", foreign_keys=[user_id])
    superseded_by = relationship("LongTermMemory", remote_side=[id], foreign_keys=[superseded_by_id])

    __table_args__ = (
        Index("ix_ltm_user_status", "user_id", "status"),
        Index("ix_ltm_user_type", "user_id", "memory_type"),
    )


class HITLPending(Base):
    """
    Stores pending Human-in-the-Loop web search approval requests.
    Expires after HITL_PENDING_TTL_MINUTES.
    """
    __tablename__ = "hitl_pending"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    pending_query = Column(Text, nullable=False)
    memory_context = Column(Text, nullable=True)
    status = Column(String(20), default="pending", nullable=False)  # pending | resolved | expired
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)

    user = relationship("User", foreign_keys=[user_id])
    conversation = relationship("Conversation", foreign_keys=[conversation_id])

