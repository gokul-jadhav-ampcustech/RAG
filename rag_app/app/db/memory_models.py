"""
Memory System Database Models
============================

Implements different types of memory for personalized AI experiences:
- User authentication
- Short-term memory (STM) - session-based
- Long-term memory (LTM) - persistent important information
- Different memory types (episodic, semantic, procedural)
"""
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import enum

from app.db.models import Base


class MemoryType(enum.Enum):
    """Types of memory for different cognitive functions."""
    WORKING = "working"          # Current conversation context
    EPISODIC = "episodic"        # Specific experiences and events
    SEMANTIC = "semantic"        # Facts and knowledge
    PROCEDURAL = "procedural"    # Skills and patterns
    EMOTIONAL = "emotional"      # Emotional associations
    PREFERENCE = "preference"    # User preferences and settings


class MemoryImportance(enum.Enum):
    """Memory importance levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class User(Base):
    """User authentication and profile."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user")
    memories = relationship("Memory", back_populates="user")
    memory_sessions = relationship("MemorySession", back_populates="user")


class MemorySession(Base):
    """Short-term memory session for temporary context."""
    __tablename__ = "memory_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(String(100), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Session metadata
    context_summary = Column(Text, nullable=True)
    topic_tags = Column(String(500), nullable=True)  # JSON string
    
    # Relationships
    user = relationship("User", back_populates="memory_sessions")
    working_memories = relationship("Memory", 
                                  foreign_keys="Memory.session_id",
                                  back_populates="session")


class Memory(Base):
    """Individual memory entries with different types and importance."""
    __tablename__ = "memories"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(Integer, ForeignKey("memory_sessions.id"), nullable=True)
    
    # Memory classification
    memory_type = Column(Enum(MemoryType), nullable=False)
    importance = Column(Enum(MemoryImportance), default=MemoryImportance.MEDIUM)
    
    # Memory content
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    keywords = Column(String(500), nullable=True)  # JSON string
    
    # Context and relationships
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    related_document = Column(String(200), nullable=True)
    source_message_id = Column(Integer, nullable=True)
    
    # Memory metadata
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, nullable=True)
    decay_factor = Column(Float, default=1.0)  # For memory decay simulation
    
    # Temporal information
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    event_date = Column(DateTime, nullable=True)  # When the remembered event occurred
    
    # Memory flags
    is_pinned = Column(Boolean, default=False)  # User-pinned important memories
    is_verified = Column(Boolean, default=False)  # Verified accuracy
    is_private = Column(Boolean, default=True)  # Privacy setting
    
    # Relationships
    user = relationship("User", back_populates="memories")
    conversation = relationship("Conversation")
    session = relationship("MemorySession", 
                          foreign_keys=[session_id],
                          back_populates="working_memories")


class MemoryAssociation(Base):
    """Associations between different memories for graph-like connections."""
    __tablename__ = "memory_associations"
    
    id = Column(Integer, primary_key=True, index=True)
    source_memory_id = Column(Integer, ForeignKey("memories.id"), nullable=False)
    target_memory_id = Column(Integer, ForeignKey("memories.id"), nullable=False)
    
    # Association metadata
    association_type = Column(String(50), nullable=False)  # "relates_to", "contradicts", "confirms"
    strength = Column(Float, default=0.5)  # 0.0 to 1.0
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    source_memory = relationship("Memory", foreign_keys=[source_memory_id])
    target_memory = relationship("Memory", foreign_keys=[target_memory_id])


class MemoryVector(Base):
    """Vector embeddings for memory content for semantic search."""
    __tablename__ = "memory_vectors"
    
    id = Column(Integer, primary_key=True, index=True)
    memory_id = Column(Integer, ForeignKey("memories.id"), nullable=False)
    embedding = Column(Text, nullable=False)  # JSON-encoded vector
    model_version = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    memory = relationship("Memory")