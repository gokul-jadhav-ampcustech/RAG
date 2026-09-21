"""
Memory Schemas
==============

Pydantic models for memory operations.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from app.db.memory_models import MemoryType, MemoryImportance, Memory


class MemoryCreate(BaseModel):
    title: str = Field(..., max_length=200)
    content: str = Field(..., max_length=10000)
    memory_type: MemoryType
    importance: MemoryImportance = MemoryImportance.MEDIUM
    keywords: Optional[List[str]] = []
    conversation_id: Optional[int] = None
    is_pinned: bool = False


class MemoryResponse(BaseModel):
    id: int
    title: str
    content: str
    memory_type: MemoryType
    importance: MemoryImportance
    keywords: List[str]
    conversation_id: Optional[int]
    access_count: int
    last_accessed: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    is_pinned: bool
    is_verified: bool

    @classmethod
    def from_memory(cls, memory: Memory) -> 'MemoryResponse':
        """Create response from Memory model."""
        import json
        
        keywords = []
        if memory.keywords:
            try:
                keywords = json.loads(memory.keywords)
            except:
                keywords = []
        
        return cls(
            id=memory.id,
            title=memory.title,
            content=memory.content,
            memory_type=memory.memory_type,
            importance=memory.importance,
            keywords=keywords,
            conversation_id=memory.conversation_id,
            access_count=memory.access_count,
            last_accessed=memory.last_accessed,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
            is_pinned=memory.is_pinned,
            is_verified=memory.is_verified
        )

    class Config:
        from_attributes = True


class MemorySearch(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    memory_types: Optional[List[MemoryType]] = None
    limit: int = Field(default=20, le=50)


class MemorySearchResult(BaseModel):
    memory: MemoryResponse
    similarity: float
    relevance_score: float


class MemorySummary(BaseModel):
    total_memories: int
    memory_by_type: Dict[str, int]
    pinned_memories: int
    session_active: bool


class PinMemoryRequest(BaseModel):
    memory_id: int