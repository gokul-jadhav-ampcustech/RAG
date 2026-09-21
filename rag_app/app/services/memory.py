"""
Memory Management Service
=========================

Handles different types of AI memory:
- Short-term Memory (STM): Working memory for current session
- Long-term Memory (LTM): Important information stored permanently
- Memory types: Episodic, Semantic, Procedural, Emotional, Preference
"""
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_

from app.db.memory_models import (
    Memory, MemoryType, MemoryImportance, MemorySession, 
    MemoryAssociation, MemoryVector, User
)
from app.services import embeddings

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages all types of AI memory for personalized experiences."""
    
    def __init__(self, db: Session, user_id: int, session: Optional[MemorySession] = None):
        self.db = db
        self.user_id = user_id
        self.session = session
    
    # ==================== MEMORY CREATION ====================
    
    def store_memory(
        self,
        title: str,
        content: str,
        memory_type: MemoryType,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        keywords: List[str] = None,
        conversation_id: int = None,
        is_pinned: bool = False
    ) -> Memory:
        """Store a new memory."""
        
        # Create memory record
        memory = Memory(
            user_id=self.user_id,
            session_id=self.session.id if self.session else None,
            memory_type=memory_type,
            importance=importance,
            title=title,
            content=content,
            keywords=json.dumps(keywords or []),
            conversation_id=conversation_id,
            is_pinned=is_pinned,
            created_at=datetime.utcnow(),
            event_date=datetime.utcnow()
        )
        
        self.db.add(memory)
        self.db.commit()
        self.db.refresh(memory)
        
        # Create vector embedding for semantic search
        self._create_memory_embedding(memory)
        
        logger.info(f"Stored {memory_type.value} memory: {title[:50]}...")
        return memory
    
    def store_working_memory(self, content: str, context: str = None) -> Memory:
        """Store working memory for current conversation."""
        title = f"Working Memory - {datetime.now().strftime('%H:%M')}"
        return self.store_memory(
            title=title,
            content=content,
            memory_type=MemoryType.WORKING,
            importance=MemoryImportance.LOW
        )
    
    def store_episodic_memory(self, event: str, context: str, importance: MemoryImportance = MemoryImportance.MEDIUM) -> Memory:
        """Store episodic memory (specific experiences)."""
        return self.store_memory(
            title=f"Experience: {event[:50]}",
            content=f"Event: {event}\nContext: {context}",
            memory_type=MemoryType.EPISODIC,
            importance=importance
        )
    
    def store_semantic_memory(self, fact: str, domain: str, importance: MemoryImportance = MemoryImportance.HIGH) -> Memory:
        """Store semantic memory (learned facts)."""
        return self.store_memory(
            title=f"Fact: {domain}",
            content=fact,
            memory_type=MemoryType.SEMANTIC,
            importance=importance,
            keywords=[domain]
        )
    
    def store_preference_memory(self, preference_type: str, value: str) -> Memory:
        """Store user preferences."""
        return self.store_memory(
            title=f"Preference: {preference_type}",
            content=f"{preference_type}: {value}",
            memory_type=MemoryType.PREFERENCE,
            importance=MemoryImportance.HIGH,
            keywords=[preference_type]
        )
    
    # ==================== MEMORY RETRIEVAL ====================
    
    def get_working_memory(self, limit: int = 10) -> List[Memory]:
        """Get recent working memory for current session."""
        if not self.session:
            return []
        
        return self.db.query(Memory).filter(
            Memory.session_id == self.session.id,
            Memory.memory_type == MemoryType.WORKING
        ).order_by(desc(Memory.created_at)).limit(limit).all()
    
    def get_long_term_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        importance: Optional[MemoryImportance] = None,
        limit: int = 50
    ) -> List[Memory]:
        """Get long-term memories with optional filtering."""
        query = self.db.query(Memory).filter(Memory.user_id == self.user_id)
        
        # Exclude working memory for LTM
        query = query.filter(Memory.memory_type != MemoryType.WORKING)
        
        if memory_type:
            query = query.filter(Memory.memory_type == memory_type)
        
        if importance:
            query = query.filter(Memory.importance >= importance.value)
        
        return query.order_by(desc(Memory.importance), desc(Memory.last_accessed), desc(Memory.created_at)).limit(limit).all()
    
    def get_pinned_memories(self) -> List[Memory]:
        """Get user-pinned important memories."""
        return self.db.query(Memory).filter(
            Memory.user_id == self.user_id,
            Memory.is_pinned == True
        ).order_by(desc(Memory.created_at)).all()
    
    def search_memories(
        self,
        query: str,
        memory_types: List[MemoryType] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search memories using semantic similarity."""
        try:
            # Get query embedding
            query_embedding = embeddings.embed_text(query)
            
            # Find similar memories using vector search
            # This is a simplified version - in production, use proper vector DB
            memories = self.db.query(Memory).filter(Memory.user_id == self.user_id)
            
            if memory_types:
                memories = memories.filter(Memory.memory_type.in_(memory_types))
            
            memories = memories.all()
            
            # Calculate similarity scores (simplified)
            results = []
            for memory in memories:
                # Simplified similarity calculation
                content_lower = memory.content.lower()
                query_lower = query.lower()
                
                # Simple keyword matching score
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                similarity = len(query_words.intersection(content_words)) / len(query_words.union(content_words))
                
                if similarity > 0.1:  # Minimum similarity threshold
                    results.append({
                        'memory': memory,
                        'similarity': similarity,
                        'relevance_score': similarity * memory.importance.value
                    })
            
            # Sort by relevance score
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    # ==================== MEMORY MANAGEMENT ====================
    
    def pin_memory(self, memory_id: int) -> bool:
        """Pin a memory as important."""
        memory = self.db.query(Memory).filter(
            Memory.id == memory_id,
            Memory.user_id == self.user_id
        ).first()
        
        if memory:
            memory.is_pinned = True
            memory.importance = MemoryImportance.CRITICAL
            self.db.commit()
            return True
        
        return False
    
    def unpin_memory(self, memory_id: int) -> bool:
        """Unpin a memory."""
        memory = self.db.query(Memory).filter(
            Memory.id == memory_id,
            Memory.user_id == self.user_id
        ).first()
        
        if memory:
            memory.is_pinned = False
            self.db.commit()
            return True
        
        return False
    
    def update_memory_access(self, memory_id: int):
        """Update memory access statistics."""
        memory = self.db.query(Memory).filter(Memory.id == memory_id).first()
        if memory:
            memory.access_count += 1
            memory.last_accessed = datetime.utcnow()
            self.db.commit()
    
    def consolidate_working_memory(self) -> List[Memory]:
        """Consolidate important working memories into long-term memory."""
        if not self.session:
            return []
        
        working_memories = self.get_working_memory(limit=50)
        consolidated = []
        
        # Simple consolidation: convert high-access working memories to episodic
        for memory in working_memories:
            if memory.access_count >= 3:  # Threshold for consolidation
                consolidated_memory = self.store_episodic_memory(
                    event=f"Consolidated from session: {memory.title}",
                    context=memory.content,
                    importance=MemoryImportance.MEDIUM
                )
                consolidated.append(consolidated_memory)
        
        return consolidated
    
    def cleanup_old_working_memory(self, days_old: int = 1):
        """Clean up old working memory entries."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        self.db.query(Memory).filter(
            Memory.user_id == self.user_id,
            Memory.memory_type == MemoryType.WORKING,
            Memory.created_at < cutoff_date,
            Memory.is_pinned == False
        ).delete()
        
        self.db.commit()
    
    # ==================== HELPER METHODS ====================
    
    def _create_memory_embedding(self, memory: Memory):
        """Create vector embedding for memory content."""
        try:
            embedding_vector = embeddings.embed_text(memory.content)
            
            memory_vector = MemoryVector(
                memory_id=memory.id,
                embedding=json.dumps(embedding_vector),
                model_version=embeddings.get_model_name()
            )
            
            self.db.add(memory_vector)
            self.db.commit()
            
        except Exception as e:
            logger.warning(f"Failed to create memory embedding: {e}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of user's memory statistics."""
        total_memories = self.db.query(Memory).filter(Memory.user_id == self.user_id).count()
        
        memory_counts = {}
        for mem_type in MemoryType:
            count = self.db.query(Memory).filter(
                Memory.user_id == self.user_id,
                Memory.memory_type == mem_type
            ).count()
            memory_counts[mem_type.value] = count
        
        pinned_count = self.db.query(Memory).filter(
            Memory.user_id == self.user_id,
            Memory.is_pinned == True
        ).count()
        
        return {
            'total_memories': total_memories,
            'memory_by_type': memory_counts,
            'pinned_memories': pinned_count,
            'session_active': self.session is not None
        }