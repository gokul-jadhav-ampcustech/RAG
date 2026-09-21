"""
Memory Deduplication and Conflict Resolution
=============================================
Before storing a new memory:
1. Search existing active memories by embedding similarity
2. Detect duplicates (high similarity, same type)
3. Detect conflicts (contradictory content)
4. Update/merge or supersede as appropriate
"""
import json
import logging
import re
from datetime import datetime
from typing import Optional, List, Tuple

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import LongTermMemory

logger = logging.getLogger(__name__)
settings = get_settings()

DEDUP_THRESHOLD = settings.memory_dedup_threshold  # 0.85 — very similar = duplicate
CONFLICT_THRESHOLD = 0.60  # moderately similar but different content = potential conflict

# Keywords that signal a superseding update
_UPDATE_SIGNALS = re.compile(
    r"\b(migrated?|switched?|changed?|updated?|now\s+use|replaced?|moved?\s+to|"
    r"no\s+longer|instead\s+of|switched?\s+from)\b",
    re.IGNORECASE,
)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def find_similar_memories(
    db: Session,
    user_id: int,
    embedding: List[float],
    memory_type: str,
    top_k: int = 5,
) -> List[Tuple[LongTermMemory, float]]:
    """
    Find existing active memories for this user that are semantically similar.
    Returns list of (memory, similarity_score) sorted by similarity desc.
    """
    existing = (
        db.query(LongTermMemory)
        .filter(
            LongTermMemory.user_id == user_id,
            LongTermMemory.status == "active",
        )
        .all()
    )

    scored = []
    for mem in existing:
        if mem.embedding is None:
            continue
        try:
            mem_vec = list(mem.embedding) if not isinstance(mem.embedding, list) else mem.embedding
            sim = cosine_similarity(embedding, mem_vec)
            scored.append((mem, sim))
        except Exception:
            continue

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def resolve(
    db: Session,
    user_id: int,
    new_content: str,
    new_embedding: List[float],
    memory_type: str,
    importance_score: float,
    confidence_score: float,
    conversation_id: Optional[int],
    source_message_id: Optional[int],
) -> Optional[LongTermMemory]:
    """
    Main deduplication + conflict resolution entry point.

    Returns:
        The stored/updated LongTermMemory, or None if skipped as duplicate.
    """
    similar = find_similar_memories(db, user_id, new_embedding, memory_type)

    for existing_mem, sim in similar:
        if sim >= DEDUP_THRESHOLD:
            # Very high similarity — treat as duplicate, just update metadata
            logger.info(
                "[MEMORY] Duplicate detected (sim=%.3f) — updating existing memory %d",
                sim, existing_mem.id,
            )
            existing_mem.access_count += 1
            existing_mem.last_accessed_at = datetime.utcnow()
            # Boost confidence slightly
            existing_mem.confidence_score = min(
                1.0, existing_mem.confidence_score + 0.05
            )
            db.commit()
            return existing_mem

        if sim >= CONFLICT_THRESHOLD and existing_mem.memory_type == memory_type:
            # Moderate similarity with same type — check for conflict/update
            is_update = bool(_UPDATE_SIGNALS.search(new_content))
            if is_update or importance_score > existing_mem.importance_score:
                logger.info(
                    "[MEMORY] Conflict/update detected (sim=%.3f) — superseding memory %d",
                    sim, existing_mem.id,
                )
                # Mark old as superseded
                existing_mem.status = "superseded"
                existing_mem.updated_at = datetime.utcnow()
                db.flush()

                # Store new memory with reference to superseded
                new_mem = _create_memory(
                    db, user_id, new_content, new_embedding, memory_type,
                    importance_score, confidence_score, conversation_id,
                    source_message_id,
                )
                new_mem.superseded_by_id = None  # new one is active
                existing_mem.superseded_by_id = new_mem.id
                db.commit()
                logger.info("[MEMORY] Stored superseding memory %d", new_mem.id)
                return new_mem

    # No duplicate or conflict — store fresh
    new_mem = _create_memory(
        db, user_id, new_content, new_embedding, memory_type,
        importance_score, confidence_score, conversation_id, source_message_id,
    )
    db.commit()
    logger.info("[MEMORY] Stored new memory %d (type=%s, importance=%.2f)",
                new_mem.id, memory_type, importance_score)
    return new_mem


def _create_memory(
    db: Session,
    user_id: int,
    content: str,
    embedding: List[float],
    memory_type: str,
    importance_score: float,
    confidence_score: float,
    conversation_id: Optional[int],
    source_message_id: Optional[int],
) -> LongTermMemory:
    mem = LongTermMemory(
        user_id=user_id,
        conversation_id=conversation_id,
        source_message_id=source_message_id,
        memory_type=memory_type,
        content=content,
        importance_score=importance_score,
        confidence_score=confidence_score,
        embedding=embedding,
        status="active",
        access_count=0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(mem)
    db.flush()
    return mem
