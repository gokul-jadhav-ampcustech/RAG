"""
Memory Retriever
================
Retrieves only relevant long-term memories for a given query.
Uses semantic similarity + importance + recency weighting.
Strict user isolation enforced.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Tuple

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import LongTermMemory
from app.memory.deduplicator import cosine_similarity

logger = logging.getLogger(__name__)
settings = get_settings()


def retrieve_relevant(
    db: Session,
    user_id: int,
    query_embedding: List[float],
    top_k: int = None,
    min_importance: float = None,
) -> List[LongTermMemory]:
    """
    Retrieve top-k relevant memories for a user query.

    Ranking formula:
        final_score = 0.60 * semantic_sim
                    + 0.25 * importance_score
                    + 0.10 * recency_score
                    + 0.05 * (access_count_normalized)

    Only returns memories with status='active' for this user.
    """
    top_k = top_k or settings.memory_top_k
    min_importance = min_importance or 0.30  # don't retrieve very low importance

    memories = (
        db.query(LongTermMemory)
        .filter(
            LongTermMemory.user_id == user_id,
            LongTermMemory.status == "active",
            LongTermMemory.importance_score >= min_importance,
        )
        .all()
    )

    if not memories:
        return []

    now = datetime.utcnow()
    scored: List[Tuple[float, LongTermMemory]] = []

    for mem in memories:
        # Semantic similarity
        if mem.embedding is not None:
            try:
                mem_vec = list(mem.embedding) if not isinstance(mem.embedding, list) else mem.embedding
                sem_sim = cosine_similarity(query_embedding, mem_vec)
            except Exception:
                sem_sim = 0.0
        else:
            sem_sim = 0.0

        # Skip if below similarity threshold
        if sem_sim < settings.memory_similarity_threshold * 0.5:
            continue

        # Recency score: 1.0 if created today, decays over 90 days
        age_days = (now - mem.created_at).days
        recency = max(0.0, 1.0 - age_days / 90.0)

        # Access count normalized (cap at 20)
        access_norm = min(mem.access_count, 20) / 20.0

        final_score = (
            0.60 * sem_sim
            + 0.25 * mem.importance_score
            + 0.10 * recency
            + 0.05 * access_norm
        )

        scored.append((final_score, mem))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [mem for _, mem in scored[:top_k]]

    # Update access tracking
    for mem in top:
        mem.access_count += 1
        mem.last_accessed_at = now
    if top:
        db.commit()

    logger.info("[MEMORY] Retrieved %d relevant memories for user %d", len(top), user_id)
    return top


def format_for_context(memories: List[LongTermMemory]) -> str:
    """Format retrieved memories as context string for LLM."""
    if not memories:
        return ""
    lines = ["=== RELEVANT USER MEMORY ==="]
    for mem in memories:
        lines.append(f"[{mem.memory_type.upper()}] {mem.content}")
    lines.append("=== END MEMORY ===")
    return "\n".join(lines)
