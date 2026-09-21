"""
Memory API — Long-Term Memory CRUD

GET    /memory/          -> list active memories for current user
DELETE /memory/{id}      -> delete a specific memory
DELETE /memory/          -> delete all memories for current user

The old mem0 explicit-add endpoint is removed; memory is now automatic.
The mem0 panel in the frontend is replaced by this API.
"""
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.memory_models import User
from app.db.models import LongTermMemory
from app.api.auth import get_current_user_from_token

logger = logging.getLogger(__name__)
router = APIRouter(tags=["memory"])


@router.get("/")
def list_memories(
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Get all active long-term memories for the current user."""
    memories = (
        db.query(LongTermMemory)
        .filter(
            LongTermMemory.user_id == current_user.id,
            LongTermMemory.status == "active",
        )
        .order_by(LongTermMemory.importance_score.desc(), LongTermMemory.created_at.desc())
        .all()
    )
    return {
        "memories": [
            {
                "id": m.id,
                "memory": m.content,
                "memory_type": m.memory_type,
                "importance_score": round(m.importance_score, 2),
                "confidence_score": round(m.confidence_score, 2),
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "access_count": m.access_count,
            }
            for m in memories
        ],
        "total": len(memories),
    }


@router.delete("/{memory_id}")
def delete_memory(
    memory_id: int,
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Delete a specific memory (user-scoped)."""
    mem = db.query(LongTermMemory).filter(
        LongTermMemory.id == memory_id,
        LongTermMemory.user_id == current_user.id,
    ).first()
    if not mem:
        raise HTTPException(status_code=404, detail="Memory not found")
    mem.status = "deleted"
    db.commit()
    logger.info("[MEMORY] Deleted memory %d for user %d", memory_id, current_user.id)
    return {"message": "Memory deleted"}


@router.delete("/")
def delete_all_memories(
    current_user: User = Depends(get_current_user_from_token),
    db: Session = Depends(get_db)
):
    """Delete all memories for the current user."""
    db.query(LongTermMemory).filter(
        LongTermMemory.user_id == current_user.id,
    ).update({"status": "deleted"})
    db.commit()
    logger.info("[MEMORY] Deleted all memories for user %d", current_user.id)
    return {"message": "All memories deleted"}
