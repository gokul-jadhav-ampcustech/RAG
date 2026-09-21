"""
Memory Pipeline
===============
Orchestrates the full automatic memory lifecycle:
  User message → Extract candidates → Score → Deduplicate → Store

Designed to run after the main response is returned (non-blocking).
A failure here must NEVER affect the chat response.
"""
import logging
from typing import List, Optional

from sqlalchemy.orm import Session

from app.config import get_settings
from app.memory.extractor import MemoryCandidate, extract_candidates, is_low_value
from app.memory import deduplicator
from app.services.embeddings import embed_text

logger = logging.getLogger(__name__)
settings = get_settings()


def _llm_call(messages: list) -> str:
    """Thin wrapper around the Groq LLM for memory extraction."""
    from app.services.llm import _call_groq
    return _call_groq(messages, temperature=0.1, max_tokens=600)


def run_memory_pipeline(
    db: Session,
    user_id: int,
    user_message: str,
    conversation_id: Optional[int],
    source_message_id: Optional[int],
    conversation_context: List[dict],
) -> int:
    """
    Full memory pipeline. Returns number of memories stored.
    Safe to call in background — all exceptions are caught.
    """
    stored = 0
    try:
        if is_low_value(user_message):
            logger.info("[MEMORY] Skipping low-value message")
            return 0

        candidates: List[MemoryCandidate] = extract_candidates(
            user_message=user_message,
            conversation_context=conversation_context,
            llm_call_fn=_llm_call,
        )

        for candidate in candidates:
            if not candidate.should_store:
                continue
            if candidate.importance_score < settings.memory_importance_threshold:
                logger.info(
                    "[MEMORY] Candidate below importance threshold (%.2f < %.2f): %r",
                    candidate.importance_score,
                    settings.memory_importance_threshold,
                    candidate.content[:60],
                )
                continue

            logger.info("[MEMORY] Candidate detected: type=%s importance=%.2f content=%r",
                        candidate.memory_type, candidate.importance_score, candidate.content[:60])

            try:
                embedding = embed_text(candidate.content)
            except Exception as e:
                logger.error("[MEMORY] Embedding failed for candidate: %s", e)
                continue

            logger.info("[MEMORY] Duplicate check started")
            result = deduplicator.resolve(
                db=db,
                user_id=user_id,
                new_content=candidate.content,
                new_embedding=embedding,
                memory_type=candidate.memory_type,
                importance_score=candidate.importance_score,
                confidence_score=candidate.confidence_score,
                conversation_id=conversation_id,
                source_message_id=source_message_id,
            )
            logger.info("[MEMORY] Duplicate check completed")

            if result:
                stored += 1
                logger.info("[MEMORY] Stored memory id=%d", result.id)

    except Exception as e:
        logger.error("[MEMORY] Pipeline error (non-fatal): %s", e)

    return stored
