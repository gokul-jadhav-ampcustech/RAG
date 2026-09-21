"""
Retrieval: given a question embedding, find the most relevant chunks in
PostgreSQL using pgvector cosine distance, and apply a relevance threshold.
"""
import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Document, DocumentChunk

logger = logging.getLogger(__name__)

settings = get_settings()


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    chunk_text: str
    similarity: float  # 1.0 = identical, 0.0 = unrelated (cosine similarity)


def retrieve_relevant_chunks(
    db: Session,
    query_embedding: list[float],
    top_k: int | None = None,
    similarity_threshold: float | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve the top_k most similar chunks and keep only those at or above
    similarity_threshold. pgvector's `<=>` operator returns COSINE DISTANCE
    (0 = identical, 2 = opposite), so similarity = 1 - distance.
    """
    top_k = top_k or settings.top_k
    similarity_threshold = (
        similarity_threshold if similarity_threshold is not None else settings.similarity_threshold
    )

    # cosine_distance is provided by pgvector's SQLAlchemy comparator (Vector type)
    distance_expr = DocumentChunk.embedding.cosine_distance(query_embedding)

    results = (
        db.query(DocumentChunk, Document.filename, distance_expr.label("distance"))
        .join(Document, Document.id == DocumentChunk.document_id)
        .order_by(distance_expr)
        .limit(top_k)
        .all()
    )

    retrieved: list[RetrievedChunk] = []
    for chunk, filename, distance in results:
        similarity = 1 - float(distance)
        logger.info(
            "  candidate: chunk %d (idx %d) similarity=%.4f",
            chunk.id,
            chunk.chunk_index,
            similarity,
        )
        if similarity >= similarity_threshold:
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    filename=filename,
                    chunk_index=chunk.chunk_index,
                    chunk_text=chunk.chunk_text,
                    similarity=round(similarity, 4),
                )
            )

    logger.info(
        "Retrieved %d/%d candidate chunks above similarity threshold %.2f",
        len(retrieved),
        len(results),
        similarity_threshold,
    )
    return retrieved