"""
Simple, understandable word-based chunking with overlap.

Deliberately NOT semantic/sentence-aware chunking - that's a good upgrade
for later, but a beginner-friendly fixed-size sliding window is easiest to
reason about while learning the RAG fundamentals.
"""
from app.config import get_settings

settings = get_settings()


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Split `text` into overlapping chunks of `chunk_size` words, moving
    forward by (chunk_size - chunk_overlap) words each step.

    Example: chunk_size=700, overlap=100 -> each chunk shares its last
    100 words with the start of the next chunk, so context isn't lost
    at chunk boundaries.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - chunk_overlap
    chunks = []

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break

    return chunks
