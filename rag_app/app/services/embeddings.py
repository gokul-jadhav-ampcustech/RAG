"""
Local embedding generation via sentence-transformers.

We use a local model (no paid API) so the embedding step never depends on
external network availability or cost. The model is loaded once and reused
(singleton pattern) since loading it is relatively expensive.
"""
import logging

from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

_model: SentenceTransformer | None = None


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s'...", settings.embedding_model)
        try:
            _model = SentenceTransformer(settings.embedding_model)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load embedding model: %s", exc)
            raise EmbeddingError(
                f"Could not load embedding model '{settings.embedding_model}'."
            ) from exc
    return _model


def embed_text(text: str) -> list[float]:
    """Embed a single string (e.g. a user question)."""
    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings (e.g. document chunks)."""
    if not texts:
        return []
    model = get_embedding_model()
    try:
        vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    except Exception as exc:  # noqa: BLE001
        logger.error("Embedding generation failed: %s", exc)
        raise EmbeddingError("Failed to generate embeddings for the given text.") from exc
    return [v.tolist() for v in vectors]


def get_model_name() -> str:
    return settings.embedding_model
