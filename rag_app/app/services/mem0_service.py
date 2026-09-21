"""
Mem0-based Memory Service (mem0ai 2.x)
=======================================

Uses mem0ai with:
- Groq LLM for memory extraction
- App's own sentence-transformers embedder (already loaded by the RAG pipeline)
- Qdrant on-disk vector store for persistence

The custom embedder reuses the already-loaded SentenceTransformer model,
avoiding any re-import of sentence_transformers (which is already imported
by the time this service is called in the running server).
"""
import logging
import re
from typing import List, Optional, Dict, Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

MEMORY_TRIGGERS = re.compile(
    r"^(remember\s+(that\s+)?|please\s+remember\s+(that\s+)?"
    r"|keep\s+in\s+mind\s+(that\s+)?|note\s+(that\s+)?"
    r"|don'?t\s+forget\s+(that\s+)?|save\s+(that\s+)?)",
    re.IGNORECASE,
)

_mem0_client = None


def _get_mem0_client():
    """Get or create mem0 client using app's own embedder."""
    global _mem0_client
    if _mem0_client is not None:
        return _mem0_client

    from mem0 import Memory
    from mem0.embeddings.base import EmbeddingBase
    from mem0.configs.embeddings.base import BaseEmbedderConfig

    class AppEmbedder(EmbeddingBase):
        """Wraps the app's already-loaded SentenceTransformer model."""

        def __init__(self, config: Optional[BaseEmbedderConfig] = None):
            # Don't call super().__init__ to avoid any provider loading
            self.config = config or BaseEmbedderConfig()
            self.config.embedding_dims = 384

        def embed(self, text, memory_action=None):
            from app.services.embeddings import embed_text
            return embed_text(text)

        def embed_batch(self, texts, memory_action=None):
            from app.services.embeddings import embed_texts
            return embed_texts(texts)

    config = {0
        "llm": {
            "provider": "groq",
            "config": {
                "model": "openai/gpt-oss-20b",  # smaller model for mem0 extraction (lower token usage)
                "api_key": settings.groq_api_key,
                "temperature": 0.1,
                "max_tokens": 500,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "smartrag_memories",
                "embedding_model_dims": 384,
                "path": "./mem0_store",
            },
        },
        "embedder": {
            "provider": "openai",   # placeholder — overridden below
            "config": {
                "model": "text-embedding-3-small",
                "api_key": "placeholder",
            },
        },
    }

    try:
        mem = Memory.from_config(config)
        # Override the embedding model with our custom one
        mem.embedding_model = AppEmbedder()
        _mem0_client = mem
        logger.info("mem0 initialized with app embedder + on-disk Qdrant")
        return _mem0_client
    except Exception as e:
        logger.error("mem0 initialization failed: %s", e)
        raise RuntimeError(f"Could not initialize mem0: {e}") from e


def is_memory_request(text: str) -> bool:
    return bool(MEMORY_TRIGGERS.match(text.strip()))


def extract_memory_content(text: str) -> str:
    cleaned = MEMORY_TRIGGERS.sub("", text.strip()).strip()
    return cleaned.rstrip(".,!?").strip()


def add_memory(user_id: str, content: str, metadata: Dict[str, Any] = None) -> List[Dict]:
    """Add a memory for a user. Uses infer=False to skip LLM extraction and store directly."""
    try:
        client = _get_mem0_client()
        result = client.add(content, user_id=user_id, metadata=metadata or {}, infer=False)
        logger.info("Added memory for user %s: %s", user_id, content[:60])
        if isinstance(result, dict):
            return result.get("results", [])
        return result if isinstance(result, list) else []
    except Exception as e:
        logger.error("Failed to add memory for user %s: %s", user_id, e)
        return []


def get_relevant_memories(user_id: str, query: str, limit: int = 5) -> List[Dict]:
    try:
        client = _get_mem0_client()
        results = client.search(query, filters={"user_id": user_id}, top_k=limit)
        memories = results.get("results", []) if isinstance(results, dict) else results
        logger.info("Retrieved %d memories for user %s", len(memories), user_id)
        return memories
    except Exception as e:
        logger.error("Failed to retrieve memories for user %s: %s", user_id, e)
        return []


def get_all_memories(user_id: str) -> List[Dict]:
    try:
        client = _get_mem0_client()
        results = client.get_all(filters={"user_id": user_id})
        return results.get("results", []) if isinstance(results, dict) else results
    except Exception as e:
        logger.error("Failed to get all memories for user %s: %s", user_id, e)
        return []


def delete_memory(memory_id: str) -> bool:
    try:
        client = _get_mem0_client()
        client.delete(memory_id)
        logger.info("Deleted memory %s", memory_id)
        return True
    except Exception as e:
        logger.error("Failed to delete memory %s: %s", memory_id, e)
        return False


def delete_all_memories(user_id: str) -> bool:
    try:
        client = _get_mem0_client()
        client.delete_all(filters={"user_id": user_id})
        logger.info("Deleted all memories for user %s", user_id)
        return True
    except Exception as e:
        logger.error("Failed to delete all memories for user %s: %s", user_id, e)
        return False


def format_memories_for_context(memories: List[Dict]) -> str:
    if not memories:
        return ""
    lines = ["User Memory Context:"]
    for m in memories:
        text = m.get("memory", m.get("text", ""))
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines)
