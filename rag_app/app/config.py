"""
Central application configuration.

All tunable values (models, chunking, retrieval thresholds, DB connection)
live here and are loaded from environment variables / .env.

Keeping this in ONE place means future features (reranking, query rewriting,
memory, etc.) can add their own settings here without touching business logic.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- Groq LLM ----
    groq_api_key: str
    llm_model: str = "openai/gpt-oss-120b"

    # ---- PostgreSQL ----
    database_url: str

    # ---- Embeddings ----
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ---- Retrieval ----
    top_k: int = 5
    similarity_threshold: float = 0.65

    # ---- Response Modes ----
    enable_general_responses: bool = True
    show_response_indicators: bool = True

    # ---- Chunking ----
    chunk_size: int = 700
    chunk_overlap: int = 100

    # ---- Long-Term Memory ----
    memory_top_k: int = 5
    memory_similarity_threshold: float = 0.75
    memory_importance_threshold: float = 0.50
    memory_dedup_threshold: float = 0.85

    # ---- Web Search ----
    web_search_max_results: int = 5
    web_search_timeout_seconds: int = 10

    # ---- HITL ----
    hitl_pending_ttl_minutes: int = 30

    # ---- MCP ----
    mcp_enabled: bool = True
    mcp_server_url: str = "https://mcptoolserverg.fastmcp.app/mcp"
    mcp_api_token: str = ""          # Bearer token for the hosted MCP server
    mcp_timeout: int = 30            # seconds per MCP call
    mcp_tool_cache_ttl: int = 3600   # seconds to cache tool list

    # ---- App ----
    app_host: str = "localhost"
    app_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Settings are cached so the .env file is only parsed once."""
    return Settings()
