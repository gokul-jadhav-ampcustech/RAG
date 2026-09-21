"""
Database engine, session management, and initialization helpers.

This module is intentionally the ONLY place that talks to SQLAlchemy's
engine/session machinery, so swapping database details later (pooling,
read replicas, etc.) doesn't ripple through the rest of the app.
"""
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_connection() -> bool:
    """Verify we can actually reach PostgreSQL."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Database connection failed: %s", exc)
        return False


def ensure_pgvector_extension() -> bool:
    """Make sure the pgvector extension is enabled on this database."""
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Could not enable the pgvector extension automatically (%s). "
            "You may need to run 'CREATE EXTENSION IF NOT EXISTS vector;' "
            "manually as a superuser.",
            exc,
        )
        return False


def init_db() -> None:
    """Create all tables defined in app.db.models if they don't exist yet."""
    from app.db import memory_models  # noqa: F401  (registers User, Memory etc.)
    from app.db import models  # noqa: F401  (registers Document, Conversation, Message, LongTermMemory, HITLPending)

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured (created if missing).")
