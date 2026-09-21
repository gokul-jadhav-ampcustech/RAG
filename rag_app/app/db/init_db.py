"""
Standalone database initialization script.

Run with:
    python -m app.db.init_db

This is also called automatically from run.py, so most users won't need
to run it by hand.
"""
import logging
import sys

from app.db.database import check_connection, ensure_pgvector_extension, init_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Checking PostgreSQL connection...")
    if not check_connection():
        logger.error(
            "Could not connect to PostgreSQL. Make sure it is running and that "
            "DATABASE_URL in .env is correct."
        )
        sys.exit(1)
    logger.info("PostgreSQL connection OK.")

    logger.info("Ensuring pgvector extension is enabled...")
    if not ensure_pgvector_extension():
        logger.error(
            "pgvector extension could not be enabled automatically. "
            "Connect as a superuser and run: CREATE EXTENSION IF NOT EXISTS vector;"
        )
        sys.exit(1)
    logger.info("pgvector extension OK.")

    logger.info("Creating tables (if they don't already exist)...")
    init_db()
    logger.info("Database initialization complete.")


if __name__ == "__main__":
    main()
