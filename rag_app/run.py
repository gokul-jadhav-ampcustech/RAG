"""
One-command startup script.
"""
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import logging
import sys
import threading
import webbrowser

import uvicorn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("run")


def main() -> None:
    from app.config import get_settings
    from app.db.database import check_connection, ensure_pgvector_extension, init_db

    settings = get_settings()

    logger.info("Step 1/5: Environment variables loaded.")

    logger.info("Step 2/5: Checking PostgreSQL connection...")
    if not check_connection():
        logger.error(
            "Could not connect to PostgreSQL using DATABASE_URL from .env.\n"
            "  - Is PostgreSQL running?\n"
            "  - Does the 'RAG' database exist?\n"
            "  - Is the username/password in .env correct? (special characters "
            "like '@' must be URL-encoded, e.g. admin@123 -> admin%40123)"
        )
        sys.exit(1)
    logger.info("PostgreSQL connection OK.")

    logger.info("Step 3/5: Ensuring pgvector extension is enabled...")
    if not ensure_pgvector_extension():
        logger.error(
            "pgvector extension is not enabled and could not be created automatically.\n"
            "Connect to the 'RAG' database as a superuser and run:\n"
            "    CREATE EXTENSION IF NOT EXISTS vector;"
        )
        sys.exit(1)
    logger.info("pgvector extension OK.")

    logger.info("Step 4/5: Creating tables if necessary...")
    init_db()
    logger.info("Database ready.")

    url = f"http://{settings.app_host}:{settings.app_port}"
    docs_url = f"{url}/docs"

    logger.info("Step 5/5: Starting FastAPI server...")

    def open_browser():
        import time

        time.sleep(1.5)
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=open_browser, daemon=True).start()

    print("\n" + "=" * 60)
    print(f"Application running at: {url}")
    print(f"API docs (Swagger):     {docs_url}")
    print("=" * 60 + "\n")

    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=False)


if __name__ == "__main__":
    main()
