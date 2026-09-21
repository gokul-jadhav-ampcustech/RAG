"""
FastAPI application entrypoint.
"""
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import chat, upload, auth, memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="SmartRAG Application",
    description="Multi-user RAG pipeline with authentication and mem0 memory.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth")
app.include_router(memory.router, prefix="/memory")
app.include_router(upload.router)
app.include_router(chat.router)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.on_event("startup")
def on_startup():
    """Initialize DB tables and demo user on startup."""
    from app.db.database import init_db
    init_db()  # creates new tables (long_term_memories, hitl_pending) if missing

    from app.db.database import SessionLocal
    from app.services.auth import get_user_by_username, create_user
    db = SessionLocal()
    try:
        if not get_user_by_username(db, "demo"):
            create_user(
                db=db,
                username="demo",
                email="demo@smartrag.local",
                password="demo123",
                full_name="Demo User",
            )
            logging.getLogger(__name__).info("Demo user created: demo / demo123")
    except Exception as e:
        logging.getLogger(__name__).warning("Demo user init: %s", e)
    finally:
        db.close()


@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/login")
def serve_login():
    return FileResponse(FRONTEND_DIR / "login.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/mcp/tools")
def mcp_tools_list():
    """List MCP tools discovered from the hosted server. Useful for diagnostics."""
    from app.config import get_settings
    from app.services.mcp_client import get_tools, format_tools_for_llm
    cfg = get_settings()
    if not cfg.mcp_enabled:
        return {"mcp_enabled": False, "tools": []}
    tools = get_tools()
    return {
        "mcp_enabled": True,
        "mcp_server_url": cfg.mcp_server_url,
        "tool_count": len(tools),
        "tools": tools,
        "formatted": format_tools_for_llm(tools),
    }
