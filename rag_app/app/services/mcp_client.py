"""
Async MCP Client
================
Connects to a hosted FastMCP server using the Streamable-HTTP transport.
Dynamically discovers tools on first use and caches them.

Key design decisions:
- Uses the venv's MCP version: streamablehttp_client (no underscore),
  httpx.Auth for bearer token, yields (read, write, get_session_id).
- All errors are caught and surfaced as MCPError — never crash the RAG app.
- Tool list is cached in-process for `mcp_tool_cache_ttl` seconds.
- Bearer token is read from settings (MCP_API_TOKEN env var).
"""
import asyncio
import logging
import time
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── In-process tool cache ─────────────────────────────────────────────────────
_tool_cache: list[dict] | None = None
_tool_cache_ts: float = 0.0


class MCPError(Exception):
    """Raised when the MCP server call fails."""


class _BearerAuth(httpx.Auth):
    """Simple Bearer token auth for httpx."""

    def __init__(self, token: str) -> None:
        self._token = token

    def auth_flow(self, request: httpx.Request):
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request


def _get_auth() -> httpx.Auth | None:
    token = (settings.mcp_api_token or "").strip()
    return _BearerAuth(token) if token else None


# ── Core async helpers ────────────────────────────────────────────────────────

async def _async_list_tools() -> list[dict]:
    """Connect to MCP server and return the tool list."""
    async with streamablehttp_client(
        url=settings.mcp_server_url,
        timeout=float(settings.mcp_timeout),
        auth=_get_auth(),
    ) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            tools = []
            for t in result.tools:
                tools.append({
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema if hasattr(t, "inputSchema") else {},
                })
            logger.info("[MCP] Discovered %d tools: %s", len(tools), [t["name"] for t in tools])
            return tools


async def _async_call_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """Call a single MCP tool and return its text content."""
    async with streamablehttp_client(
        url=settings.mcp_server_url,
        timeout=float(settings.mcp_timeout),
        auth=_get_auth(),
    ) as (read, write, _get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            parts = []
            for item in result.content:
                if hasattr(item, "text"):
                    parts.append(item.text)
                else:
                    parts.append(str(item))
            return "\n".join(parts) if parts else ""


# ── Sync-safe public API (called from FastAPI sync endpoints) ─────────────────

def _run(coro):
    """Run an async coroutine from a sync context safely."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside a running event loop — offload to a fresh thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=settings.mcp_timeout + 5)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def get_tools(force_refresh: bool = False) -> list[dict]:
    """
    Return cached tool list, refreshing if stale or forced.
    Returns [] on any error — never raises.
    """
    global _tool_cache, _tool_cache_ts
    now = time.time()
    if (
        not force_refresh
        and _tool_cache is not None
        and (now - _tool_cache_ts) < settings.mcp_tool_cache_ttl
    ):
        return _tool_cache

    try:
        tools = _run(_async_list_tools())
        _tool_cache = tools
        _tool_cache_ts = now
        return tools
    except Exception as exc:
        logger.warning("[MCP] Tool discovery failed: %s", exc)
        return _tool_cache or []  # return stale cache if available


def call_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """
    Execute an MCP tool synchronously.
    Raises MCPError on failure.
    """
    try:
        result = _run(_async_call_tool(tool_name, arguments))
        logger.info("[MCP] Tool '%s' executed successfully", tool_name)
        return result
    except Exception as exc:
        logger.error("[MCP] Tool '%s' failed: %s", tool_name, exc)
        raise MCPError(f"MCP tool '{tool_name}' failed: {exc}") from exc


def is_available() -> bool:
    """Quick health check — returns True if tool list is reachable."""
    if not settings.mcp_enabled:
        return False
    try:
        tools = get_tools()
        return isinstance(tools, list)
    except Exception:
        return False


def format_tools_for_llm(tools: list[dict]) -> str:
    """Format tool list as a compact string for LLM prompts."""
    if not tools:
        return "No MCP tools available."
    lines = []
    for t in tools:
        schema = t.get("input_schema", {})
        props = schema.get("properties", {})
        params = ", ".join(
            f"{k}({v.get('type', 'any')})" for k, v in props.items()
        ) if props else "no params"
        lines.append(f"- {t['name']}: {t['description']} | params: {params}")
    return "\n".join(lines)
