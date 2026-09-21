"""
MCP Router
==========
Uses the LLM to decide how to answer a question:
  - "rag"      → existing 9-stage RAG pipeline only
  - "mcp"      → MCP tool call only
  - "rag+mcp"  → RAG first, then enrich with MCP tool
  - "general"  → plain LLM answer (no docs, no tools)

Also handles: tool argument extraction, result formatting, and
combining RAG + MCP answers into a single coherent response.
"""
import json
import logging
import re
from typing import Any

from app.services import llm
from app.services.mcp_client import (
    MCPError,
    call_tool,
    format_tools_for_llm,
    get_tools,
)

logger = logging.getLogger(__name__)

# ── Intent classification ─────────────────────────────────────────────────────

_ROUTE_SYSTEM = (
    "You are a routing assistant. Given a user question and a list of available MCP tools, "
    "decide the best way to answer.\n\n"
    "Reply with ONLY a JSON object — no markdown, no explanation:\n"
    '{"route": "<rag|mcp|rag+mcp|general>", "tool": "<tool_name or null>", '
    '"arguments": {<tool arguments or {}>}, "reason": "<one sentence>"}\n\n'
    "Rules:\n"
    "- Use 'rag' when the question is about uploaded documents.\n"
    "- Use 'mcp' when an MCP tool can directly answer the question.\n"
    "- Use 'rag+mcp' when documents provide context AND a tool provides live/computed data.\n"
    "- Use 'general' for general knowledge questions with no relevant tool.\n"
    "- If no tool fits, set tool=null and route to 'rag' or 'general'.\n"
    "- Extract tool arguments from the question text.\n"
    "- If a tool requires arguments you cannot extract, route to 'general' instead."
)


def classify_intent(question: str, tools: list[dict]) -> dict:
    """
    Ask the LLM to classify the question and pick a tool if needed.
    Returns a dict: {route, tool, arguments, reason}
    Falls back to {"route": "rag", "tool": None, "arguments": {}} on any error.
    """
    fallback = {"route": "rag", "tool": None, "arguments": {}, "reason": "fallback"}

    if not tools:
        return {"route": "rag", "tool": None, "arguments": {}, "reason": "no mcp tools available"}

    tools_text = format_tools_for_llm(tools)
    user_msg = (
        f"Available MCP tools:\n{tools_text}\n\n"
        f"User question: {question}\n\n"
        "Respond with the JSON routing decision."
    )

    try:
        raw = llm._call_groq(
            messages=[
                {"role": "system", "content": _ROUTE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        decision = json.loads(raw)
        route = decision.get("route", "rag")
        if route not in ("rag", "mcp", "rag+mcp", "general"):
            route = "rag"
        return {
            "route": route,
            "tool": decision.get("tool"),
            "arguments": decision.get("arguments") or {},
            "reason": decision.get("reason", ""),
        }
    except Exception as exc:
        logger.warning("[MCP_ROUTER] Intent classification failed: %s — defaulting to rag", exc)
        return fallback


# ── Tool execution with result formatting ─────────────────────────────────────

def execute_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
    """
    Execute an MCP tool. Returns (success, result_text).
    Never raises — errors are returned as (False, error_message).
    """
    try:
        result = call_tool(tool_name, arguments)
        return True, result
    except MCPError as exc:
        logger.warning("[MCP_ROUTER] Tool execution error: %s", exc)
        return False, str(exc)


# ── Combined RAG + MCP answer synthesis ───────────────────────────────────────

_COMBINE_SYSTEM = (
    "You are a helpful assistant. You have been given:\n"
    "1. An answer from the user's document knowledge base (RAG answer).\n"
    "2. Live data from an external tool (MCP tool result).\n\n"
    "Synthesize both into a single, coherent, well-structured answer. "
    "Clearly distinguish document-based information from live tool data. "
    "Be concise and factual."
)


def synthesize_rag_and_mcp(question: str, rag_answer: str, tool_name: str, tool_result: str) -> str:
    """Combine a RAG answer with an MCP tool result into one response."""
    user_msg = (
        f"User question: {question}\n\n"
        f"=== Document-based answer ===\n{rag_answer}\n\n"
        f"=== Live data from MCP tool '{tool_name}' ===\n{tool_result}\n\n"
        "Please synthesize these into a single answer."
    )
    try:
        return llm._call_groq(
            messages=[
                {"role": "system", "content": _COMBINE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
    except Exception as exc:
        logger.warning("[MCP_ROUTER] Synthesis failed: %s — returning RAG answer", exc)
        return f"{rag_answer}\n\n**Additional info from {tool_name}:**\n{tool_result}"


def format_mcp_only_answer(question: str, tool_name: str, tool_result: str) -> str:
    """Ask the LLM to present an MCP tool result as a natural language answer."""
    user_msg = (
        f"User question: {question}\n\n"
        f"=== Result from MCP tool '{tool_name}' ===\n{tool_result}\n\n"
        "Present this result as a clear, helpful answer to the user's question."
    )
    try:
        return llm._call_groq(
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Present tool results clearly and concisely."},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
    except Exception as exc:
        logger.warning("[MCP_ROUTER] MCP answer formatting failed: %s", exc)
        return f"**{tool_name} result:**\n{tool_result}"
