"""
Automatic Memory Extraction
============================
Analyzes user messages to identify long-term memory candidates.
Uses rule-based pre-filtering + LLM extraction.
The user never needs to say "remember this".
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Rule-based pre-filter: skip obviously low-value messages ──────────────────

_LOW_VALUE_PATTERNS = re.compile(
    r"^(thanks?|thank you|ok(ay)?|sure|got it|yes|no|hi|hello|bye|"
    r"good|great|nice|cool|awesome|perfect|alright|fine|understood|"
    r"please|sorry|excuse me|pardon|what\?|huh\?|really\?|wow)[\s!?.]*$",
    re.IGNORECASE,
)

_MIN_WORDS = 6  # messages shorter than this are almost never worth storing

# ── Memory types ──────────────────────────────────────────────────────────────

MEMORY_TYPES = {
    "preference", "project", "goal", "technical_context",
    "decision", "fact", "instruction", "profile",
}

# ── Extraction prompt ─────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are a Long-Term Memory Extraction system.

Analyze the user's message and conversation context.
Identify ONLY information that is likely to be useful in FUTURE conversations.

Extract information ONLY when it is:
1. Stable or likely to remain relevant over time
2. Useful for future personalization or assistance
3. Related to user preferences (language, framework, style, format)
4. Related to long-term goals
5. Related to ongoing projects
6. Related to technical architecture or stack
7. Related to important decisions
8. Related to persistent instructions

DO NOT extract:
1. Greetings, casual conversation, thanks
2. Temporary questions ("what is X?")
3. Short-lived debugging info
4. Duplicate information already in context
5. Information with low future usefulness
6. Random facts the user asks about (not about themselves)

Return ONLY valid JSON array. Each item must have:
{
  "memory_type": one of [preference, project, goal, technical_context, decision, fact, instruction, profile],
  "content": "concise third-person statement about the user",
  "importance_score": 0.0-1.0,
  "confidence_score": 0.0-1.0,
  "reason": "brief reason why this is worth storing",
  "should_store": true or false
}

If nothing is worth storing, return: []
Do NOT invent information. Do NOT create fictional memories."""


@dataclass
class MemoryCandidate:
    memory_type: str
    content: str
    importance_score: float
    confidence_score: float
    reason: str
    should_store: bool
    raw: dict = field(default_factory=dict)


def is_low_value(message: str) -> bool:
    """Quick rule-based check before calling LLM."""
    msg = message.strip()
    if len(msg.split()) < _MIN_WORDS:
        return True
    if _LOW_VALUE_PATTERNS.match(msg):
        return True
    return False


def extract_candidates(
    user_message: str,
    conversation_context: List[dict],
    llm_call_fn,
) -> List[MemoryCandidate]:
    """
    Extract memory candidates from a user message.

    Args:
        user_message: The raw user message
        conversation_context: Recent messages [{"role": ..., "content": ...}]
        llm_call_fn: Callable(messages) -> str  (the LLM call)

    Returns:
        List of MemoryCandidate objects
    """
    logger.info("[MEMORY] Extraction started for message: %r", user_message[:80])

    if is_low_value(user_message):
        logger.info("[MEMORY] Low-value message — skipping extraction")
        return []

    # Build context summary (last 3 exchanges max)
    context_lines = []
    for msg in conversation_context[-6:]:
        role = msg.get("role", "")
        content = msg.get("content", "")[:200]
        context_lines.append(f"{role.upper()}: {content}")
    context_str = "\n".join(context_lines) if context_lines else "No prior context."

    user_prompt = (
        f"Recent conversation context:\n{context_str}\n\n"
        f"Current user message:\n{user_message}\n\n"
        f"Extract long-term memory candidates as JSON array."
    )

    try:
        raw_response = llm_call_fn([
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
    except Exception as e:
        logger.error("[MEMORY] LLM extraction call failed: %s", e)
        return []

    candidates = _parse_response(raw_response)
    logger.info("[MEMORY] Extraction complete: %d candidates", len(candidates))
    return candidates


def _parse_response(raw: str) -> List[MemoryCandidate]:
    """Parse LLM JSON response into MemoryCandidate objects."""
    # Extract JSON array from response
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []

    try:
        items = json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.warning("[MEMORY] JSON parse error: %s", e)
        return []

    candidates = []
    for item in items:
        if not isinstance(item, dict):
            continue
        memory_type = item.get("memory_type", "fact")
        if memory_type not in MEMORY_TYPES:
            memory_type = "fact"
        candidates.append(MemoryCandidate(
            memory_type=memory_type,
            content=str(item.get("content", "")).strip(),
            importance_score=float(item.get("importance_score", 0.5)),
            confidence_score=float(item.get("confidence_score", 0.5)),
            reason=str(item.get("reason", "")),
            should_store=bool(item.get("should_store", False)),
            raw=item,
        ))

    return [c for c in candidates if c.content and c.should_store]
