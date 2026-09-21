"""
Enhanced LLM wrapper with caching, rate limiting, and memory context support.
"""
import logging
import time
import hashlib

from groq import Groq

from app.config import get_settings
from app.prompts.rag_prompt import SYSTEM_PROMPT, GENERAL_SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

settings = get_settings()

_client: Groq | None = None

_response_cache: dict = {}
_cache_timeout = 3600


class LLMError(Exception):
    """Raised when the Groq API call fails."""


def _get_cache_key(question: str, is_general: bool = False) -> str:
    prefix = "general" if is_general else "doc"
    return hashlib.md5(f"{prefix}:{question}".encode()).hexdigest()


def _check_cache(cache_key: str) -> str | None:
    if cache_key in _response_cache:
        cached = _response_cache[cache_key]
        if time.time() - cached["timestamp"] < _cache_timeout:
            return cached["response"]
        del _response_cache[cache_key]
    return None


def _set_cache(cache_key: str, response: str):
    _response_cache[cache_key] = {"response": response, "timestamp": time.time()}


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=settings.groq_api_key)
    return _client


def _call_groq(messages: list, temperature: float = 0.1, max_tokens: int = 1024) -> str:
    """Shared Groq API call with error handling."""
    client = get_client()
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        error_msg = str(exc).lower()
        if "rate" in error_msg or "quota" in error_msg:
            logger.warning("Rate limit hit: %s", exc)
            time.sleep(1)
            raise LLMError("API rate limit reached. Please wait a moment.") from exc
        logger.error("Groq API call failed: %s", exc)
        raise LLMError(
            "The language model could not be reached. Check your GROQ_API_KEY."
        ) from exc

    answer = response.choices[0].message.content
    if not answer:
        raise LLMError("The language model returned an empty response.")
    return answer.strip()


def generate_answer(question: str, context_chunks: list[str], memory_context: str = "") -> str:
    """Call Groq with RAG system prompt + retrieved context + optional memory."""
    cache_key = _get_cache_key(question, is_general=False)
    if not memory_context:
        cached = _check_cache(cache_key)
        if cached:
            return cached

    user_prompt = build_user_prompt(question, context_chunks, memory_context=memory_context)
    answer = _call_groq(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    if not memory_context:
        _set_cache(cache_key, answer)
    return answer


def generate_general_answer(question: str, memory_context: str = "") -> str:
    """Generate a general AI response without document context."""
    cache_key = _get_cache_key(question, is_general=True)
    if not memory_context:
        cached = _check_cache(cache_key)
        if cached:
            return cached

    user_msg = question
    if memory_context:
        user_msg = f"{memory_context}\n\nUser question: {question}"

    answer = _call_groq(
        messages=[
            {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
    )
    if not memory_context:
        _set_cache(cache_key, answer)
    return answer


def get_model_name() -> str:
    return settings.llm_model
