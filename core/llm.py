"""
Smart LLM routing layer.
Routes each task to the cheapest model that can handle it well.

Task routing:
  search_rewrite    → Groq  (fast, cheap, good enough)
  classification    → Groq  (fast, cheap)
  search_explain    → Groq  (short output)
  pdf_chat          → Claude (accuracy critical)
  research_summary  → Claude (quality critical)
  copilot_analysis  → Claude (complex reasoning)
  property_extract  → Claude (structured extraction)
"""

import httpx
import logging
from core.config import get_settings

logger = logging.getLogger(__name__)

FALLBACK_TRIGGERS = (
    "credit", "quota", "billing", "insufficient_quota",
    "overloaded", "capacity", "529",
)

# Task → preferred model
TASK_ROUTING = {
    "search_rewrite":   "groq",
    "classification":   "groq",
    "search_explain":   "groq",
    "pdf_chat":         "claude",
    "research_summary": "claude",
    "copilot_analysis": "claude",
    "property_extract": "claude",
    "default":          "claude",
}

GROQ_MODEL    = "llama-3.3-70b-versatile"
CLAUDE_MODEL  = "claude-haiku-4-5"


def _should_fallback(error: Exception) -> bool:
    return any(t in str(error).lower() for t in FALLBACK_TRIGGERS)


async def _call_anthropic(system: str, messages: list, max_tokens: int) -> str:
    # BUG FIX: use AsyncAnthropic so we don't block the event loop
    import anthropic
    s = get_settings()
    if not s.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    client = anthropic.AsyncAnthropic(api_key=s.anthropic_api_key)
    r = await client.messages.create(
        model=CLAUDE_MODEL, max_tokens=max_tokens,
        system=system, messages=messages,
    )
    return r.content[0].text


async def _call_groq(system: str, messages: list, max_tokens: int) -> str:
    s = get_settings()
    if not s.groq_api_key:
        raise ValueError("GROQ_API_KEY not set")
    groq_messages = [{"role": "system", "content": system}] + messages
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {s.groq_api_key}", "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": groq_messages, "max_tokens": max_tokens, "temperature": 0.2},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def llm_call(
    system: str,
    messages: list,
    max_tokens: int = 1024,
    prefer_json: bool = False,
    task: str = "default",
) -> str:
    if prefer_json:
        system += "\n\nReturn ONLY valid JSON. No markdown fences, no explanation."

    s = get_settings()
    preferred = TASK_ROUTING.get(task, "claude")

    # Try preferred model first
    if preferred == "groq" and s.groq_api_key:
        try:
            text = await _call_groq(system, messages, max_tokens)
            logger.debug(f"LLM[{task}]: Groq OK")
            return text
        except Exception as e:
            logger.warning(f"Groq failed for task '{task}', falling back to Claude: {e}")

    if s.anthropic_api_key:
        try:
            text = await _call_anthropic(system, messages, max_tokens)
            logger.debug(f"LLM[{task}]: Claude OK")
            return text
        except Exception as e:
            if _should_fallback(e):
                logger.warning(f"Claude quota issue, trying Groq: {e}")
            else:
                raise

    if s.groq_api_key:
        try:
            text = await _call_groq(system, messages, max_tokens)
            logger.info(f"LLM[{task}]: Groq fallback OK")
            return text
        except Exception as e:
            raise RuntimeError(f"Both models failed. Last error: {e}")

    raise RuntimeError("No LLM available. Set ANTHROPIC_API_KEY or GROQ_API_KEY.")
