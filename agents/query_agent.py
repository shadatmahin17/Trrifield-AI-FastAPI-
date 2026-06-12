"""
LLM-powered query rewriting agent.
Transforms raw user queries into optimised multi-source search queries.
Falls back to rule-based query_intelligence if LLM fails.
"""
import json
import logging
import asyncio
from core.llm import llm_call
from prompts.templates import QUERY_REWRITE
from services.query_intelligence import analyse_query, QueryAnalysis

logger = logging.getLogger(__name__)


async def rewrite_query(raw_query: str, discipline: str = "all") -> dict:
    """
    Use LLM to rewrite and expand a user query.
    Returns dict with expanded_query, search_queries, keywords, intent, discipline.
    Falls back to rule-based analysis if LLM fails or is slow.
    """
    # Run rule-based analysis always (instant, as baseline)
    rule_based = analyse_query(raw_query, discipline)

    # Try LLM rewriting (Groq — fast and cheap for this task)
    try:
        raw = await asyncio.wait_for(
            llm_call(
                system=QUERY_REWRITE,
                messages=[{"role": "user", "content": f"User query: {raw_query}\nDiscipline hint: {discipline}"}],
                max_tokens=400,
                prefer_json=True,
                task="search_rewrite",
            ),
            timeout=15.0  # Groq cold start can be slow on free tier
        )
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        llm_result = json.loads(raw)

        # Merge LLM result with rule-based (rule-based adds typo correction)
        return {
            "original_query":  raw_query,
            "expanded_query":  llm_result.get("expanded_query", rule_based.primary_query),
            "primary_keywords": llm_result.get("primary_keywords", rule_based.entities),
            "synonyms":        llm_result.get("synonyms", rule_based.expanded_terms[:4]),
            "search_queries":  llm_result.get("search_queries", [rule_based.primary_query] + rule_based.secondary_queries),
            "intent":          llm_result.get("intent", rule_based.intent),
            "discipline":      llm_result.get("discipline", rule_based.discipline),
            "corrected_query": rule_based.corrected,
            "rewrite_source":  "llm",
        }

    except asyncio.TimeoutError:
        logger.warning("LLM query rewrite timed out — using rule-based fallback")
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"LLM query rewrite failed ({e}) — using rule-based fallback")

    # Rule-based fallback
    return {
        "original_query":  raw_query,
        "expanded_query":  rule_based.primary_query,
        "primary_keywords": rule_based.entities,
        "synonyms":        rule_based.expanded_terms[:4],
        "search_queries":  [rule_based.primary_query] + rule_based.secondary_queries[:2],
        "intent":          rule_based.intent,
        "discipline":      rule_based.discipline,
        "corrected_query": rule_based.corrected,
        "rewrite_source":  "rules",
    }
