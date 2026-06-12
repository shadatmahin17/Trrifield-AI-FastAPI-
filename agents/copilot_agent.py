"""
Research Copilot Agent.
Analyses a set of search results and generates:
  - Key papers summary
  - Research trends
  - Research gaps
  - Future directions
  - Suggested experiments
"""
import json
import logging
from core.llm import llm_call
from prompts.templates import COPILOT_ANALYSIS, RESEARCH_SUMMARY

logger = logging.getLogger(__name__)


def _papers_to_context(papers: list[dict]) -> str:
    """Format paper list into LLM-readable context."""
    lines = []
    for i, p in enumerate(papers[:12], 1):
        title    = p.get("title", "Untitled")
        year     = p.get("year", "?")
        journal  = p.get("journal", "Unknown journal")
        cited    = p.get("citation_count", 0)
        abstract = (p.get("abstract") or "No abstract available")[:400]
        lines.append(
            f"[{i}] {title} ({year}) — {journal} — {cited} citations\n"
            f"    Abstract: {abstract}"
        )
    return "\n\n".join(lines)


async def run_copilot(query: str, papers: list[dict]) -> dict:
    """
    Generate full research intelligence report from search results.
    Routes to Claude (complex reasoning task).
    """
    if not papers:
        return {
            "summary":              "No papers found to analyse.",
            "key_papers":           [],
            "research_trends":      [],
            "research_gaps":        [],
            "future_directions":    [],
            "suggested_experiments": [],
        }

    context = _papers_to_context(papers)

    prompt = (
        f"Research query: '{query}'\n\n"
        f"Papers found:\n{context}\n\n"
        "Generate a research intelligence report for this field."
    )

    try:
        raw = await llm_call(
            system=COPILOT_ANALYSIS,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            prefer_json=True,
            task="copilot_analysis",   # routes to Claude
        )
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result["query"] = query
        result["paper_count"] = len(papers)
        return result

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Copilot analysis failed: {e}")
        # Graceful fallback
        return {
            "query":       query,
            "summary":     f"Analysed {len(papers)} papers on '{query}'.",
            "key_papers":  [{"title": p.get("title",""), "year": p.get("year"), "significance": "High citation count"} for p in papers[:3]],
            "research_trends":       ["Analysis temporarily unavailable"],
            "research_gaps":         ["Analysis temporarily unavailable"],
            "future_directions":     [],
            "suggested_experiments": [],
            "paper_count": len(papers),
            "error": str(e),
        }


async def generate_summary(query: str, papers: list[dict]) -> str:
    """
    Lightweight research landscape summary (shorter than full copilot).
    """
    if not papers:
        return "No papers available to summarise."

    context = _papers_to_context(papers[:6])
    prompt  = f"Query: '{query}'\n\nPapers:\n{context}"

    return await llm_call(
        system=RESEARCH_SUMMARY,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        task="research_summary",
    )
