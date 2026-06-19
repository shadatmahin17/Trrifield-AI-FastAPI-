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

MAX_PAPERS = 10   # always give exactly 10 papers to context and return 10 refs


def _papers_to_context(papers: list[dict]) -> str:
    """Format paper list into LLM-readable context (up to MAX_PAPERS)."""
    lines = []
    for i, p in enumerate(papers[:MAX_PAPERS], 1):
        title    = p.get("title", "Untitled")
        year     = p.get("year", "?")
        journal  = p.get("journal", "Unknown journal")
        cited    = p.get("citation_count", 0)
        authors  = p.get("authors") or []
        # Build "First Author et al." string for context so LLM can cite correctly
        if authors:
            first = (authors[0].get("name") or authors[0]) if isinstance(authors[0], dict) else authors[0]
            last_name = first.split()[-1] if first else "Unknown"
            author_str = f"{last_name} et al." if len(authors) > 1 else first
        else:
            author_str = "Unknown"
        # Use full abstract up to 800 chars — enough for meaningful synthesis
        abstract = (p.get("abstract") or "No abstract available")[:800]
        lines.append(
            f"[{i}] {title} ({year}) — {author_str} — {journal} — {cited} citations\n"
            f"    Abstract: {abstract}"
        )
    return "\n\n".join(lines)


def _build_papers_payload(papers: list[dict]) -> list[dict]:
    """Return enriched paper dicts (top MAX_PAPERS) for the frontend references list."""
    result = []
    for p in papers[:MAX_PAPERS]:
        authors = p.get("authors") or []
        result.append({
            "title":          p.get("title", "Untitled"),
            "year":           p.get("year"),
            "journal":        p.get("journal"),
            "citation_count": p.get("citation_count", 0),
            "url":            p.get("url"),
            "open_access_url": p.get("open_access_url"),
            "authors":        authors,
        })
    return result


async def run_copilot(query: str, papers: list[dict]) -> dict:
    """
    Generate full research intelligence report from search results.
    Always returns exactly MAX_PAPERS references with url/open_access_url
    so the frontend can render clickable (Author et al., Year) citations.
    """
    if not papers:
        return {
            "summary":               "No papers found to analyse.",
            "key_papers":            [],
            "papers":                [],
            "research_trends":       [],
            "research_gaps":         [],
            "future_directions":     [],
            "suggested_experiments": [],
        }

    # Cap to MAX_PAPERS
    papers = papers[:MAX_PAPERS]
    context = _papers_to_context(papers)
    papers_payload = _build_papers_payload(papers)

    prompt = (
        f"Research query: '{query}'\n\n"
        f"Papers found:\n{context}\n\n"
        f"Generate a research intelligence report. "
        f"You MUST reference papers using [N] markers matching the numbers above. "
        f"In the summary field use inline citations like (Author et al., Year) AND [N] markers."
    )

    try:
        raw = await llm_call(
            system=COPILOT_ANALYSIS,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,          # raised from 1500 — prevents JSON truncation
            prefer_json=True,
            task="copilot_analysis",
        )
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result["query"]       = query
        result["paper_count"] = len(papers)
        # Always attach full enriched papers list for frontend reference rendering
        result["papers"]      = papers_payload
        # Ensure key_papers also carries url fields if LLM returned them without
        if "key_papers" in result:
            for kp in result["key_papers"]:
                # Try to match by title to inject url from papers_payload
                matched = next((p for p in papers_payload if p["title"] == kp.get("title")), None)
                if matched:
                    kp.setdefault("url", matched.get("url"))
                    kp.setdefault("open_access_url", matched.get("open_access_url"))
                    kp.setdefault("authors", matched.get("authors"))
        return result

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Copilot analysis failed: {e!r}")
        # Fallback: return ALL papers (not just 3) with real metadata
        return {
            "query":       query,
            "summary":     f"Analysed {len(papers)} papers on '{query}'. Detailed analysis temporarily unavailable.",
            "key_papers":  papers_payload,   # all MAX_PAPERS, not [:3]
            "papers":      papers_payload,
            "research_trends":       ["Analysis temporarily unavailable — please retry."],
            "research_gaps":         ["Analysis temporarily unavailable — please retry."],
            "future_directions":     [],
            "suggested_experiments": [],
            "paper_count": len(papers),
            "error":       str(e),
        }


async def generate_summary(query: str, papers: list[dict]) -> dict:
    """
    Lightweight research landscape summary.
    Returns a dict with 'summary' text and 'papers' list (for frontend ref rendering).
    """
    if not papers:
        return {"summary": "No papers available to summarise.", "papers": []}

    papers = papers[:MAX_PAPERS]
    papers_payload = _build_papers_payload(papers)
    context = _papers_to_context(papers)
    prompt  = (
        f"Query: '{query}'\n\nPapers:\n{context}\n\n"
        "In your summary use inline citations formatted as (Author et al., Year) "
        "AND [N] markers matching the paper numbers above so citations are clickable."
    )

    text = await llm_call(
        system=RESEARCH_SUMMARY,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,             # raised from 600
        task="research_summary",
    )
    return {"summary": text, "papers": papers_payload}
