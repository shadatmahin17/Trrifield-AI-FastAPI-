from fastapi import APIRouter, Query
from fastapi import HTTPException
from analytics.tracker import get_tracker
from core.config import get_settings

router = APIRouter()

EMPTY_STATS = {
    "total_searches":         0,
    "successful_searches":    0,
    "failed_searches":        0,
    "success_rate_pct":       0.0,
    "avg_latency_ms":         0.0,
    "p95_latency_ms":         0.0,
    "avg_results_per_search": 0.0,
    "top_queries":            [],
    "top_disciplines":        [],
    "top_intents":            [],
    "top_failed_queries":     [],
    "total_pdf_uploads":      0,
    "total_saved_papers":     0,
    "total_citations_saved":  0,
}


@router.get("/")
async def get_analytics():
    """
    Usage analytics — prefers persistent DB stats, falls back to in-memory.
    Returns a consistent shape regardless of whether any events exist.
    """
    if get_settings().database_url:
        try:
            from db.repositories import get_analytics_stats
            return await get_analytics_stats()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"DB analytics failed, using in-memory: {e}")

    # In-memory fallback
    stats = get_tracker().get_stats()
    if "message" in stats:
        return {**EMPTY_STATS, "note": stats["message"]}
    return {**EMPTY_STATS, **stats}


@router.get("/searches")
async def recent_searches(
    limit:      int = Query(50, le=200, ge=1),
    discipline: str = Query(None),
):
    """Recent search history with timestamps."""
    if not get_settings().database_url:
        raise HTTPException(503, "Database not configured")
    from db.repositories import get_search_history
    rows = await get_search_history(limit=limit, discipline=discipline)
    for r in rows:
        if r.get("created_at"):
            r["created_at"] = r["created_at"].isoformat()
    return {"total": len(rows), "searches": rows}


@router.get("/top-queries")
async def top_queries(limit: int = Query(10, le=50)):
    """Most frequently searched queries."""
    if not get_settings().database_url:
        # Fall back to in-memory counter
        stats = get_tracker().get_stats()
        return {"top_queries": stats.get("top_queries", [])}
    from db.repositories import get_top_queries
    rows = await get_top_queries(limit=limit)
    return {"top_queries": [[r["query"], r["count"]] for r in rows]}
