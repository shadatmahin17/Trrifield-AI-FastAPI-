from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
import asyncio, json, time
from services.search_service import search_papers
from analytics.tracker import get_tracker
from models.schemas import SearchResponse
from core.config import get_settings

router = APIRouter()


async def _record_search_to_db(
    query, discipline, result_count, latency_ms, intent, success
):
    """Fire-and-forget DB write — never blocks the response."""
    if not get_settings().database_url:
        return
    try:
        from db.repositories import insert_search, insert_event
        await insert_search(
            query=query, discipline=discipline, result_count=result_count,
            latency_ms=latency_ms, intent=intent, success=success,
        )
        await insert_event(
            event_type="search", discipline=discipline, intent=intent,
            latency_ms=latency_ms, success=success,
            meta={"query": query, "result_count": result_count},
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"DB write failed (non-fatal): {e}")


@router.get("/", response_model=SearchResponse)
async def search(
    query:      str = Query(..., description="Natural language — typos, abbreviations, synonyms handled"),
    discipline: str = Query("all", description="all | aerospace | materials | textile"),
    year_from:  int = Query(None),
    year_to:    int = Query(None),
    limit:      int = Query(10, le=50, ge=1),
):
    t0      = time.time()
    tracker = get_tracker()
    try:
        papers, meta = await search_papers(
            query=query, discipline=discipline,
            year_from=year_from, year_to=year_to, limit=limit,
        )
        latency = round((time.time() - t0) * 1000, 1)
        intent  = meta.get("intent", "general")

        # In-memory tracker (instant)
        tracker.record_search(
            query=query, discipline=discipline, result_count=len(papers),
            latency_ms=latency, intent=intent, success=True,
        )
        # Persistent DB write (non-blocking)
        asyncio.create_task(_record_search_to_db(
            query, discipline, len(papers), latency, intent, True
        ))

        return SearchResponse(
            query               = query,
            interpreted_query   = meta.get("expanded_query", query),
            intent              = intent,
            detected_discipline = meta.get("discipline", discipline),
            rewrite_source      = meta.get("rewrite_source", "rules"),
            total               = len(papers),
            discipline          = discipline,
            papers              = papers,
        )
    except Exception as e:
        latency = round((time.time() - t0) * 1000, 1)
        tracker.record_search(
            query=query, discipline=discipline, result_count=0,
            latency_ms=latency, success=False,
        )
        asyncio.create_task(_record_search_to_db(
            query, discipline, 0, latency, "general", False
        ))
        msg = str(e)
        if "429" in msg or "rate limit" in msg.lower():
            raise HTTPException(status_code=429, detail="Rate limit — wait 30s and retry.")
        raise HTTPException(status_code=500, detail=msg)


@router.get("/history")
async def search_history(
    limit:      int = Query(50, le=200, ge=1),
    discipline: str = Query(None),
):
    """Recent search history from the database."""
    if not get_settings().database_url:
        raise HTTPException(503, "Database not configured")
    from db.repositories import get_search_history
    rows = await get_search_history(limit=limit, discipline=discipline)
    # Convert datetime objects for JSON serialisation
    for r in rows:
        if "created_at" in r and r["created_at"]:
            r["created_at"] = r["created_at"].isoformat()
    return {"total": len(rows), "history": rows}


@router.get("/stream")
async def search_stream(
    query:      str = Query(...),
    discipline: str = Query("all"),
    year_from:  int = Query(None),
    year_to:    int = Query(None),
    limit:      int = Query(10, le=50, ge=1),
):
    """
    SSE streaming search — emits progress events as each source completes,
    then the final ranked results. Frontend shows live progress.
    """
    import httpx
    from services.search_service import (
        _search_openalex, _search_crossref, _search_arxiv, _search_pubmed,
        _deduplicate, _enrich, _is_junk, _is_excluded, _tag_discipline,
        _cache_key, _get_cached, _set_cache,
    )
    from services.paper_scoring import rank_papers
    from agents.query_agent import rewrite_query

    async def event_stream():
        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        yield sse("start", {"message": "Query intelligence processing…"})

        rewrite = await rewrite_query(query, discipline)
        yield sse("rewrite", {
            "expanded_query": rewrite.get("expanded_query", query),
            "intent":         rewrite.get("intent", "general"),
            "discipline":     rewrite.get("discipline", discipline),
        })

        effective_disc = discipline if discipline != "all" else rewrite.get("discipline", "all")
        primary_query  = (rewrite.get("search_queries") or [query])[0]
        all_raw: list[dict] = []

        t0 = time.time()
        async with httpx.AsyncClient(timeout=20) as client:
            sources = [
                ("OpenAlex", _search_openalex(primary_query, effective_disc, year_from, year_to, limit, client)),
                ("Crossref", _search_crossref(primary_query, effective_disc, year_from, year_to, limit, client)),
                ("arXiv",    _search_arxiv(   primary_query, effective_disc, year_from, year_to, limit, client)),
                ("PubMed",   _search_pubmed(  primary_query, effective_disc, year_from, year_to, limit, client)),
            ]
            tasks = {name: asyncio.ensure_future(coro) for name, coro in sources}
            done_count = 0

            while tasks:
                done, _ = await asyncio.wait(list(tasks.values()), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    name = next(n for n, t in tasks.items() if t is task)
                    del tasks[name]
                    done_count += 1
                    try:
                        result = task.result()
                        all_raw.extend(result)
                        yield sse("source_complete", {
                            "source": name, "count": len(result),
                            "done": done_count, "total": 4,
                        })
                    except Exception as e:
                        yield sse("source_error", {"source": name, "error": str(e)})

            yield sse("ranking", {"message": f"Ranking {len(all_raw)} papers…"})

            clean  = [p for p in all_raw if not _is_junk(p) and not _is_excluded(p.get("concepts") or [])]
            merged = _deduplicate(clean)
            query_terms = rewrite.get("primary_keywords", query.split())
            ranked = rank_papers(merged, query_terms, query_terms, effective_disc)

            candidates   = ranked[:limit * 2]
            enriched     = await asyncio.gather(*[_enrich(p, client) for p in candidates], return_exceptions=True)
            final        = [r if isinstance(r, dict) else o for o, r in zip(candidates, enriched)]
            final_ranked = rank_papers(final, query_terms, query_terms, effective_disc)
            top          = final_ranked[:limit]

        papers = []
        for p in top:
            abstract = p.get("abstract")
            concepts = p.get("concepts") or []
            papers.append({
                "paper_id":        p["paper_id"],
                "title":           p.get("title") or "Untitled",
                "authors":         [{"name": a.name if hasattr(a, "name") else a.get("name", "")} for a in (p.get("authors") or [])],
                "year":            p.get("year"),
                "abstract":        abstract,
                "citation_count":  p.get("citation_count") or 0,
                "url":             p.get("url") or "",
                "open_access_url": p.get("open_access_url"),
                "journal":         p.get("journal"),
                "discipline_tag":  _tag_discipline(p.get("title", ""), abstract, concepts),
                "quality_score":   p.get("_quality_score", 0),
            })

        latency = round((time.time() - t0) * 1000, 1)
        intent  = rewrite.get("intent", "general")

        # Persist stream search to DB too
        asyncio.create_task(_record_search_to_db(
            query, effective_disc, len(papers), latency, intent, True
        ))

        yield sse("results", {
            "query":       query,
            "interpreted": rewrite.get("expanded_query", query),
            "intent":      intent,
            "discipline":  effective_disc,
            "total":       len(papers),
            "papers":      papers,
        })
        yield sse("done", {"message": "Search complete"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
