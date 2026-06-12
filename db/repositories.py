"""
Database access layer — one function per operation.
All functions are async and use the shared connection pool.
"""
import json
import logging
from datetime import datetime, timezone
from db.connection import get_pool

logger = logging.getLogger(__name__)


# ── Search history ──────────────────────────────────────────────────────────

async def insert_search(
    query: str,
    discipline: str,
    result_count: int,
    latency_ms: float,
    intent: str = "general",
    success: bool = True,
) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO search_history
                (query, discipline, result_count, latency_ms, intent, success)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            """,
            query, discipline, result_count, latency_ms, intent, success,
        )
        return row["id"]


async def get_search_history(limit: int = 50, discipline: str | None = None) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        if discipline and discipline != "all":
            rows = await conn.fetch(
                "SELECT * FROM search_history WHERE discipline=$1 ORDER BY created_at DESC LIMIT $2",
                discipline, limit,
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM search_history ORDER BY created_at DESC LIMIT $1", limit
            )
        return [dict(r) for r in rows]


async def get_top_queries(limit: int = 10) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT LOWER(query) AS query, COUNT(*) AS count
            FROM search_history
            GROUP BY LOWER(query)
            ORDER BY count DESC
            LIMIT $1
            """,
            limit,
        )
        return [dict(r) for r in rows]


# ── Analytics events ─────────────────────────────────────────────────────────

async def insert_event(
    event_type: str,
    discipline: str | None = None,
    intent: str | None = None,
    latency_ms: float | None = None,
    success: bool = True,
    meta: dict | None = None,
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO analytics_events
                (event_type, discipline, intent, latency_ms, success, meta)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            event_type, discipline, intent, latency_ms, success,
            json.dumps(meta) if meta else None,
        )


async def get_analytics_stats() -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Overall search counts
        totals = await conn.fetchrow(
            """
            SELECT
                COUNT(*)                                          AS total_searches,
                COUNT(*) FILTER (WHERE success = TRUE)           AS successful,
                COUNT(*) FILTER (WHERE success = FALSE)          AS failed,
                ROUND(AVG(latency_ms)::numeric, 1)               AS avg_latency_ms,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP
                    (ORDER BY latency_ms)::numeric, 1)           AS p95_latency_ms,
                ROUND(AVG(result_count)::numeric, 1)             AS avg_results
            FROM search_history
            """
        )

        top_queries = await conn.fetch(
            """
            SELECT LOWER(query) AS query, COUNT(*) AS count
            FROM search_history GROUP BY LOWER(query)
            ORDER BY count DESC LIMIT 10
            """
        )

        top_disciplines = await conn.fetch(
            """
            SELECT discipline, COUNT(*) AS count
            FROM search_history GROUP BY discipline
            ORDER BY count DESC LIMIT 5
            """
        )

        top_intents = await conn.fetch(
            """
            SELECT intent, COUNT(*) AS count
            FROM search_history GROUP BY intent
            ORDER BY count DESC LIMIT 5
            """
        )

        top_failed = await conn.fetch(
            """
            SELECT LOWER(query) AS query, COUNT(*) AS count
            FROM search_history WHERE success = FALSE
            GROUP BY LOWER(query) ORDER BY count DESC LIMIT 5
            """
        )

        pdf_count = await conn.fetchval("SELECT COUNT(*) FROM pdf_sessions")
        saved_count = await conn.fetchval("SELECT COUNT(*) FROM saved_papers")
        citation_count = await conn.fetchval("SELECT COUNT(*) FROM citations")

        total = totals["total_searches"] or 0
        successful = totals["successful"] or 0

        return {
            "total_searches":         total,
            "successful_searches":    successful,
            "failed_searches":        totals["failed"] or 0,
            "success_rate_pct":       round(successful / total * 100, 1) if total else 0.0,
            "avg_latency_ms":         float(totals["avg_latency_ms"] or 0),
            "p95_latency_ms":         float(totals["p95_latency_ms"] or 0),
            "avg_results_per_search": float(totals["avg_results"] or 0),
            "top_queries":            [[r["query"], r["count"]] for r in top_queries],
            "top_disciplines":        [[r["discipline"], r["count"]] for r in top_disciplines],
            "top_intents":            [[r["intent"], r["count"]] for r in top_intents],
            "top_failed_queries":     [[r["query"], r["count"]] for r in top_failed],
            "total_pdf_uploads":      pdf_count or 0,
            "total_saved_papers":     saved_count or 0,
            "total_citations_saved":  citation_count or 0,
        }


# ── PDF sessions ──────────────────────────────────────────────────────────────

async def insert_pdf_session(
    session_id: str,
    filename: str,
    file_size_mb: float,
    chunk_count: int,
    latency_ms: float,
):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO pdf_sessions
                (session_id, filename, file_size_mb, chunk_count, latency_ms)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (session_id) DO UPDATE
                SET last_accessed = NOW()
            """,
            session_id, filename, file_size_mb, chunk_count, latency_ms,
        )


async def get_pdf_session(session_id: str) -> dict | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM pdf_sessions WHERE session_id=$1", session_id
        )
        if row:
            await conn.execute(
                "UPDATE pdf_sessions SET last_accessed=NOW() WHERE session_id=$1", session_id
            )
            return dict(row)
        return None


async def session_exists_in_db(session_id: str) -> bool:
    pool = await get_pool()
    async with pool.acquire() as conn:
        val = await conn.fetchval(
            "SELECT 1 FROM pdf_sessions WHERE session_id=$1", session_id
        )
        return val is not None


async def get_pdf_sessions(limit: int = 20) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM pdf_sessions ORDER BY created_at DESC LIMIT $1", limit
        )
        return [dict(r) for r in rows]


# ── PDF chat history ──────────────────────────────────────────────────────────

async def append_chat_message(session_id: str, role: str, content: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO pdf_chat_history (session_id, role, content)
            VALUES ($1, $2, $3)
            """,
            session_id, role, content,
        )


async def get_chat_history(session_id: str) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT role, content, created_at
            FROM pdf_chat_history
            WHERE session_id=$1
            ORDER BY created_at ASC
            """,
            session_id,
        )
        return [dict(r) for r in rows]


# ── Saved papers ──────────────────────────────────────────────────────────────

async def save_paper(paper: dict) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO saved_papers
                (paper_id, title, authors, year, abstract, citation_count,
                 url, open_access_url, journal, discipline_tag, doi)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
            ON CONFLICT (paper_id) DO UPDATE
                SET title          = EXCLUDED.title,
                    citation_count = EXCLUDED.citation_count,
                    saved_at       = NOW()
            RETURNING id
            """,
            paper.get("paper_id"),
            paper.get("title", "Untitled"),
            json.dumps(paper.get("authors", [])),
            paper.get("year"),
            paper.get("abstract"),
            paper.get("citation_count", 0),
            paper.get("url"),
            paper.get("open_access_url"),
            paper.get("journal"),
            paper.get("discipline_tag"),
            paper.get("doi"),
        )
        return row["id"]


async def get_saved_papers(discipline: str | None = None, limit: int = 50) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        if discipline and discipline != "all":
            rows = await conn.fetch(
                "SELECT * FROM saved_papers WHERE discipline_tag=$1 ORDER BY saved_at DESC LIMIT $2",
                discipline, limit,
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM saved_papers ORDER BY saved_at DESC LIMIT $1", limit
            )
        result = []
        for r in rows:
            d = dict(r)
            d["authors"] = json.loads(d["authors"]) if isinstance(d["authors"], str) else d["authors"]
            result.append(d)
        return result


async def delete_saved_paper(paper_id: str) -> bool:
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM saved_papers WHERE paper_id=$1", paper_id
        )
        return result == "DELETE 1"


# ── Citation collections ──────────────────────────────────────────────────────

async def save_citation(
    title: str,
    formatted: str,
    style: str,
    authors: list | None = None,
    year: int | None = None,
    journal: str | None = None,
    volume: str | None = None,
    pages: str | None = None,
    doi: str | None = None,
    paper_id: str | None = None,
) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO citations
                (paper_id, title, authors, year, journal, volume, pages, doi, style, formatted)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
            RETURNING id
            """,
            paper_id, title,
            json.dumps(authors or []),
            year, journal, volume, pages, doi, style, formatted,
        )
        return row["id"]


async def get_citations(style: str | None = None, limit: int = 100) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        if style:
            rows = await conn.fetch(
                "SELECT * FROM citations WHERE style=$1 ORDER BY saved_at DESC LIMIT $2",
                style, limit,
            )
        else:
            rows = await conn.fetch(
                "SELECT * FROM citations ORDER BY saved_at DESC LIMIT $1", limit
            )
        result = []
        for r in rows:
            d = dict(r)
            d["authors"] = json.loads(d["authors"]) if isinstance(d["authors"], str) else d["authors"]
            result.append(d)
        return result


async def delete_citation(citation_id: int) -> bool:
    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM citations WHERE id=$1", citation_id
        )
        return result == "DELETE 1"
