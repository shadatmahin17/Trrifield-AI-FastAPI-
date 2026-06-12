"""
Health endpoint — always public (no auth), used by Railway uptime checks.
Probes all three backing services: Qdrant, PostgreSQL, LLM keys.
"""
import asyncio
import logging
from datetime import datetime
from fastapi import APIRouter
from core.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)


async def _check_qdrant() -> dict:
    try:
        from vectorstore.qdrant_store import get_store
        get_store()._get_client().get_collections()
        return {"status": "reachable", "error": None}
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


async def _check_postgres() -> dict:
    s = get_settings()
    if not s.database_url:
        return {"status": "not_configured", "error": "DATABASE_URL not set"}
    try:
        from db.connection import get_pool
        pool = await get_pool()
        async with pool.acquire() as conn:
            # Lightweight probe — returns server version and table counts
            version = await conn.fetchval("SELECT version()")
            tables  = await conn.fetch(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY tablename
                """
            )
            counts = {}
            for t in tables:
                name = t["tablename"]
                try:
                    n = await conn.fetchval(f"SELECT COUNT(*) FROM {name}")
                    counts[name] = n
                except Exception:
                    counts[name] = "?"
        return {
            "status":  "reachable",
            "version": version.split(",")[0] if version else "unknown",
            "tables":  counts,
            "error":   None,
        }
    except Exception as e:
        logger.warning(f"Postgres health check failed: {e}")
        return {"status": "unreachable", "error": str(e)}


async def _check_llm_keys() -> dict:
    s = get_settings()
    return {
        "anthropic": "set"   if s.anthropic_api_key else "missing",
        "groq":      "set"   if s.groq_api_key      else "missing",
        "routing":   "task-aware (Claude=quality, Groq=speed)",
    }


@router.get("/health")
async def health():
    s = get_settings()
    has_qdrant_cloud = bool(s.qdrant_url and s.qdrant_api_key)

    # Run all three probes concurrently
    qdrant_result, pg_result, llm_result = await asyncio.gather(
        _check_qdrant(),
        _check_postgres(),
        _check_llm_keys(),
        return_exceptions=True,
    )

    # If any gather item threw (shouldn't happen — we catch inside), degrade gracefully
    if isinstance(qdrant_result, Exception):
        qdrant_result = {"status": "error", "error": str(qdrant_result)}
    if isinstance(pg_result, Exception):
        pg_result = {"status": "error", "error": str(pg_result)}
    if isinstance(llm_result, Exception):
        llm_result = {"anthropic": "unknown", "groq": "unknown"}

    # Overall status — degraded if any service is unreachable
    services_ok = (
        qdrant_result["status"] == "reachable"
        and pg_result["status"] in ("reachable", "not_configured")
        and (llm_result["anthropic"] == "set" or llm_result["groq"] == "set")
    )
    overall = "healthy" if services_ok else "degraded"

    return {
        "status":      overall,
        "timestamp":   datetime.utcnow().isoformat(),
        "version":     "2.0.0",
        "platform":    "TriField AI",
        "auth":        "enabled" if s.api_key else "disabled",
        "disciplines": ["Aerospace", "Materials Science", "Textile Engineering"],
        "services": {
            "llm":      llm_result,
            "qdrant": {
                "engine": "Qdrant Cloud" if has_qdrant_cloud else "Qdrant Local",
                **qdrant_result,
            },
            "postgres": pg_result,
        },
        "features": {
            "search":           "OpenAlex + Crossref + arXiv + PubMed + Unpaywall",
            "query_rewriting":  "LLM (Groq) + rule-based fallback",
            "paper_scoring":    "weighted: relevance 40% + citations 25% + recency 15% + journal 10% + OA 10%",
            "streaming_search": "SSE — live source progress",
            "pdf_rag":          "Qdrant semantic retrieval + Claude",
            "copilot":          "Research gaps + trends + experiments",
            "citations":        "APA, IEEE, AIAA, Harvard, MLA, Chicago",
            "analytics":        "search latency, top queries, success rate",
            "persistence":      "PostgreSQL" if s.database_url else "in-memory only",
        },
    }


@router.get("/health/db")
async def health_db():
    """Detailed PostgreSQL health — table row counts, version, connection pool."""
    result = await _check_postgres()
    if result["status"] == "not_configured":
        from fastapi import HTTPException
        raise HTTPException(503, "DATABASE_URL not configured")
    return result


@router.get("/health/ready")
async def readiness():
    """
    Minimal readiness probe for Railway / k8s.
    Returns 200 if the app can serve requests, 503 if critically broken.
    """
    s = get_settings()
    has_llm = bool(s.anthropic_api_key or s.groq_api_key)
    if not has_llm:
        from fastapi import HTTPException
        raise HTTPException(503, "No LLM keys configured — ANTHROPIC_API_KEY or GROQ_API_KEY required")
    return {"ready": True}
