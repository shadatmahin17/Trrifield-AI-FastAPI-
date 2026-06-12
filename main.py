import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from routers import search, pdf, citations, health, copilot, analytics
from routers import papers
from core.auth import require_api_key

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────────────────
    from core.config import get_settings
    s = get_settings()

    if s.database_url:
        try:
            from db.migrations import run_migrations
            await run_migrations()
            logger.info("PostgreSQL ready")
        except Exception as e:
            logger.error(f"DB startup failed (continuing without DB): {e}")
    else:
        logger.warning("DATABASE_URL not set — running without persistence")

    if s.api_key:
        logger.info("API key authentication ENABLED")
    else:
        logger.warning("API_KEY not set — running in open access mode")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    try:
        from db.connection import close_pool
        await close_pool()
    except Exception:
        pass


app = FastAPI(
    title="TriField AI Backend",
    description="AI Research Workspace v2 — Aerospace · Materials · Textile Engineering",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ── Auth dependency (applied to all /api/* routes) ────────────────────────
# /health and / are always public — no auth needed for uptime monitoring.
_auth = [Depends(require_api_key)]

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(health.router,                                                    tags=["Health"])
app.include_router(search.router,    prefix="/api/search",    dependencies=_auth,   tags=["Search"])
app.include_router(pdf.router,       prefix="/api/pdf",       dependencies=_auth,   tags=["PDF"])
app.include_router(citations.router, prefix="/api/citations", dependencies=_auth,   tags=["Citations"])
app.include_router(copilot.router,   prefix="/api/copilot",   dependencies=_auth,   tags=["Copilot"])
app.include_router(analytics.router, prefix="/api/analytics", dependencies=_auth,   tags=["Analytics"])
app.include_router(papers.router,    prefix="/api/saved-papers", dependencies=_auth, tags=["Saved Papers"])


@app.get("/")
def root():
    from core.config import get_settings
    auth_enabled = bool(get_settings().api_key)
    return {
        "name":         "TriField AI",
        "version":      "2.0.0",
        "status":       "running",
        "docs":         "/docs",
        "auth":         "enabled — pass X-API-Key header" if auth_enabled else "disabled (set API_KEY to enable)",
        "disciplines":  ["Aerospace", "Materials Science", "Textile Engineering"],
        "endpoints": {
            "search":          "GET  /api/search/?query=...",
            "search_stream":   "GET  /api/search/stream?query=...",
            "search_history":  "GET  /api/search/history",
            "analytics":       "GET  /api/analytics/",
            "recent_searches": "GET  /api/analytics/searches",
            "pdf_upload":      "POST /api/pdf/upload",
            "pdf_chat":        "POST /api/pdf/chat",
            "pdf_sessions":    "GET  /api/pdf/sessions",
            "saved_papers":    "GET  /api/saved-papers/",
            "save_paper":      "POST /api/saved-papers/",
            "citations":       "POST /api/citations/",
            "saved_citations": "GET  /api/citations/saved",
            "docs":            "/docs",
        },
    }
