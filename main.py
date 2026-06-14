import os, logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from routers import search, pdf, citations, health, copilot, analytics, library
from routers import papers
from core.auth import require_api_key

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from core.config import get_settings
    s = get_settings()
    if s.database_url:
        try:
            from db.migrations import run_migrations, run_library_migrations
            await run_migrations()
            await run_library_migrations()
            logger.info("PostgreSQL + Library tables ready")
        except Exception as e:
            logger.error(f"DB startup failed: {e}")
    else:
        logger.warning("DATABASE_URL not set — running without persistence")
    if s.api_key:  logger.info("API key auth ENABLED")
    else:          logger.warning("API_KEY not set — open access mode")
    from core.storage import r2_enabled
    if r2_enabled(): logger.info("Cloudflare R2 storage ENABLED")
    else:            logger.warning("R2 not configured — PDFs not persisted")
    yield
    try:
        from db.connection import close_pool
        await close_pool()
    except Exception:
        pass


app = FastAPI(
    title="TriField AI Backend",
    description="AI Research Workspace — Aerospace · Materials · Textile Engineering",
    version="3.0.0",
    lifespan=lifespan,
)

_raw = os.getenv("ALLOWED_ORIGINS", "")
ORIGINS = [o.strip() for o in _raw.split(",") if o.strip()]
if ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
else:
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"])

_auth = [Depends(require_api_key)]

app.include_router(health.router,                                          tags=["Health"])
app.include_router(search.router,    prefix="/api/search",    dependencies=_auth, tags=["Search"])
app.include_router(pdf.router,       prefix="/api/pdf",       dependencies=_auth, tags=["PDF"])
app.include_router(citations.router, prefix="/api/citations", dependencies=_auth, tags=["Citations"])
app.include_router(copilot.router,   prefix="/api/copilot",   dependencies=_auth, tags=["Copilot"])
app.include_router(analytics.router, prefix="/api/analytics", dependencies=_auth, tags=["Analytics"])
app.include_router(papers.router,    prefix="/api/saved-papers", dependencies=_auth, tags=["Saved Papers"])
app.include_router(library.router,   prefix="/api/library",   dependencies=_auth, tags=["Library"])


@app.get("/")
def root():
    from core.config import get_settings
    from core.storage import r2_enabled
    s = get_settings()
    return {
        "name": "TriField AI", "version": "3.0.0", "status": "running",
        "auth": "enabled" if s.api_key else "disabled",
        "storage": "Cloudflare R2" if r2_enabled() else "in-memory only",
        "endpoints": {
            "library":         "GET  /api/library/",
            "library_upload":  "POST /api/library/upload",
            "library_extract": "POST /api/library/{id}/extract/{column}",
            "library_columns": "GET  /api/library/columns",
            "search":          "GET  /api/search/?query=...",
            "pdf_upload":      "POST /api/pdf/upload",
            "pdf_chat":        "POST /api/pdf/chat",
            "analytics":       "GET  /api/analytics/",
            "docs":            "/docs",
        },
    }
