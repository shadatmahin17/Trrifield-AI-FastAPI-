"""
Run once at startup — creates all tables if they don't exist.
Safe to call repeatedly (uses IF NOT EXISTS).
"""
import logging
from db.connection import get_pool

logger = logging.getLogger(__name__)

SCHEMA = """
-- ── Search history ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS search_history (
    id            BIGSERIAL PRIMARY KEY,
    query         TEXT        NOT NULL,
    discipline    TEXT        NOT NULL DEFAULT 'all',
    result_count  INT         NOT NULL DEFAULT 0,
    latency_ms    FLOAT       NOT NULL DEFAULT 0,
    intent        TEXT        NOT NULL DEFAULT 'general',
    success       BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_search_history_created_at  ON search_history (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_search_history_query       ON search_history USING gin(to_tsvector('english', query));
CREATE INDEX IF NOT EXISTS idx_search_history_discipline  ON search_history (discipline);

-- ── Analytics events (aggregated counters) ──────────────────────────────────
CREATE TABLE IF NOT EXISTS analytics_events (
    id            BIGSERIAL PRIMARY KEY,
    event_type    TEXT        NOT NULL,   -- 'search' | 'pdf_upload' | 'citation' | 'copilot'
    discipline    TEXT,
    intent        TEXT,
    latency_ms    FLOAT,
    success       BOOLEAN     NOT NULL DEFAULT TRUE,
    meta          JSONB,                  -- flexible extra fields
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analytics_events_type       ON analytics_events (event_type);
CREATE INDEX IF NOT EXISTS idx_analytics_events_created_at ON analytics_events (created_at DESC);

-- ── PDF sessions ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pdf_sessions (
    session_id    TEXT        PRIMARY KEY,
    filename      TEXT        NOT NULL,
    file_size_mb  FLOAT,
    chunk_count   INT         NOT NULL DEFAULT 0,
    latency_ms    FLOAT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pdf_sessions_created_at ON pdf_sessions (created_at DESC);

-- ── PDF chat history (per session) ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pdf_chat_history (
    id            BIGSERIAL PRIMARY KEY,
    session_id    TEXT        NOT NULL REFERENCES pdf_sessions(session_id) ON DELETE CASCADE,
    role          TEXT        NOT NULL,   -- 'user' | 'assistant'
    content       TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pdf_chat_session ON pdf_chat_history (session_id, created_at ASC);

-- ── Saved papers ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS saved_papers (
    id            BIGSERIAL PRIMARY KEY,
    paper_id      TEXT        NOT NULL,
    title         TEXT        NOT NULL,
    authors       JSONB       NOT NULL DEFAULT '[]',
    year          INT,
    abstract      TEXT,
    citation_count INT        NOT NULL DEFAULT 0,
    url           TEXT,
    open_access_url TEXT,
    journal       TEXT,
    discipline_tag TEXT,
    doi           TEXT,
    saved_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_saved_papers_paper_id ON saved_papers (paper_id);
CREATE INDEX IF NOT EXISTS idx_saved_papers_saved_at        ON saved_papers (saved_at DESC);
CREATE INDEX IF NOT EXISTS idx_saved_papers_discipline      ON saved_papers (discipline_tag);

-- ── Citation collections ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS citations (
    id            BIGSERIAL PRIMARY KEY,
    paper_id      TEXT,
    title         TEXT        NOT NULL,
    authors       JSONB       NOT NULL DEFAULT '[]',
    year          INT,
    journal       TEXT,
    volume        TEXT,
    pages         TEXT,
    doi           TEXT,
    style         TEXT        NOT NULL DEFAULT 'apa',
    formatted     TEXT        NOT NULL,
    saved_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_citations_saved_at  ON citations (saved_at DESC);
CREATE INDEX IF NOT EXISTS idx_citations_style     ON citations (style);
"""


async def run_migrations():
    """Create all tables. Called once at app startup."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA)
        logger.info("Database migrations complete")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


async def run_library_migrations():
    """Library-specific tables. Called at startup alongside run_migrations()."""
    LIBRARY_SCHEMA = """
    -- ── Library papers ──────────────────────────────────────────────────────────
    CREATE TABLE IF NOT EXISTS library_papers (
        id            BIGSERIAL    PRIMARY KEY,
        session_id    TEXT         NOT NULL UNIQUE,
        filename      TEXT         NOT NULL,
        title         TEXT,
        authors       JSONB        NOT NULL DEFAULT '[]',
        abstract      TEXT,
        doi           TEXT,
        journal       TEXT,
        year          INT,
        discipline    TEXT         NOT NULL DEFAULT 'general',
        file_size_mb  FLOAT,
        r2_key        TEXT,                     -- Cloudflare R2 object key
        r2_url        TEXT,                     -- public URL
        chunk_count   INT          NOT NULL DEFAULT 0,
        uploaded_by   TEXT         NOT NULL DEFAULT 'anonymous',
        is_public     BOOLEAN      NOT NULL DEFAULT TRUE,
        view_count    INT          NOT NULL DEFAULT 0,
        created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        last_accessed TIMESTAMPTZ  NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_lib_created_at  ON library_papers (created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_lib_discipline  ON library_papers (discipline);
    CREATE INDEX IF NOT EXISTS idx_lib_session     ON library_papers (session_id);
    CREATE INDEX IF NOT EXISTS idx_lib_is_public   ON library_papers (is_public);
    CREATE INDEX IF NOT EXISTS idx_lib_title       ON library_papers
        USING gin(to_tsvector('english', COALESCE(title,'')));

    -- ── Extracted column cache ────────────────────────────────────────────────
    -- Columns extracted on-demand and cached so we don't re-run LLM every time
    CREATE TABLE IF NOT EXISTS library_columns (
        id            BIGSERIAL    PRIMARY KEY,
        paper_id      BIGINT       NOT NULL REFERENCES library_papers(id) ON DELETE CASCADE,
        column_key    TEXT         NOT NULL,   -- e.g. 'tldr', 'results', 'methods_used'
        content       TEXT         NOT NULL,
        extracted_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
        UNIQUE (paper_id, column_key)
    );

    CREATE INDEX IF NOT EXISTS idx_lib_col_paper ON library_columns (paper_id);
    CREATE INDEX IF NOT EXISTS idx_lib_col_key   ON library_columns (column_key);
    """
    try:
        from db.connection import get_pool
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(LIBRARY_SCHEMA)
        import logging
        logging.getLogger(__name__).info("Library migrations complete")
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Library migration failed: {e}")
        raise
