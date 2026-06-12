"""
Async PostgreSQL connection pool using asyncpg.
Railway injects DATABASE_URL automatically when you add a Postgres plugin.
"""
import asyncpg
import logging
from core.config import get_settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        s = get_settings()
        if not s.database_url:
            raise RuntimeError("DATABASE_URL not set. Add a Postgres plugin on Railway.")
        _pool = await asyncpg.create_pool(
            dsn=s.database_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
            # Railway Postgres requires SSL
            ssl="require",
        )
        logger.info("PostgreSQL pool created")
    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL pool closed")
