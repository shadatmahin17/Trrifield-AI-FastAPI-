"""
API key authentication dependency.

Usage — protect any endpoint:
    from core.auth import require_api_key
    @router.get("/secret", dependencies=[Depends(require_api_key)])

Or protect a whole router in main.py:
    app.include_router(router, dependencies=[Depends(require_api_key)])

Key is read from the request in this order (standard conventions):
  1. Header:      X-API-Key: <key>
  2. Header:      Authorization: Bearer <key>
  3. Query param: ?api_key=<key>

If API_KEY is not set in env, all requests pass through (open mode).
This lets you develop locally without auth, and enable it on Railway
just by setting the env var — no code change needed.
"""
import secrets
import logging
from fastapi import HTTPException, Security, Request
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from core.config import get_settings

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_scheme  = HTTPBearer(auto_error=False)


async def require_api_key(
    request:     Request,
    x_api_key:   str | None = Security(_api_key_header),
    bearer:      HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
):
    """
    FastAPI dependency — call this to protect an endpoint or router.
    Raises HTTP 401 if the key is wrong; passes through if API_KEY is not configured.
    """
    expected = get_settings().api_key
    if not expected:
        # Auth disabled — open access (dev mode or internal network)
        return None

    # Extract key from whichever method was used
    provided = (
        x_api_key
        or (bearer.credentials if bearer else None)
        or request.query_params.get("api_key")
    )

    if not provided:
        raise HTTPException(
            status_code=401,
            detail="API key required. Pass it via X-API-Key header, "
                   "Authorization: Bearer <key>, or ?api_key=<key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(provided.strip(), expected.strip()):
        logger.warning(f"Invalid API key attempt from {request.client.host if request.client else 'unknown'}")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return provided


async def optional_api_key(
    request:   Request,
    x_api_key: str | None = Security(_api_key_header),
    bearer:    HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> str | None:
    """
    Soft auth — returns the key if valid, None if no key is configured.
    Use for endpoints that work both authenticated and unauthenticated
    but may return different data (e.g. rate-limited vs full results).
    """
    expected = get_settings().api_key
    if not expected:
        return None
    try:
        return await require_api_key(request, x_api_key, bearer)
    except HTTPException:
        raise
