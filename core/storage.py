"""
Cloudflare R2 storage client (S3-compatible API).
Stores uploaded PDFs permanently so the Library can serve them back.

Required env vars:
  R2_ACCOUNT_ID        — Cloudflare account ID
  R2_ACCESS_KEY_ID     — R2 API token access key
  R2_SECRET_ACCESS_KEY — R2 API token secret
  R2_BUCKET_NAME       — bucket name (e.g. trifield-papers)
  R2_PUBLIC_URL        — public URL prefix (e.g. https://pub-xxx.r2.dev)
"""
import logging
from functools import lru_cache
from core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def _get_r2_client():
    import boto3
    s = get_settings()
    if not all([s.r2_account_id, s.r2_access_key_id, s.r2_secret_access_key]):
        raise RuntimeError(
            "R2 not configured. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, "
            "R2_SECRET_ACCESS_KEY in Railway env vars."
        )
    return boto3.client(
        "s3",
        endpoint_url=f"https://{s.r2_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=s.r2_access_key_id,
        aws_secret_access_key=s.r2_secret_access_key,
        region_name="auto",
    )


def r2_enabled() -> bool:
    s = get_settings()
    return bool(s.r2_account_id and s.r2_access_key_id and s.r2_secret_access_key)


async def upload_pdf_to_r2(file_bytes: bytes, key: str, filename: str) -> str:
    """
    Upload PDF bytes to R2. Returns the public URL.
    key format: "papers/{session_id}/{filename}"
    """
    s = get_settings()
    client = _get_r2_client()
    client.put_object(
        Bucket=s.r2_bucket_name,
        Key=key,
        Body=file_bytes,
        ContentType="application/pdf",
        ContentDisposition=f'inline; filename="{filename}"',
    )
    url = f"{s.r2_public_url.rstrip('/')}/{key}"
    logger.info(f"R2 upload: {key} → {url}")
    return url


async def delete_pdf_from_r2(key: str):
    """Delete a PDF from R2."""
    s = get_settings()
    try:
        _get_r2_client().delete_object(Bucket=s.r2_bucket_name, Key=key)
        logger.info(f"R2 delete: {key}")
    except Exception as e:
        logger.warning(f"R2 delete failed for {key}: {e}")


async def get_presigned_url(key: str, expires_in: int = 3600) -> str:
    """Generate a pre-signed URL for direct browser download."""
    s = get_settings()
    client = _get_r2_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": s.r2_bucket_name, "Key": key},
        ExpiresIn=expires_in,
    )
