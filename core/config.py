from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = ""
    groq_api_key:      str = ""

    # Vector DB
    qdrant_url:        str = ""
    qdrant_api_key:    str = ""
    qdrant_local_path: str = "./qdrant_db"

    # PostgreSQL
    database_url:      str = ""

    # Cloudflare R2
    r2_account_id:        str = ""
    r2_access_key_id:     str = ""
    r2_secret_access_key: str = ""
    r2_bucket_name:       str = "trifield-papers"
    r2_public_url:        str = ""   # e.g. https://pub-xxx.r2.dev

    # API auth
    api_key:           str = ""

    # App
    app_env:           str = "development"
    max_pdf_size_mb:   int = 20
    chroma_path:       str = "./chroma_db"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
