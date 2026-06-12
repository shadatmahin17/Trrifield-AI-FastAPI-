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

    # PostgreSQL — Railway injects DATABASE_URL automatically
    database_url:      str = ""

    # API key authentication
    # Set API_KEY in Railway env vars to enable auth on all /api/* routes.
    # Leave blank to run without auth (development / trusted network only).
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
