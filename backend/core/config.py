"""Application configuration loaded from environment variables."""

from functools import lru_cache
from typing import ClassVar
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the Indus-Guardian backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Core app
    app_name: str = "Indus-Guardian API"
    app_env: str = "dev"

    # Database
    sqlite_path: str = "backend/data/indus_guardian.db"

   # Google AI Studio (Cloud LLM)
    google_api_key: str = Field(default=os.getenv("GOOGLE_API_KEY", ""))
    gemini_model: str = Field(default="gemini-3-flash-preview")
    embedding_model: str = Field(default="models/gemini-embedding-2")

    # Vector DB (Pinecone)
    pinecone_api_key: str = Field(default="")
    pinecone_index_name: str = "indus-guardian-index"

    # Web search fallback
    tavily_api_key: str = Field(default="")

    # Performance tuning
    semantic_cache_ttl_seconds: int = 3600
    top_k_retrieval: int = 20
    top_k_after_rerank: int = 5
    request_timeout_seconds: int = 120
    max_uploaded_file_mb: int = 25

    # System paths
    log_dir: str = "backend/logs"
    log_level: str = "INFO"
    backup_dir: str = "backend/backups"

    # Embedding model
    # FIXED: Added explicit type annotation to satisfy Pydantic V2 requirements
    embedding_model_name: str = "gemini-embedding-2"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached config instance."""
    return Settings()