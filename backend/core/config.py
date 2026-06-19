"""Application configuration loaded from environment variables."""

import os
from functools import lru_cache
from typing import Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Runtime settings for the Indus-Guardian backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False 
    )

    app_name: str = "Indus-Guardian API"
    app_env: str = "dev"
    sqlite_path: str = "data/indus_guardian.db"
    google_api_key: str = "" 
    
    # Using the standard, highly gemini-3.1-flash-lite to avoid 429 and 404 errors
    gemini_model: str = "gemini-3.1-flash-lite"
    
    pinecone_api_key: str = ""
    pinecone_index_name: str = "indus-guardian-index"
    tavily_api_key: str = ""

    semantic_cache_ttl_seconds: int = 3600
    top_k_retrieval: int = 20
    top_k_after_rerank: int = 5
    request_timeout_seconds: int = 120
    max_uploaded_file_mb: int = 25

    log_dir: str = "logs"
    log_level: str = "INFO"
    backup_dir: str = "backups"
    embedding_model_name: str = "models/gemini-embedding-2"

    def model_post_init(self, __context: Any) -> None:
        """
        CRITICAL FIX: Forces the API key into the OS environment.
        This prevents the 401 UNAUTHENTICATED crash when LangChain changes contexts
        to the Web Search node.
        """
        if self.google_api_key:
            os.environ["GOOGLE_API_KEY"] = self.google_api_key

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached config instance."""
    return Settings()