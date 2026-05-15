"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Runtime settings for the Indus-Guardian backend."""

    # Using Pydantic's built-in env loading
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False # Allows GOOGLE_API_KEY to map to google_api_key
    )

    # Core app
    app_name: str = "Indus-Guardian API"
    app_env: str = "dev"

    # Database
    sqlite_path: str = "data/indus_guardian.db"

    # Google AI Studio (Cloud LLM)
    # Senior Tip: Don't use os.getenv here, let BaseSettings handle it from .env
    google_api_key: str = "" 
    gemini_model: str = "gemini-3-flash-preview"
    
    # Vector DB (Pinecone)
    pinecone_api_key: str = ""
    pinecone_index_name: str = "indus-guardian-index"

    # Web search fallback
    tavily_api_key: str = ""

    # Performance tuning
    semantic_cache_ttl_seconds: int = 3600
    top_k_retrieval: int = 20
    top_k_after_rerank: int = 5
    request_timeout_seconds: int = 120
    max_uploaded_file_mb: int = 25

    # System paths
    log_dir: str = "logs"
    log_level: str = "INFO"
    backup_dir: str = "backups"

    # Embedding model
    # Matches the latest Google GenAI embedding model
    embedding_model_name: str = "models/gemini-embedding-2"

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached config instance."""
    return Settings()