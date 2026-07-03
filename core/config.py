from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    tavily_api_key: Optional[str] = None
    langsmith_api_key: Optional[str] = None

    # Model provider and model selection
    llm_provider: Literal["google", "openai", "bedrock"] = "google"
    llm_model: str = "gemini-2.0-flash"

    embedding_provider: Literal["google", "openai", "bedrock"] = "google"
    embedding_model: str = "models/text-embedding-004"

    vision_provider: Literal["google", "openai", "bedrock"] = "google"
    vision_model: str = "gemini-2.0-flash"

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    chroma_persist_dir: Path = project_root / "data" / "chroma"
    uploads_dir: Path = project_root / "data" / "uploads"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    top_k: int = 8
    rerank_top_k: int = 4

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-base"

    # Application
    app_name: str = "PDF Q&A Bot"
    debug: bool = False

    def ensure_dirs(self) -> None:
        """Create required directories if they do not exist."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
