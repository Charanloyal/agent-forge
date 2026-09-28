import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """
    Central application configuration for AgentForge.
    All attributes can be overridden via environment variables or .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # Application Meta
    APP_NAME: str = "AgentForge"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # PostgreSQL Persistence
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="agentforge")
    DATABASE_URL: str | None = None

    @field_validator("DATABASE_URL", mode="before")
    def assemble_db_url(cls, v: str | None, values) -> str:
        if v and len(v.strip()) > 0:
            url_str = v.strip()
            if url_str.startswith("postgres://"):
                url_str = url_str.replace("postgres://", "postgresql+asyncpg://", 1)
            elif url_str.startswith("postgresql://") and not url_str.startswith("postgresql+asyncpg://"):
                url_str = url_str.replace("postgresql://", "postgresql+asyncpg://", 1)
            return url_str
        data = values.data if hasattr(values, "data") else values
        user = data.get("POSTGRES_USER", "postgres")
        pwd = data.get("POSTGRES_PASSWORD", "postgres")
        host = data.get("POSTGRES_HOST", "localhost")
        port = data.get("POSTGRES_PORT", 5432)
        db = data.get("POSTGRES_DB", "agentforge")
        return f"postgresql+asyncpg://{user}:{pwd}@{host}:{port}/{db}"

    # Qdrant Vector DB
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)
    QDRANT_GRPC_PORT: int = Field(default=6334)
    QDRANT_COLLECTION: str = Field(default="agentforge_chunks")
    QDRANT_API_KEY: str | None = None

    # Sentence Transformers & Reranking Models
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval & Fusion Parameters
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 80
    TOP_K_DENSE: int = 15
    TOP_K_SPARSE: int = 15
    RRF_K: int = 60
    TOP_K_FUSED: int = 10
    TOP_K_RERANKED: int = 5

    # Agentic Evaluation & Gate Thresholds
    RERANKER_SCORE_THRESHOLD: float = 0.20
    RELEVANCE_PASS_THRESHOLD: float = 0.25
    FAITHFULNESS_PASS_THRESHOLD: float = 0.60
    MAX_REFLECTION_ITERATIONS: int = 2


@lru_cache()
def get_settings() -> Settings:
    return Settings()
