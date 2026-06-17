from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_cors_allowed_origins() -> list[str]:
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="rag-backend", alias="APP_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/rag_backend",
        alias="DATABASE_URL",
    )
    alembic_database_url: str | None = Field(default=None, alias="ALEMBIC_DATABASE_URL")
    jwt_secret_key: str = Field(default="change-me", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(
        default=60,
        alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    local_storage_path: Path = Field(default=Path("./data/documents"), alias="LOCAL_STORAGE_PATH")
    cors_allowed_origins: list[str] = Field(
        default_factory=_default_cors_allowed_origins,
        alias="CORS_ALLOWED_ORIGINS",
    )
    max_upload_size_mb: int = Field(default=10, alias="MAX_UPLOAD_SIZE_MB")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_embedding_model: str = Field(
        default="models/text-embedding-004",
        alias="GEMINI_EMBEDDING_MODEL",
    )
    gemini_llm_model: str = Field(default="models/gemini-2.5-flash", alias="GEMINI_LLM_MODEL")
    rag_chunk_size: int = Field(default=1000, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=200, alias="RAG_CHUNK_OVERLAP")
    rag_search_top_k: int = Field(default=5, alias="RAG_SEARCH_TOP_K")
    rag_answer_max_context_chunks: int = Field(default=5, alias="RAG_ANSWER_MAX_CONTEXT_CHUNKS")
    rag_min_relevance_score: float = Field(default=0.30, alias="RAG_MIN_RELEVANCE_SCORE")
    rag_retrieval_mode: str = Field(default="hybrid", alias="RAG_RETRIEVAL_MODE")
    rag_vector_weight: float = Field(default=0.65, alias="RAG_VECTOR_WEIGHT")
    rag_keyword_weight: float = Field(default=0.35, alias="RAG_KEYWORD_WEIGHT")
    rag_vector_candidates: int = Field(default=20, alias="RAG_VECTOR_CANDIDATES")
    rag_keyword_candidates: int = Field(default=20, alias="RAG_KEYWORD_CANDIDATES")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("rag_retrieval_mode")
    @classmethod
    def validate_rag_retrieval_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"vector", "keyword", "hybrid"}:
            raise ValueError("RAG_RETRIEVAL_MODE must be vector, keyword, or hybrid")
        return normalized

    @model_validator(mode="after")
    def validate_security_settings(self) -> "Settings":
        if self.environment != "development" and self.jwt_secret_key == "change-me":
            raise ValueError(
                "JWT_SECRET_KEY cannot be 'change-me' outside development. Set a secure secret."
            )
        if not 0 <= self.rag_vector_weight <= 1:
            raise ValueError("RAG_VECTOR_WEIGHT must be between 0 and 1")
        if not 0 <= self.rag_keyword_weight <= 1:
            raise ValueError("RAG_KEYWORD_WEIGHT must be between 0 and 1")
        if self.rag_retrieval_mode == "hybrid" and (
            self.rag_vector_weight == 0 and self.rag_keyword_weight == 0
        ):
            raise ValueError("At least one RAG retrieval weight must be positive")
        if self.rag_vector_candidates < 1:
            raise ValueError("RAG_VECTOR_CANDIDATES must be at least 1")
        if self.rag_keyword_candidates < 1:
            raise ValueError("RAG_KEYWORD_CANDIDATES must be at least 1")
        return self

    @property
    def effective_alembic_database_url(self) -> str:
        return self.alembic_database_url or self.database_url

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
