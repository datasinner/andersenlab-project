from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"

    database_url: str

    llm_provider: Literal["openai", "fake"] = "openai"
    llm_model: str = "gpt-5.6-luna"
    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    # Must match the vector(n) column created by the first migration.
    embedding_dimensions: int = Field(default=1536, gt=0)
    llm_temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_seconds: int = Field(default=60, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    llm_max_concurrency: int = Field(default=8, gt=0)

    agent_max_tool_calls: int = Field(default=8, gt=0)
    agent_max_revisions: int = Field(default=2, ge=0)
    agent_critic_on_cache_hit: bool = False
    calculation_timeout_seconds: int = Field(default=300, gt=0)

    retrieval_top_k: int = Field(default=8, gt=0)
    retrieval_rrf_k: int = Field(default=60, gt=0)

    parser_vision_fallback: bool = False
    tariffs_dir: str = "./data/tariffs"
    max_upload_mb: int = Field(default=25, gt=0)

    api_auth_key: str | None = None

    langfuse_tracing_enabled: bool = False
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    langfuse_capture_content: bool = False

    uvicorn_workers: int = Field(default=2, gt=0)


settings = Settings()
