from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # An empty value (e.g. "LLM_REASONING_EFFORT=") means "not set".
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_parse_none_str=""
    )

    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"

    database_url: str

    llm_provider: Literal["openai", "fake"] = "openai"
    llm_model: str = "gpt-5.6-luna"
    # Model for the offline, cached work: the ingestion profile and catalogue,
    # and rule compilation. Empty = LLM_MODEL. A stronger model here costs more
    # once per document and port, and makes compiled rules more reliable.
    llm_compile_model: str | None = None
    openai_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    # Must match the vector(n) column created by the first migration.
    embedding_dimensions: int = Field(default=1536, gt=0)
    # None omits the parameter, for models that only accept their default.
    llm_temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    # Only for reasoning models; None leaves the model's default.
    llm_reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    # Extracting a large rule (many reductions and surcharges) can take over a minute.
    llm_timeout_seconds: int = Field(default=120, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    llm_max_concurrency: int = Field(default=8, gt=0)
    # Answer repeated runtime and ingestion calls from the database (app/llm/cache.py).
    llm_response_cache: bool = True

    agent_max_tool_calls: int = Field(default=8, gt=0)
    agent_max_revisions: int = Field(default=1, ge=0)
    agent_critic_on_cache_hit: bool = False
    # Facts per fact-resolution call: bigger batches cost fewer tokens, smaller answer sooner.
    agent_facts_per_batch: int = Field(default=40, gt=0)
    # A prompt change normally means recompiling every cached rule (minutes and
    # about a million tokens per port). When true, rules compiled with older
    # prompts (same rule schema) stay in use until recompiled with refresh.
    rules_reuse_older_prompts: bool = True
    calculation_timeout_seconds: int = Field(default=300, gt=0)

    retrieval_top_k: int = Field(default=8, gt=0)
    retrieval_rrf_k: int = Field(default=60, gt=0)

    parser_vision_fallback: bool = False
    tariffs_dir: str = "./data/tariffs"
    # Exported rulebooks loaded at startup, so a fresh database needn't recompile.
    rulebooks_dir: str | None = "./data/rulebooks"
    # Example requests shown in Swagger; kept outside app/ (they name real ports).
    examples_dir: str = "./examples"
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
