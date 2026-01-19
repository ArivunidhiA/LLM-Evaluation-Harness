"""Configuration management for the evaluation harness."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    openai_api_key: str = ""
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/llm_eval"
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"
    max_concurrent_requests: int = 50
    default_model: str = "gpt-4-turbo"
    evaluation_timeout_seconds: int = 120
    retry_attempts: int = 3
    request_timeout_seconds: int = 60
    rate_limit_tpm: int = 10_000
    rate_limit_rpm: int = 500


settings = Settings()
