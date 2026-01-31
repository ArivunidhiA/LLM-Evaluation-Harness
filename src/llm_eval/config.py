"""Configuration management for the evaluation harness."""

from __future__ import annotations

from pydantic import Field, field_serializer
from pydantic_settings import BaseSettings, SettingsConfigDict


def _redact_key(value: str) -> str:
    """Redact API key for logging/serialization; never expose full key."""
    return "***" if value else ""


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Sensitive fields (e.g. openai_api_key) are never logged or serialized in full.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    openai_api_key: str = Field(default="", description="OpenAI API key (never logged)")
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

    @field_serializer("openai_api_key")
    def _redact_api_key(self, value: str) -> str:
        """Redact API key in model_dump() / JSON so it is never logged or exposed."""
        return _redact_key(value) if value else ""

    def __repr__(self) -> str:
        """Avoid exposing secrets in repr()."""
        return f"Settings(log_level={self.log_level!r}, default_model={self.default_model!r}, ...)"


settings = Settings()
