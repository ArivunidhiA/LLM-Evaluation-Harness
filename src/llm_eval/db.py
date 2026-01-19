"""Database session and engine setup."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from llm_eval.config import settings


def create_engine() -> AsyncEngine:
    """Create async SQLAlchemy engine."""

    return create_async_engine(str(settings.database_url), pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create session factory bound to engine."""

    return async_sessionmaker(engine, expire_on_commit=False)
