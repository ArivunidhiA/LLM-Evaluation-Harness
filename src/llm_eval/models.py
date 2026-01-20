"""SQLAlchemy ORM models."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base ORM class."""


class EvaluationRun(Base):
    """Evaluation run metadata."""

    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(64), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    config_json: Mapped[dict] = mapped_column(JSON)

    results: Mapped[list["TestResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_evaluation_runs_started_at", "started_at"),)


class TestResult(Base):
    """Per-test result row."""

    __tablename__ = "test_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("evaluation_runs.id"), index=True)
    test_case_id: Mapped[str] = mapped_column(String(128), index=True)
    prompt: Mapped[str] = mapped_column(Text)
    output: Mapped[str] = mapped_column(Text)
    metrics_json: Mapped[dict] = mapped_column(JSON)
    duration_ms: Mapped[int] = mapped_column(Integer)
    tokens_used: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    run: Mapped["EvaluationRun"] = relationship(back_populates="results")

    __table_args__ = (Index("ix_test_results_run_test", "run_id", "test_case_id"),)


class Baseline(Base):
    """Baseline outputs for regression comparisons."""

    __tablename__ = "baselines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    suite: Mapped[str] = mapped_column(String(128), index=True)
    test_case_id: Mapped[str] = mapped_column(String(128), index=True)
    baseline_output: Mapped[str] = mapped_column(Text)
    baseline_metrics: Mapped[dict] = mapped_column(JSON)
    version: Mapped[str] = mapped_column(String(64), index=True)

    __table_args__ = (Index("ix_baselines_suite_version", "suite", "version"),)


class Regression(Base):
    """Regression analysis row."""

    __tablename__ = "regressions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    test_case_id: Mapped[str] = mapped_column(String(128))
    baseline_run_id: Mapped[str] = mapped_column(String(64))
    metric_name: Mapped[str] = mapped_column(String(128))
    baseline_score: Mapped[float] = mapped_column(Float)
    current_score: Mapped[float] = mapped_column(Float)
    diff: Mapped[float] = mapped_column(Float)

    __table_args__ = (Index("ix_regressions_run_metric", "run_id", "metric_name"),)
