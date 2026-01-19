"""Pydantic schemas for evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """Definition of a single test case."""

    test_id: str
    suite: str
    prompt_template: str
    input_variables: Dict[str, Any]
    expected_output: Optional[str]
    evaluation_metrics: List[str]
    thresholds: Dict[str, float]
    tags: List[str]
    version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvaluationRunConfig(BaseModel):
    """Runtime configuration for a single evaluation run."""

    suite: str
    baseline_version: Optional[str] = None
    model: str
    max_concurrency: int
    timeout_seconds: int
    metrics: List[str]


class MetricResult(BaseModel):
    """Result for a single metric."""

    name: str
    score: float
    details: Optional[Dict[str, Any]] = None


class TestResult(BaseModel):
    """Result for a single test case."""

    test_case_id: str
    prompt: str
    output: str
    metrics: List[MetricResult]
    duration_ms: int
    tokens_used: int


class EvaluationRunResult(BaseModel):
    """Aggregated run output for reporting and storage."""

    run_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    config: EvaluationRunConfig
    results: List[TestResult]
