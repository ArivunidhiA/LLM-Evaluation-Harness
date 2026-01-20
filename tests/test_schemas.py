"""Tests for Pydantic schemas."""

from datetime import datetime

import pytest

from llm_eval.schemas import TestCase, MetricResult, EvaluationRunConfig


class TestTestCase:
    """Tests for TestCase schema."""

    def test_create_test_case(self, sample_test_case):
        """Test creating a valid test case."""
        tc = TestCase(**sample_test_case)
        assert tc.test_id == "test_001"
        assert tc.suite == "qa"
        assert tc.input_variables["question"] == "What is 2+2?"

    def test_test_case_defaults(self, sample_test_case):
        """Test that created_at has a default value."""
        tc = TestCase(**sample_test_case)
        assert tc.created_at is not None
        assert isinstance(tc.created_at, datetime)


class TestMetricResult:
    """Tests for MetricResult schema."""

    def test_create_metric_result(self):
        """Test creating a metric result."""
        result = MetricResult(name="exact_match", score=1.0)
        assert result.name == "exact_match"
        assert result.score == 1.0
        assert result.details is None

    def test_metric_result_with_details(self):
        """Test metric result with details."""
        result = MetricResult(
            name="semantic_similarity",
            score=0.92,
            details={"model": "text-embedding-3-small"},
        )
        assert result.details["model"] == "text-embedding-3-small"


class TestEvaluationRunConfig:
    """Tests for EvaluationRunConfig schema."""

    def test_create_run_config(self):
        """Test creating an evaluation run config."""
        config = EvaluationRunConfig(
            suite="summarization",
            model="gpt-4-turbo",
            max_concurrency=50,
            timeout_seconds=120,
            metrics=["bleu", "rouge"],
        )
        assert config.suite == "summarization"
        assert config.baseline_version is None
        assert len(config.metrics) == 2
