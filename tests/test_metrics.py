"""Tests for evaluation metrics."""

import pytest

from llm_eval.metrics import (
    AVAILABLE_METRICS,
    BLEUMetric,
    ExactMatchMetric,
    LatencyMetric,
    MetricsEngine,
    ROUGEMetric,
    ToxicityMetric,
)
from llm_eval.schemas import TestCase


@pytest.fixture
def test_case():
    """Create a test case for metric testing."""
    return TestCase(
        test_id="metric_test",
        suite="test",
        prompt_template="Test prompt",
        input_variables={},
        expected_output="The quick brown fox jumps over the lazy dog",
        evaluation_metrics=["exact_match"],
        thresholds={},
        tags=[],
        version="1.0",
    )


class TestAvailableMetrics:
    """Tests for available metrics list."""

    def test_all_metrics_listed(self):
        """Test that expected metrics are available."""
        assert "exact_match" in AVAILABLE_METRICS
        assert "semantic_similarity" in AVAILABLE_METRICS
        assert "llm_judge" in AVAILABLE_METRICS
        assert "bleu" in AVAILABLE_METRICS
        assert "rouge" in AVAILABLE_METRICS


class TestExactMatchMetric:
    """Tests for ExactMatchMetric."""

    @pytest.mark.asyncio
    async def test_exact_match_true(self, test_case):
        """Test exact match when outputs are identical."""
        metric = ExactMatchMetric()
        result = await metric.compute(
            test_case,
            "The quick brown fox jumps over the lazy dog",
            "prompt",
            100,
        )
        assert result.score == 1.0
        assert result.details["match"] is True

    @pytest.mark.asyncio
    async def test_exact_match_case_insensitive(self, test_case):
        """Test case-insensitive matching."""
        metric = ExactMatchMetric()
        result = await metric.compute(
            test_case,
            "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG",
            "prompt",
            100,
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_exact_match_false(self, test_case):
        """Test exact match when outputs differ."""
        metric = ExactMatchMetric()
        result = await metric.compute(
            test_case,
            "A different output",
            "prompt",
            100,
        )
        assert result.score == 0.0
        assert result.details["match"] is False


class TestLatencyMetric:
    """Tests for LatencyMetric."""

    @pytest.mark.asyncio
    async def test_latency_within_threshold(self, test_case):
        """Test latency within threshold."""
        test_case.thresholds["latency"] = 1000
        metric = LatencyMetric()
        result = await metric.compute(test_case, "output", "prompt", 500)
        assert result.score == 1.0
        assert result.details["passed"] is True

    @pytest.mark.asyncio
    async def test_latency_exceeds_threshold(self, test_case):
        """Test latency exceeding threshold."""
        test_case.thresholds["latency"] = 1000
        metric = LatencyMetric()
        result = await metric.compute(test_case, "output", "prompt", 2000)
        assert result.score < 1.0
        assert result.details["passed"] is False


class TestBLEUMetric:
    """Tests for BLEUMetric."""

    @pytest.mark.asyncio
    async def test_bleu_identical(self, test_case):
        """Test BLEU score for identical strings."""
        metric = BLEUMetric()
        result = await metric.compute(
            test_case,
            "The quick brown fox jumps over the lazy dog",
            "prompt",
            100,
        )
        assert result.score > 0.9

    @pytest.mark.asyncio
    async def test_bleu_different(self, test_case):
        """Test BLEU score for different strings."""
        metric = BLEUMetric()
        result = await metric.compute(
            test_case,
            "Completely different text here",
            "prompt",
            100,
        )
        assert result.score < 0.5


class TestROUGEMetric:
    """Tests for ROUGEMetric."""

    @pytest.mark.asyncio
    async def test_rouge_identical(self, test_case):
        """Test ROUGE-L score for identical strings."""
        metric = ROUGEMetric()
        result = await metric.compute(
            test_case,
            "The quick brown fox jumps over the lazy dog",
            "prompt",
            100,
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_rouge_partial_overlap(self, test_case):
        """Test ROUGE-L score for partial overlap."""
        metric = ROUGEMetric()
        result = await metric.compute(
            test_case,
            "The quick brown fox",
            "prompt",
            100,
        )
        assert 0.3 < result.score < 0.7


class TestToxicityMetric:
    """Tests for ToxicityMetric."""

    @pytest.mark.asyncio
    async def test_clean_output(self, test_case):
        """Test toxicity score for clean output."""
        metric = ToxicityMetric()
        result = await metric.compute(
            test_case,
            "This is a perfectly normal and friendly response.",
            "prompt",
            100,
        )
        assert result.score == 1.0
        assert result.details["is_clean"] is True

    @pytest.mark.asyncio
    async def test_toxic_output(self, test_case):
        """Test toxicity score for problematic output."""
        metric = ToxicityMetric()
        result = await metric.compute(
            test_case,
            "This response contains hate speech.",
            "prompt",
            100,
        )
        assert result.score < 1.0
        assert result.details["is_clean"] is False


class TestMetricsEngine:
    """Tests for MetricsEngine."""

    @pytest.mark.asyncio
    async def test_compute_multiple_metrics(self, test_case):
        """Test computing multiple metrics at once."""
        engine = MetricsEngine(["exact_match", "latency", "toxicity"])
        results = await engine.compute(
            test_case,
            "The quick brown fox jumps over the lazy dog",
            "prompt",
            500,
        )
        assert len(results) == 3
        names = [r.name for r in results]
        assert "exact_match" in names
        assert "latency" in names
        assert "toxicity" in names

    def test_summarize_results(self):
        """Test summarizing metric results."""
        from llm_eval.schemas import MetricResult

        results = [
            MetricResult(name="exact_match", score=1.0),
            MetricResult(name="latency", score=0.9),
        ]
        summary = MetricsEngine.summarize(results)
        assert summary["exact_match"] == 1.0
        assert summary["latency"] == 0.9
