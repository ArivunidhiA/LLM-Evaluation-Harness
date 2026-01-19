"""Evaluation metrics implementations (scaffold)."""

from __future__ import annotations

from typing import Any, Dict, List

from llm_eval.schemas import MetricResult, TestCase

AVAILABLE_METRICS = [
    "exact_match",
    "semantic_similarity",
    "llm_judge",
    "bleu",
    "rouge",
    "latency",
    "token_efficiency",
    "toxicity",
    "factual_consistency",
    "format_validation",
]


class MetricsEngine:
    """Compute metrics for a given output."""

    def __init__(self, enabled_metrics: List[str]) -> None:
        self.enabled_metrics = enabled_metrics

    async def compute(
        self, test_case: TestCase, output: str, prompt: str
    ) -> List[MetricResult]:
        results: List[MetricResult] = []
        for metric in self.enabled_metrics:
            results.append(
                MetricResult(
                    name=metric,
                    score=0.0,
                    details={"status": "not_implemented"},
                )
            )
        return results

    @staticmethod
    def summarize(results: List[MetricResult]) -> Dict[str, Any]:
        return {result.name: result.score for result in results}
