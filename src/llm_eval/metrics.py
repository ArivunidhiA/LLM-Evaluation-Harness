"""Evaluation metrics implementations."""

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

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


class BaseMetric(ABC):
    """Abstract base class for all metrics."""

    name: str

    @abstractmethod
    async def compute(
        self,
        test_case: TestCase,
        output: str,
        prompt: str,
        duration_ms: int,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute the metric score."""
        ...


class ExactMatchMetric(BaseMetric):
    """Exact string equality check (case-insensitive option)."""

    name = "exact_match"

    async def compute(
        self,
        test_case: TestCase,
        output: str,
        prompt: str,
        duration_ms: int,
        **kwargs: Any,
    ) -> MetricResult:
        expected = test_case.expected_output or ""
        case_insensitive = kwargs.get("case_insensitive", True)

        if case_insensitive:
            match = output.strip().lower() == expected.strip().lower()
        else:
            match = output.strip() == expected.strip()

        return MetricResult(
            name=self.name,
            score=1.0 if match else 0.0,
            details={"expected": expected, "actual": output[:200], "match": match},
        )


class SemanticSimilarityMetric(BaseMetric):
    """Cosine similarity using OpenAI embeddings."""

    name = "semantic_similarity"

    def __init__(self, client: AsyncOpenAI) -> None:
        self.client = client
        self.model = "text-embedding-3-small"

    async def compute(
        self,
        test_case: TestCase,
        output: str,
        prompt: str,
        duration_ms: int,
        **kwargs: Any,
    ) -> MetricResult:
        expected = test_case.expected_output
        if not expected:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": "No expected output for comparison"},
            )

        try:
            embeddings = await self.client.embeddings.create(
                model=self.model,
                input=[output, expected],
            )
            vec1 = embeddings.data[0].embedding
            vec2 = embeddings.data[1].embedding
            similarity = self._cosine_similarity(vec1, vec2)
            return MetricResult(
                name=self.name,
                score=similarity,
                details={"model": self.model},
            )
        except Exception as e:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"error": str(e)},
            )

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)


class LLMJudgeMetric(BaseMetric):
    """GPT-4 scores output on 1-10 scale with reasoning."""

    name = "llm_judge"

    def __init__(self, client: AsyncOpenAI) -> None:
        self.client = client

    async def compute(
        self,
        test_case: TestCase,
        output: str,
        prompt: str,
        duration_ms: int,
        **kwargs: Any,
    ) -> MetricResult:
        judge_prompt = f"""You are an expert evaluator. Score the following LLM output on a scale of 1-10.

PROMPT: {prompt}

EXPECTED OUTPUT: {test_case.expected_output or 'N/A'}

ACTUAL OUTPUT: {output}

Evaluate based on:
1. Accuracy and correctness
2. Completeness
3. Clarity and coherence
4. Relevance to the prompt

Respond in JSON format:
{{"score": <1-10>, "reasoning": "<brief explanation>"}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": judge_prompt}],
                timeout=60,
            )
            content = response.choices[0].message.content or "{}"
            # Parse JSON from response
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                score = float(result.get("score", 5)) / 10.0  # Normalize to 0-1
                return MetricResult(
                    name=self.name,
                    score=score,
                    details={
                        "raw_score": result.get("score"),
                        "reasoning": result.get("reasoning", ""),
                    },
                )
        except Exception as e:
            return MetricResult(
                name=self.name,
                score=0.5,
                details={"error": str(e)},
            )

        return MetricResult(name=self.name, score=0.5, details={"error": "Parse failed"})


class LatencyMetric(BaseMetric):
    """Check if response time is within threshold."""

    name = "latency"

    async def compute(
        self,
        test_case: TestCase,
        output: str,
        prompt: str,
        duration_ms: int,
        **kwargs: Any,
    ) -> MetricResult:
        threshold_ms = test_case.thresholds.get("latency", 5000)
        passed = duration_ms <= threshold_ms
        score = 1.0 if passed else max(0, 1 - (duration_ms - threshold_ms) / threshold_ms)

        return MetricResult(
            name=self.name,
            score=score,
            details={
                "duration_ms": duration_ms,
                "threshold_ms": threshold_ms,
                "passed": passed,
            },
        )


class FormatValidationMetric(BaseMetric):
    """Validate JSON/XML/Code syntax."""

    name = "format_validation"

    async def compute(
        self,
        test_case: TestCase,
        output: str,
        prompt: str,
        duration_ms: int,
        **kwargs: Any,
    ) -> MetricResult:
        expected_format = kwargs.get("format", "json")
        valid = False
        error_msg: Optional[str] = None

        if expected_format == "json":
            try:
                json.loads(output)
                valid = True
            except json.JSONDecodeError as e:
                error_msg = str(e)
        elif expected_format == "code":
            # Basic syntax check - could be extended
            valid = not any(
                err in output.lower()
                for err in ["syntax error", "undefined", "traceback"]
            )

        return MetricResult(
            name=self.name,
            score=1.0 if valid else 0.0,
            details={"format": expected_format, "valid": valid, "error": error_msg},
        )


class BLEUMetric(BaseMetric):
    """BLEU score for summarization tasks."""

    name = "bleu"

    async def compute(
        self,
        test_case: TestCase,
        output: str,
        prompt: str,
        duration_ms: int,
        **kwargs: Any,
    ) -> MetricResult:
        expected = test_case.expected_output or ""
        if not expected:
            return MetricResult(name=self.name, score=0.0, details={"error": "No reference"})

        # Simple n-gram BLEU approximation
        score = self._compute_bleu(output, expected)
        return MetricResult(name=self.name, score=score, details={"n_grams": 4})

    @staticmethod
    def _compute_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
        """Compute simplified BLEU score."""
        cand_tokens = candidate.lower().split()
        ref_tokens = reference.lower().split()

        if not cand_tokens or not ref_tokens:
            return 0.0

        precisions = []
        for n in range(1, max_n + 1):
            cand_ngrams = [
                tuple(cand_tokens[i : i + n]) for i in range(len(cand_tokens) - n + 1)
            ]
            ref_ngrams = set(
                tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
            )
            if not cand_ngrams:
                precisions.append(0.0)
                continue
            matches = sum(1 for ng in cand_ngrams if ng in ref_ngrams)
            precisions.append(matches / len(cand_ngrams))

        if not precisions or all(p == 0 for p in precisions):
            return 0.0

        # Geometric mean of precisions
        from math import exp, log

        log_precisions = [log(p) if p > 0 else -100 for p in precisions]
        return exp(sum(log_precisions) / len(log_precisions))


class ROUGEMetric(BaseMetric):
    """ROUGE-L score for summarization tasks."""

    name = "rouge"

    async def compute(
        self,
        test_case: TestCase,
        output: str,
        prompt: str,
        duration_ms: int,
        **kwargs: Any,
    ) -> MetricResult:
        expected = test_case.expected_output or ""
        if not expected:
            return MetricResult(name=self.name, score=0.0, details={"error": "No reference"})

        score = self._compute_rouge_l(output, expected)
        return MetricResult(name=self.name, score=score, details={"variant": "ROUGE-L"})

    @staticmethod
    def _compute_rouge_l(candidate: str, reference: str) -> float:
        """Compute ROUGE-L (longest common subsequence) F1 score."""
        cand_tokens = candidate.lower().split()
        ref_tokens = reference.lower().split()

        if not cand_tokens or not ref_tokens:
            return 0.0

        # LCS length
        m, n = len(cand_tokens), len(ref_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if cand_tokens[i - 1] == ref_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_len = dp[m][n]
        precision = lcs_len / m if m > 0 else 0
        recall = lcs_len / n if n > 0 else 0

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class ToxicityMetric(BaseMetric):
    """Simple toxicity check using keyword detection."""

    name = "toxicity"

    TOXIC_PATTERNS = [
        r"\b(hate|kill|violence|abuse|racist|sexist)\b",
        r"\b(stupid|idiot|moron|dumb)\b",
    ]

    async def compute(
        self,
        test_case: TestCase,
        output: str,
        prompt: str,
        duration_ms: int,
        **kwargs: Any,
    ) -> MetricResult:
        output_lower = output.lower()
        matches = []
        for pattern in self.TOXIC_PATTERNS:
            found = re.findall(pattern, output_lower, re.IGNORECASE)
            matches.extend(found)

        score = 1.0 if not matches else max(0, 1 - len(matches) * 0.2)
        return MetricResult(
            name=self.name,
            score=score,
            details={"toxic_matches": matches[:10], "is_clean": len(matches) == 0},
        )


class TokenEfficiencyMetric(BaseMetric):
    """Quality per token used."""

    name = "token_efficiency"

    async def compute(
        self,
        test_case: TestCase,
        output: str,
        prompt: str,
        duration_ms: int,
        **kwargs: Any,
    ) -> MetricResult:
        tokens_used = kwargs.get("tokens_used", len(output.split()))
        # Score based on conciseness - penalize very long outputs
        expected_len = len((test_case.expected_output or "").split()) or 100
        ratio = tokens_used / expected_len if expected_len > 0 else 1

        # Optimal is around 1.0, penalize both too short and too long
        if ratio < 0.5:
            score = ratio * 2
        elif ratio <= 1.5:
            score = 1.0
        else:
            score = max(0, 1 - (ratio - 1.5) * 0.3)

        return MetricResult(
            name=self.name,
            score=score,
            details={"tokens_used": tokens_used, "expected_tokens": expected_len, "ratio": ratio},
        )


class FactualConsistencyMetric(BaseMetric):
    """Check output against ground truth facts."""

    name = "factual_consistency"

    def __init__(self, client: AsyncOpenAI) -> None:
        self.client = client

    async def compute(
        self,
        test_case: TestCase,
        output: str,
        prompt: str,
        duration_ms: int,
        **kwargs: Any,
    ) -> MetricResult:
        ground_truth = test_case.expected_output
        if not ground_truth:
            return MetricResult(
                name=self.name, score=1.0, details={"note": "No ground truth to check"}
            )

        # Use LLM to check factual consistency
        check_prompt = f"""Compare the following output against the ground truth for factual accuracy.

GROUND TRUTH: {ground_truth}

OUTPUT TO CHECK: {output}

Are the facts in the output consistent with the ground truth? 
Respond with JSON: {{"consistent": true/false, "inconsistencies": ["list any wrong facts"]}}"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": check_prompt}],
                timeout=60,
            )
            content = response.choices[0].message.content or "{}"
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                consistent = result.get("consistent", False)
                return MetricResult(
                    name=self.name,
                    score=1.0 if consistent else 0.0,
                    details={
                        "consistent": consistent,
                        "inconsistencies": result.get("inconsistencies", []),
                    },
                )
        except Exception as e:
            return MetricResult(name=self.name, score=0.5, details={"error": str(e)})

        return MetricResult(name=self.name, score=0.5, details={"error": "Parse failed"})


class MetricsEngine:
    """Compute metrics for evaluation outputs.

    Args:
        enabled_metrics: List of metric names to compute.
        client: Optional AsyncOpenAI client for API-based metrics.

    Example:
        >>> engine = MetricsEngine(["exact_match", "latency"], client)
        >>> results = await engine.compute(test_case, output, prompt, 150)
    """

    def __init__(
        self, enabled_metrics: List[str], client: Optional[AsyncOpenAI] = None
    ) -> None:
        self.enabled_metrics = enabled_metrics
        self.client = client
        self._metrics: Dict[str, BaseMetric] = self._initialize_metrics()

    def _initialize_metrics(self) -> Dict[str, BaseMetric]:
        """Initialize metric instances."""
        metrics: Dict[str, BaseMetric] = {
            "exact_match": ExactMatchMetric(),
            "latency": LatencyMetric(),
            "format_validation": FormatValidationMetric(),
            "bleu": BLEUMetric(),
            "rouge": ROUGEMetric(),
            "toxicity": ToxicityMetric(),
            "token_efficiency": TokenEfficiencyMetric(),
        }

        # API-dependent metrics
        if self.client:
            metrics["semantic_similarity"] = SemanticSimilarityMetric(self.client)
            metrics["llm_judge"] = LLMJudgeMetric(self.client)
            metrics["factual_consistency"] = FactualConsistencyMetric(self.client)

        return metrics

    async def compute(
        self,
        test_case: TestCase,
        output: str,
        prompt: str,
        duration_ms: int,
        **kwargs: Any,
    ) -> List[MetricResult]:
        """Compute all enabled metrics for an output.

        Args:
            test_case: The test case being evaluated.
            output: The LLM output to evaluate.
            prompt: The prompt that was sent.
            duration_ms: Response latency in milliseconds.
            **kwargs: Additional parameters for specific metrics.

        Returns:
            List of MetricResult objects.
        """
        results: List[MetricResult] = []

        tasks = []
        for metric_name in self.enabled_metrics:
            if metric_name in self._metrics:
                metric = self._metrics[metric_name]
                tasks.append(metric.compute(test_case, output, prompt, duration_ms, **kwargs))
            else:
                results.append(
                    MetricResult(
                        name=metric_name,
                        score=0.0,
                        details={"error": f"Metric '{metric_name}' not available"},
                    )
                )

        if tasks:
            computed = await asyncio.gather(*tasks, return_exceptions=True)
            for r in computed:
                if isinstance(r, MetricResult):
                    results.append(r)
                elif isinstance(r, Exception):
                    results.append(
                        MetricResult(name="unknown", score=0.0, details={"error": str(r)})
                    )

        return results

    @staticmethod
    def summarize(results: List[MetricResult]) -> Dict[str, Any]:
        """Summarize metric results into a dictionary."""
        return {result.name: result.score for result in results}
