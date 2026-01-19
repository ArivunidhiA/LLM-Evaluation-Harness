"""Core evaluation logic."""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Iterable, List

from openai import AsyncOpenAI
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from llm_eval.config import settings
from llm_eval.executor import AsyncExecutor
from llm_eval.logging import configure_logging
from llm_eval.metrics import MetricsEngine
from llm_eval.schemas import EvaluationRunConfig, EvaluationRunResult, TestCase, TestResult

import structlog


configure_logging(settings.log_level)
logger = structlog.get_logger(__name__)


class Evaluator:
    """Evaluation runner for batches of test cases.

    Args:
        model: OpenAI model name to use for evaluation.

    Raises:
        ValueError: If OPENAI_API_KEY is not configured.

    Example:
        >>> evaluator = Evaluator(model="gpt-4-turbo")
        >>> result = await evaluator.evaluate("qa", test_cases)
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.default_model
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.executor = AsyncExecutor()
        self._shutdown_event = asyncio.Event()

    def request_shutdown(self) -> None:
        """Signal graceful shutdown - saves partial results."""
        self._shutdown_event.set()

    async def evaluate(
        self,
        suite: str,
        test_cases: Iterable[TestCase],
        metrics: List[str] | None = None,
    ) -> EvaluationRunResult:
        """Run evaluation on a batch of test cases.

        Args:
            suite: Name of the test suite being evaluated.
            test_cases: Iterable of TestCase objects to evaluate.
            metrics: List of metric names to compute. Defaults to test case metrics.

        Returns:
            EvaluationRunResult containing all test results and metadata.
        """
        run_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)
        test_case_list = list(test_cases)

        # Collect metrics from test cases if not specified
        if metrics is None:
            all_metrics = set()
            for tc in test_case_list:
                all_metrics.update(tc.evaluation_metrics)
            metrics = list(all_metrics)

        metrics_engine = MetricsEngine(metrics, self.client)
        config = EvaluationRunConfig(
            suite=suite,
            baseline_version=None,
            model=self.model,
            max_concurrency=settings.max_concurrent_requests,
            timeout_seconds=settings.evaluation_timeout_seconds,
            metrics=metrics,
        )

        results: List[TestResult] = []
        failed_count = 0

        async def _run_case(test_case: TestCase) -> TestResult | None:
            """Execute a single test case with error handling."""
            if self._shutdown_event.is_set():
                return None

            try:
                prompt = self._build_prompt(test_case)
                start = time.monotonic()
                response, tokens_used = await self._call_model(prompt)
                duration_ms = int((time.monotonic() - start) * 1000)
                computed_metrics = await metrics_engine.compute(
                    test_case, response, prompt, duration_ms
                )
                return TestResult(
                    test_case_id=test_case.test_id,
                    prompt=prompt,
                    output=response,
                    metrics=computed_metrics,
                    duration_ms=duration_ms,
                    tokens_used=tokens_used,
                )
            except Exception as exc:
                logger.error(
                    "test_case_failed",
                    test_case_id=test_case.test_id,
                    error=str(exc),
                )
                return None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                f"[cyan]Evaluating {suite}", total=len(test_case_list)
            )
            tasks = [lambda tc=tc: _run_case(tc) for tc in test_case_list]

            try:
                raw_results = await self.executor.run_tasks(
                    tasks,
                    on_task_complete=lambda _: progress.advance(task_id),
                )
                for r in raw_results:
                    if r is not None:
                        results.append(r)
                    else:
                        failed_count += 1
            except asyncio.CancelledError:
                logger.warning(
                    "evaluation_cancelled",
                    run_id=run_id,
                    completed=len(results),
                    total=len(test_case_list),
                )

        completed_at = datetime.now(timezone.utc)
        logger.info(
            "evaluation_complete",
            run_id=run_id,
            suite=suite,
            total=len(test_case_list),
            passed=len(results),
            failed=failed_count,
            duration_seconds=(completed_at - started_at).total_seconds(),
        )

        return EvaluationRunResult(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            config=config,
            results=results,
        )

    def _build_prompt(self, test_case: TestCase) -> str:
        """Build prompt from template and input variables.

        Args:
            test_case: TestCase containing template and variables.

        Returns:
            Formatted prompt string.

        Raises:
            ValueError: If a required variable is missing.
        """
        try:
            return test_case.prompt_template.format(**test_case.input_variables)
        except KeyError as exc:
            raise ValueError(f"Missing input variable: {exc}") from exc

    async def _call_model(self, prompt: str) -> tuple[str, int]:
        """Call OpenAI API and return response with token count.

        Args:
            prompt: The prompt to send to the model.

        Returns:
            Tuple of (response_text, total_tokens_used).
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            timeout=settings.request_timeout_seconds,
        )
        content = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0
        return content, tokens_used
