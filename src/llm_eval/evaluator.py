"""Core evaluation logic."""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime
from typing import Iterable, List

from openai import AsyncOpenAI
from rich.progress import Progress

from llm_eval.config import settings
from llm_eval.executor import AsyncExecutor
from llm_eval.logging import configure_logging
from llm_eval.metrics import MetricsEngine
from llm_eval.schemas import EvaluationRunConfig, EvaluationRunResult, TestCase, TestResult

import structlog


configure_logging(settings.log_level)
logger = structlog.get_logger(__name__)


class Evaluator:
    """Evaluation runner for batches of test cases."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.default_model
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.executor = AsyncExecutor()

    async def evaluate(self, suite: str, test_cases: Iterable[TestCase]) -> EvaluationRunResult:
        run_id = str(uuid.uuid4())
        started_at = datetime.utcnow()
        metrics_engine = MetricsEngine([])
        config = EvaluationRunConfig(
            suite=suite,
            baseline_version=None,
            model=self.model,
            max_concurrency=settings.max_concurrent_requests,
            timeout_seconds=settings.evaluation_timeout_seconds,
            metrics=metrics_engine.enabled_metrics,
        )

        results: List[TestResult] = []

        async def _run_case(test_case: TestCase) -> TestResult:
            prompt = self._build_prompt(test_case)
            start = time.monotonic()
            response = await self._call_model(prompt)
            duration_ms = int((time.monotonic() - start) * 1000)
            metrics = await metrics_engine.compute(test_case, response, prompt)
            tokens_used = 0
            return TestResult(
                test_case_id=test_case.test_id,
                prompt=prompt,
                output=response,
                metrics=metrics,
                duration_ms=duration_ms,
                tokens_used=tokens_used,
            )

        with Progress() as progress:
            tasks = [lambda tc=tc: _run_case(tc) for tc in test_cases]
            task_id = progress.add_task("Evaluating", total=len(tasks))

            try:
                results = await self.executor.run_tasks(
                    tasks,
                    on_task_complete=lambda _: progress.advance(task_id),
                )
            except asyncio.CancelledError:
                logger.warning("evaluation_cancelled", run_id=run_id, completed=len(results))
                raise

        completed_at = datetime.utcnow()
        return EvaluationRunResult(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            config=config,
            results=results,
        )

    def _build_prompt(self, test_case: TestCase) -> str:
        try:
            return test_case.prompt_template.format(**test_case.input_variables)
        except KeyError as exc:
            raise ValueError(f"Missing input variable: {exc}") from exc

    async def _call_model(self, prompt: str) -> str:
        response = await self.client.responses.create(
            model=self.model,
            input=prompt,
            timeout=settings.request_timeout_seconds,
        )
        return response.output_text
