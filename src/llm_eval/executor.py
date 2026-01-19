"""Async execution engine with retries, rate limits, and circuit breaker."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Deque, Iterable, TypeVar

from tenacity import AsyncRetrying, RetryError, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm_eval.config import settings
from llm_eval.logging import configure_logging

import structlog


configure_logging(settings.log_level)
logger = structlog.get_logger(__name__)

T = TypeVar("T")


@dataclass
class CircuitBreakerState:
    failures: int = 0
    opened_at: float | None = None


class CircuitBreaker:
    """Simple circuit breaker for API failures."""

    def __init__(self, failure_threshold: int = 5, reset_seconds: int = 30) -> None:
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self.state = CircuitBreakerState()

    def record_success(self) -> None:
        self.state.failures = 0
        self.state.opened_at = None

    def record_failure(self) -> None:
        self.state.failures += 1
        if self.state.failures >= self.failure_threshold:
            self.state.opened_at = time.time()

    def allow_request(self) -> bool:
        if self.state.opened_at is None:
            return True
        if time.time() - self.state.opened_at > self.reset_seconds:
            self.state.failures = 0
            self.state.opened_at = None
            return True
        return False


class RateLimiter:
    """Rolling window rate limiter for RPM and TPM."""

    def __init__(self, rpm: int, tpm: int) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self._lock = asyncio.Lock()
        self._request_times: Deque[float] = deque()
        self._token_times: Deque[tuple[float, int]] = deque()

    async def acquire(self, tokens: int) -> None:
        while True:
            async with self._lock:
                now = time.time()
                self._prune(now)
                if len(self._request_times) < self.rpm and self._token_sum() + tokens <= self.tpm:
                    self._request_times.append(now)
                    self._token_times.append((now, tokens))
                    return
            await asyncio.sleep(0.05)

    def _prune(self, now: float) -> None:
        window = 60.0
        while self._request_times and now - self._request_times[0] > window:
            self._request_times.popleft()
        while self._token_times and now - self._token_times[0][0] > window:
            self._token_times.popleft()

    def _token_sum(self) -> int:
        return sum(tokens for _, tokens in self._token_times)


class AsyncExecutor:
    """Executor for concurrent tasks with retries and rate limits."""

    def __init__(
        self,
        max_concurrency: int = settings.max_concurrent_requests,
        retry_attempts: int = settings.retry_attempts,
        rate_limit_rpm: int = settings.rate_limit_rpm,
        rate_limit_tpm: int = settings.rate_limit_tpm,
    ) -> None:
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.rate_limiter = RateLimiter(rate_limit_rpm, rate_limit_tpm)
        self.retry_attempts = retry_attempts
        self.circuit_breaker = CircuitBreaker()

    async def run_tasks(
        self,
        tasks: Iterable[Callable[[], Awaitable[T]]],
        token_estimate: int = 1,
        on_task_complete: Callable[[T], None] | None = None,
    ) -> list[T]:
        results: list[T] = []

        async def _wrapped(task: Callable[[], Awaitable[T]]) -> T:
            async with self.semaphore:
                if not self.circuit_breaker.allow_request():
                    raise RuntimeError("Circuit breaker open")
                await self.rate_limiter.acquire(token_estimate)
                try:
                    async for attempt in AsyncRetrying(
                        stop=stop_after_attempt(self.retry_attempts),
                        wait=wait_exponential(multiplier=1, min=1, max=4),
                        retry=retry_if_exception_type(Exception),
                        reraise=True,
                    ):
                        with attempt:
                            result = await task()
                            self.circuit_breaker.record_success()
                            return result
                except RetryError as exc:
                    self.circuit_breaker.record_failure()
                    logger.error("task_failed", error=str(exc))
                    raise

        pending = [asyncio.create_task(_wrapped(task)) for task in tasks]
        for completed in asyncio.as_completed(pending):
            result = await completed
            results.append(result)
            if on_task_complete is not None:
                on_task_complete(result)
        return results
