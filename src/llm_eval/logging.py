"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog for JSON logs with correlation IDs."""

    logging.basicConfig(stream=sys.stdout, level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        cache_logger_on_first_use=True,
    )


def bind_context(**kwargs: Any) -> None:
    """Bind context variables for request-scoped logging."""

    structlog.contextvars.bind_contextvars(**kwargs)
