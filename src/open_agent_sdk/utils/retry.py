"""Exponential backoff retry logic."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay_ms: int = 2000
    max_delay_ms: int = 30000
    retryable_status_codes: list[int] = field(default_factory=lambda: [429, 500, 502, 503, 529])


DEFAULT_RETRY_CONFIG = RetryConfig()


def is_retryable_error(error: Exception, config: RetryConfig | None = None) -> bool:
    """Check if an error is retryable based on status code."""
    cfg = config or DEFAULT_RETRY_CONFIG
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    if status and status in cfg.retryable_status_codes:
        return True
    # Check for connection errors
    error_name = type(error).__name__
    if error_name in ("ConnectionError", "TimeoutError", "ConnectTimeout"):
        return True
    return False


def is_auth_error(error: Exception) -> bool:
    """Check if error is an authentication error."""
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    return status in (401, 403)


def is_rate_limit_error(error: Exception) -> bool:
    """Check if error is a rate limit error."""
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    return status == 429


def is_prompt_too_long_error(error: Exception) -> bool:
    """Check if error indicates the prompt is too long."""
    msg = str(error).lower()
    return "prompt is too long" in msg or "context_length_exceeded" in msg


def get_retry_delay(attempt: int, config: RetryConfig | None = None) -> float:
    """Calculate retry delay with exponential backoff and jitter.

    Returns delay in seconds.
    """
    cfg = config or DEFAULT_RETRY_CONFIG
    delay_ms = cfg.base_delay_ms * (2 ** attempt)
    delay_ms = min(delay_ms, cfg.max_delay_ms)
    # Add jitter (0-25%)
    jitter = delay_ms * random.random() * 0.25
    return (delay_ms + jitter) / 1000.0


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    config: RetryConfig | None = None,
    abort_signal: Any = None,
) -> T:
    """Execute an async function with retry logic."""
    cfg = config or DEFAULT_RETRY_CONFIG
    last_error: Exception | None = None

    for attempt in range(cfg.max_retries + 1):
        try:
            return await fn()
        except Exception as e:
            last_error = e
            if not is_retryable_error(e, cfg):
                raise
            if attempt >= cfg.max_retries:
                raise
            delay = get_retry_delay(attempt, cfg)
            await asyncio.sleep(delay)

    raise last_error  # type: ignore


def format_api_error(error: Exception) -> str:
    """Format an API error for display."""
    status = getattr(error, "status_code", None) or getattr(error, "status", None)
    msg = str(error)
    if status:
        return f"API Error ({status}): {msg}"
    return f"API Error: {msg}"
