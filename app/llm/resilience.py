"""Concurrency, timeout and retry policy for every model call.

One ResiliencePolicy (and so one semaphore) is created per process and
shared by the chat client and the embedder, so the process never has more
than LLM_MAX_CONCURRENCY requests in flight to OpenAI.
"""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

import openai
import structlog

from app.config import settings

logger = structlog.get_logger("app.llm")

T = TypeVar("T")


class LLMError(Exception):
    """Base class for model-call failures the rest of the app handles."""


class LLMConfigurationError(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class LLMRateLimitedError(LLMError):
    pass


class LLMUnavailableError(LLMError):
    pass


class ResiliencePolicy:
    def __init__(
        self,
        *,
        max_concurrency: int,
        timeout_seconds: float,
        max_retries: int,
        backoff_base_seconds: float = 0.5,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds

    @classmethod
    def from_settings(cls) -> "ResiliencePolicy":
        return cls(
            max_concurrency=settings.llm_max_concurrency,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    async def run(self, operation: Callable[[], Awaitable[T]], *, name: str) -> T:
        """Run one model call under the semaphore and timeout. Retries only
        rate limits, 5xx responses and dropped connections; a 400 (bad
        request, invalid schema) or 401 would fail the same way again."""
        attempt = 0
        while True:
            attempt += 1
            started = time.monotonic()
            try:
                async with self._semaphore:
                    async with asyncio.timeout(self.timeout_seconds):
                        return await operation()
            except TimeoutError as exc:
                raise LLMTimeoutError(f"{name}: model call timed out") from exc
            except openai.APITimeoutError as exc:
                # The SDK's own timeout, in case it fires before ours.
                raise LLMTimeoutError(f"{name}: model call timed out") from exc
            except openai.APIConnectionError as exc:
                error: LLMError = LLMUnavailableError(f"{name}: connection to the provider failed")
                retryable = True
                cause: Exception = exc
            except openai.APIStatusError as exc:
                cause = exc
                if exc.status_code == 429 and _out_of_credit(exc):
                    # Exhausted credit, not a rate limit: retrying can't help.
                    raise LLMUnavailableError(
                        f"{name}: the provider account has no credits left"
                    ) from exc
                if exc.status_code == 429:
                    error = LLMRateLimitedError(f"{name}: provider rate limit exceeded")
                    retryable = True
                elif exc.status_code >= 500:
                    error = LLMUnavailableError(f"{name}: provider error {exc.status_code}")
                    retryable = True
                else:
                    raise LLMUnavailableError(
                        f"{name}: provider rejected the request: {exc}"
                    ) from exc
            except openai.OpenAIError as exc:
                raise LLMUnavailableError(f"{name}: provider error: {exc}") from exc

            if not retryable or attempt > self.max_retries:
                raise error from cause
            delay = self._backoff_seconds(attempt)
            logger.warning(
                "llm_call_retry",
                name=name,
                attempt=attempt,
                error=type(cause).__name__,
                elapsed_ms=int((time.monotonic() - started) * 1000),
                retry_in_ms=int(delay * 1000),
            )
            await asyncio.sleep(delay)

    def _backoff_seconds(self, attempt: int) -> float:
        base = self.backoff_base_seconds * (2 ** (attempt - 1))
        return base + random.uniform(0, base * 0.25)


# How OpenAI marks a 429 that is exhausted credit rather than a rate limit:
# the error type, or one of the codes it has used for it.
_OUT_OF_CREDIT = {"insufficient_quota", "credit_balance_exhausted"}


def _out_of_credit(exc: openai.APIStatusError) -> bool:
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error", body)
    error = error if isinstance(error, dict) else {}
    markers = {
        getattr(exc, "code", None),
        getattr(exc, "type", None),
        error.get("code"),
        error.get("type"),
    }
    return bool(markers & _OUT_OF_CREDIT)
