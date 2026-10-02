import asyncio

import httpx
import openai
import pytest

from app.llm.resilience import (
    LLMRateLimitedError,
    LLMTimeoutError,
    LLMUnavailableError,
    ResiliencePolicy,
)

REQUEST = httpx.Request("POST", "https://api.example.invalid/v1/chat/completions")


def _status_error(code: int) -> openai.APIStatusError:
    return openai.APIStatusError(
        f"status {code}", response=httpx.Response(code, request=REQUEST), body=None
    )


def _policy(**overrides) -> ResiliencePolicy:
    options = {
        "max_concurrency": 4,
        "timeout_seconds": 5,
        "max_retries": 2,
        "backoff_base_seconds": 0,
    }
    options.update(overrides)
    return ResiliencePolicy(**options)


class _Flaky:
    """Raises the given errors in order, then returns "ok"."""

    def __init__(self, *errors: Exception) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


@pytest.mark.parametrize(
    "error",
    [_status_error(429), _status_error(503), openai.APIConnectionError(request=REQUEST)],
    ids=["rate-limit", "server-error", "connection"],
)
async def test_transient_errors_are_retried(error):
    operation = _Flaky(error, error)
    assert await _policy().run(operation, name="test") == "ok"
    assert operation.calls == 3


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_status_error(429), LLMRateLimitedError),
        (_status_error(500), LLMUnavailableError),
        (openai.APIConnectionError(request=REQUEST), LLMUnavailableError),
    ],
)
async def test_retries_stop_after_the_limit(error, expected):
    operation = _Flaky(error, error, error)
    with pytest.raises(expected):
        await _policy(max_retries=2).run(operation, name="test")
    assert operation.calls == 3


@pytest.mark.parametrize("code", [400, 401, 404])
async def test_client_errors_are_not_retried(code):
    operation = _Flaky(_status_error(code))
    with pytest.raises(LLMUnavailableError, match="rejected the request"):
        await _policy().run(operation, name="test")
    assert operation.calls == 1


async def test_sdk_timeout_and_other_sdk_errors_are_not_retried():
    timeout = _Flaky(openai.APITimeoutError(request=REQUEST))
    with pytest.raises(LLMTimeoutError):
        await _policy().run(timeout, name="test")
    assert timeout.calls == 1

    other = _Flaky(openai.OpenAIError("misconfigured"))
    with pytest.raises(LLMUnavailableError, match="misconfigured"):
        await _policy().run(other, name="test")


async def test_slow_calls_time_out():
    async def slow() -> str:
        await asyncio.sleep(1)
        return "late"

    with pytest.raises(LLMTimeoutError, match="timed out"):
        await _policy(timeout_seconds=0.01).run(slow, name="test")


async def test_concurrency_is_bounded_by_the_semaphore():
    policy = _policy(max_concurrency=2)
    active = 0
    peak = 0

    async def operation() -> None:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1

    await asyncio.gather(*(policy.run(operation, name="test") for _ in range(6)))
    assert peak == 2


def test_backoff_grows_exponentially_with_jitter():
    policy = _policy(backoff_base_seconds=0.5)
    assert 0.5 <= policy._backoff_seconds(1) <= 0.625
    assert 2.0 <= policy._backoff_seconds(3) <= 2.5
