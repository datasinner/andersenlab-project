"""Chat model access: one protocol, an OpenAI implementation and a fake.

Everything the agent asks a model goes through LLMClient:

- generate_structured: Structured Outputs (strict JSON schema) into a
  Pydantic model. The response is validated by that model; a refusal, a
  truncated answer or a validation failure raises LLMOutputError carrying
  the errors, so callers can ask again with the feedback.
- generate_with_tools: one turn of a tool-calling loop (the research agent).
"""

import json
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from app.config import settings
from app.llm.resilience import (
    LLMConfigurationError,
    LLMError,
    LLMUnavailableError,
    ResiliencePolicy,
)
from app.llm.schema import strict_json_schema

logger = structlog.get_logger("app.llm")


class LLMOutputError(LLMError):
    """The model answered, but not with something usable: a refusal, a
    truncated answer, or JSON that fails the target model's validation."""

    def __init__(self, message: str, *, raw: str | None = None, errors: list[str] | None = None):
        super().__init__(message)
        self.raw = raw
        self.errors = errors or []


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )


@dataclass(frozen=True)
class StructuredResult[ModelT: BaseModel]:
    value: ModelT
    usage: Usage
    model: str
    latency_ms: int


@dataclass(frozen=True)
class ToolTurn:
    message: AIMessage
    usage: Usage
    model: str
    latency_ms: int


class LLMClient(Protocol):
    model_name: str

    async def generate_structured[ModelT: BaseModel](
        self, schema: type[ModelT], messages: Sequence[BaseMessage], *, name: str
    ) -> StructuredResult[ModelT]: ...

    async def generate_with_tools(
        self, messages: Sequence[BaseMessage], tools: Sequence[type[BaseModel]], *, name: str
    ) -> ToolTurn: ...

    def shutdown(self) -> None: ...


@lru_cache(maxsize=64)
def _response_format(schema: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": strict_json_schema(schema),
            "strict": True,
        },
    }


def validation_messages(exc: ValidationError) -> list[str]:
    """Compact, model-readable validation errors ("components.0.rate: ...")."""
    messages = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        messages.append(f"{location}: {error['msg']}" if location else error["msg"])
    return messages


class OpenAIClient:
    """Thin LangChain wrapper. Retries, timeouts and concurrency come from
    the shared ResiliencePolicy, so the SDK's own retries are disabled."""

    def __init__(self, policy: ResiliencePolicy, chat_model: BaseChatModel | None = None) -> None:
        self._policy = policy
        self.model_name = settings.llm_model
        self._model = chat_model or self._build_chat_model()

    @staticmethod
    def _build_chat_model() -> BaseChatModel:
        if not settings.openai_api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is not set. Set it, or use LLM_PROVIDER=fake to run without one."
            )
        options: dict[str, Any] = {
            "model": settings.llm_model,
            "api_key": settings.openai_api_key,
            "max_retries": 0,
            # The SDK timeout is a backstop; ResiliencePolicy's fires first.
            "timeout": settings.llm_timeout_seconds + 5,
        }
        if settings.llm_temperature is not None:
            options["temperature"] = settings.llm_temperature
        if settings.llm_reasoning_effort is not None:
            options["reasoning_effort"] = settings.llm_reasoning_effort
        return ChatOpenAI(**options)

    async def generate_structured[ModelT: BaseModel](
        self, schema: type[ModelT], messages: Sequence[BaseMessage], *, name: str
    ) -> StructuredResult[ModelT]:
        bound = self._model.bind(response_format=_response_format(schema))
        started = time.monotonic()
        response = await self._policy.run(lambda: bound.ainvoke(list(messages)), name=name)
        latency_ms = int((time.monotonic() - started) * 1000)
        usage = _usage(response)
        self._log(name, latency_ms, usage)

        refusal = response.additional_kwargs.get("refusal")
        if refusal:
            raise LLMOutputError(f"{name}: the model refused: {refusal}", raw=str(refusal))
        raw = response.text if isinstance(response, AIMessage) else str(response.content)
        if response.response_metadata.get("finish_reason") == "length":
            raise LLMOutputError(f"{name}: the answer was cut off (token limit)", raw=raw)
        try:
            value = schema.model_validate_json(raw)
        except ValidationError as exc:
            errors = validation_messages(exc)
            raise LLMOutputError(
                f"{name}: the answer does not match {schema.__name__}", raw=raw, errors=errors
            ) from exc
        return StructuredResult(
            value=value, usage=usage, model=self.model_name, latency_ms=latency_ms
        )

    async def generate_with_tools(
        self, messages: Sequence[BaseMessage], tools: Sequence[type[BaseModel]], *, name: str
    ) -> ToolTurn:
        bound = self._model.bind_tools(list(tools))
        started = time.monotonic()
        response = await self._policy.run(lambda: bound.ainvoke(list(messages)), name=name)
        latency_ms = int((time.monotonic() - started) * 1000)
        usage = _usage(response)
        self._log(name, latency_ms, usage)
        if not isinstance(response, AIMessage):  # pragma: no cover - LangChain contract
            raise LLMOutputError(f"{name}: unexpected response type {type(response).__name__}")
        return ToolTurn(message=response, usage=usage, model=self.model_name, latency_ms=latency_ms)

    def shutdown(self) -> None:
        pass

    def _log(self, name: str, latency_ms: int, usage: Usage) -> None:
        logger.info(
            "llm_call",
            name=name,
            model=self.model_name,
            latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )


def _usage(response: BaseMessage) -> Usage:
    metadata = getattr(response, "usage_metadata", None) or {}
    return Usage(metadata.get("input_tokens", 0), metadata.get("output_tokens", 0))


# A scripted response: a model instance or dict (structured calls), an
# AIMessage (tool calls), an exception to raise, or a callable that receives
# the messages and returns one of those.
FakeResponse = BaseModel | Mapping[str, Any] | AIMessage | Exception | Callable[..., Any]


@dataclass
class RecordedCall:
    name: str
    messages: list[BaseMessage]
    schema: type[BaseModel] | None = None
    tools: list[type[BaseModel]] = field(default_factory=list)


class FakeLLMClient:
    """Deterministic stand-in used by tests and LLM_PROVIDER=fake.

    Responses are scripted per call name and consumed in order. A call with
    nothing scripted fails like an unavailable provider, so a fake-mode app
    reports a clear error instead of inventing answers.
    """

    model_name = "fake-model"
    usage = Usage(prompt_tokens=100, completion_tokens=20)

    def __init__(self) -> None:
        self._scripts: dict[str, deque[FakeResponse]] = defaultdict(deque)
        self.calls: list[RecordedCall] = []

    def script(self, name: str, *responses: FakeResponse) -> None:
        self._scripts[name].extend(responses)

    async def generate_structured[ModelT: BaseModel](
        self, schema: type[ModelT], messages: Sequence[BaseMessage], *, name: str
    ) -> StructuredResult[ModelT]:
        self.calls.append(RecordedCall(name=name, messages=list(messages), schema=schema))
        response = self._next(name, messages)
        if isinstance(response, schema):
            value = response
        else:
            # LangChain messages are pydantic models too, but never structured data.
            if isinstance(response, BaseMessage) or not isinstance(response, BaseModel | Mapping):
                raise TypeError(f"Scripted response for '{name}' is not structured data")
            data = response.model_dump(mode="json") if isinstance(response, BaseModel) else response
            try:
                value = schema.model_validate(data)
            except ValidationError as exc:
                raise LLMOutputError(
                    f"{name}: the answer does not match {schema.__name__}",
                    raw=json.dumps(data, default=str),
                    errors=validation_messages(exc),
                ) from exc
        return StructuredResult(value=value, usage=self.usage, model=self.model_name, latency_ms=0)

    async def generate_with_tools(
        self, messages: Sequence[BaseMessage], tools: Sequence[type[BaseModel]], *, name: str
    ) -> ToolTurn:
        self.calls.append(RecordedCall(name=name, messages=list(messages), tools=list(tools)))
        response = self._next(name, messages)
        if not isinstance(response, AIMessage):
            raise TypeError(f"Scripted response for tool call '{name}' must be an AIMessage")
        return ToolTurn(message=response, usage=self.usage, model=self.model_name, latency_ms=0)

    def shutdown(self) -> None:
        pass

    def _next(self, name: str, messages: Sequence[BaseMessage]) -> Any:
        queue = self._scripts.get(name)
        if not queue:
            raise LLMUnavailableError(f"FakeLLMClient has no scripted response for '{name}'")
        response = queue.popleft()
        if isinstance(response, Exception):
            raise response
        if callable(response) and not isinstance(response, BaseModel | AIMessage):
            response = response(list(messages))
        return response


def build_llm_client(policy: ResiliencePolicy) -> LLMClient:
    if settings.llm_provider == "fake":
        return FakeLLMClient()
    return OpenAIClient(policy)
