from contextlib import contextmanager, nullcontext

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from app import observability as observability_module
from app.llm.client import OpenAIClient
from app.llm.resilience import ResiliencePolicy
from app.observability import DISABLED, Observability, current_callbacks
from tests.unit.test_llm_client import RecordingChatModel


class Answer(BaseModel):
    text: str


class RecordingHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.runs: list[str | None] = []

    def on_chat_model_start(self, serialized, messages, *, run_id, name=None, **kwargs):
        self.runs.append(name)


class _StubLangfuse:
    def __init__(self) -> None:
        self.traces: list[tuple[str, str]] = []

    def create_trace_id(self, seed: str) -> str:
        return f"trace-{seed}"

    @contextmanager
    def start_as_current_observation(self, *, trace_context, name, as_type, input):
        self.traces.append((trace_context["trace_id"], name))
        yield

    def shutdown(self) -> None:
        pass


def _enabled(monkeypatch, handler: BaseCallbackHandler) -> tuple[Observability, _StubLangfuse]:
    """An enabled Observability with Langfuse replaced by a stub (no network)."""
    monkeypatch.setattr(observability_module, "propagate_attributes", lambda **_: nullcontext())
    observability = Observability(
        enabled=False,
        public_key=None,
        secret_key=None,
        base_url="",
        environment="test",
        sample_rate=1.0,
        capture_content=False,
    )
    stub = _StubLangfuse()
    observability._client = stub
    observability._handler = handler
    return observability, stub


def test_disabled_tracing_adds_no_callbacks():
    assert not DISABLED.enabled
    with DISABLED.trace("calculate-port-dues", seed="x", metadata={}, tags=[]):
        assert current_callbacks() == ()


def test_enabled_tracing_needs_keys():
    with pytest.raises(RuntimeError, match="LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY"):
        Observability(
            enabled=True,
            public_key=None,
            secret_key=None,
            base_url="https://example.invalid",
            environment="test",
            sample_rate=1.0,
            capture_content=False,
        )


async def test_model_calls_inside_a_trace_carry_its_callback(monkeypatch):
    handler = RecordingHandler()
    observability, stub = _enabled(monkeypatch, handler)
    model = RecordingChatModel(responses=[AIMessage(content='{"text": "ok"}')])
    client = OpenAIClient(
        ResiliencePolicy(max_concurrency=1, timeout_seconds=5, max_retries=0), chat_model=model
    )

    with observability.trace("calculate-port-dues", seed="calc-1", metadata={}, tags=["x"]):
        assert current_callbacks() == (handler,)
        await client.generate_structured(Answer, [HumanMessage(content="hi")], name="resolve_facts")

    assert stub.traces == [("trace-calc-1", "calculate-port-dues")]
    assert handler.runs == ["resolve_facts"]
    assert current_callbacks() == ()


def test_a_trace_inside_a_trace_joins_it(monkeypatch):
    observability, stub = _enabled(monkeypatch, RecordingHandler())
    with observability.trace("calculate-port-dues", seed="outer", metadata={}, tags=[]):
        with observability.trace("compile-rule", seed="inner", metadata={}, tags=[]):
            pass
    assert stub.traces == [("trace-outer", "calculate-port-dues")]
