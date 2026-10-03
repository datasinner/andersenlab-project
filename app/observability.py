"""Optional Langfuse tracing (LANGFUSE_TRACING_ENABLED).

One trace per calculation, and one per rule compilation run on its own (CLI
or /v1/rules/compile). Every model call made inside a trace becomes a nested
generation with model, latency, tokens and cost: the LangChain callback is
carried in a context variable, so LangGraph nodes and parallel branches pick
it up without being passed anything.

LANGFUSE_CAPTURE_CONTENT=false (the default) strips prompts and responses
from exported spans and keeps the operational metadata. Adapted from the
support-assistant project.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langfuse import propagate_attributes

from app.config import settings

_callbacks: ContextVar[tuple[BaseCallbackHandler, ...]] = ContextVar("llm_callbacks", default=())

_CONTENT_ATTRIBUTE_KEYS = frozenset(
    {
        "langfuse.observation.input",
        "langfuse.observation.output",
        "langfuse.trace.input",
        "langfuse.trace.output",
    }
)
_CONTENT_ATTRIBUTE_PREFIXES = (
    "gen_ai.prompt.",
    "gen_ai.completion.",
    "gen_ai.input.messages",
    "gen_ai.output.messages",
)


def current_callbacks() -> tuple[BaseCallbackHandler, ...]:
    """Callbacks for model calls made in the current trace (empty outside one)."""
    return _callbacks.get()


def _content_attributes_to_delete(attributes: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(
        key
        for key in attributes
        if key in _CONTENT_ATTRIBUTE_KEYS
        or any(key.startswith(prefix) for prefix in _CONTENT_ATTRIBUTE_PREFIXES)
    )


def _mask_otel_content(*, params: Any) -> Any:
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}
    for identifier, span in params.spans.items():
        attributes_to_delete = _content_attributes_to_delete(span.attributes)
        if attributes_to_delete:
            patches[identifier] = OtelSpanPatch(delete_attributes=attributes_to_delete)
    return MaskOtelSpansResult(span_patches=patches) if patches else None


class Observability:
    def __init__(
        self,
        *,
        enabled: bool,
        public_key: str | None,
        secret_key: str | None,
        base_url: str,
        environment: str,
        sample_rate: float,
        capture_content: bool,
    ) -> None:
        self._capture_content = capture_content
        self._client: Any = None
        self._handler: BaseCallbackHandler | None = None
        if not enabled:
            return
        missing = [
            name
            for name, value in (
                ("LANGFUSE_PUBLIC_KEY", public_key),
                ("LANGFUSE_SECRET_KEY", secret_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Langfuse tracing is enabled but these settings are missing: " + ", ".join(missing)
            )
        # Imported here: the integration is only loaded when tracing is on.
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=base_url,
            tracing_enabled=True,
            environment=environment,
            sample_rate=sample_rate,
            mask_otel_spans=None if capture_content else _mask_otel_content,
        )
        self._handler = CallbackHandler(public_key=public_key)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        seed: str,
        metadata: dict[str, Any],
        tags: list[str],
        input_text: str | None = None,
    ) -> Iterator[None]:
        """A trace around the work in the block. Inside an existing trace (a
        compilation during a calculation), the block joins that trace."""
        if self._client is None or self._handler is None or current_callbacks():
            yield
            return
        trace_id = self._client.create_trace_id(seed=seed)
        with self._client.start_as_current_observation(
            trace_context={"trace_id": trace_id},
            name=name,
            as_type="chain",
            input=input_text if self._capture_content else None,
        ):
            with propagate_attributes(trace_name=name, tags=tags, metadata=metadata):
                token = _callbacks.set((self._handler,))
                try:
                    yield
                finally:
                    _callbacks.reset(token)

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.shutdown()


DISABLED = Observability(
    enabled=False,
    public_key=None,
    secret_key=None,
    base_url="",
    environment="",
    sample_rate=1.0,
    capture_content=False,
)


def build_observability() -> Observability:
    return Observability(
        enabled=settings.langfuse_tracing_enabled,
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
        environment=settings.app_env,
        sample_rate=settings.langfuse_sample_rate,
        capture_content=settings.langfuse_capture_content,
    )
