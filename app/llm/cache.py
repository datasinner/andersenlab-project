"""Answer repeated structured model calls from the database.

The runtime calls (reading a plain-language query, deciding a call's facts)
and the ingestion calls (document profile, charge catalogue) are pure
functions of their input as far as the system is concerned. CachingLLMClient
keys each such call by everything that determines it (model, call name,
output schema, effort, messages) and stores the answer, so pricing the same
call again, retrying, re-running an eval or rebuilding a document costs no
tokens, and the same request always gets the same answer. Any change to a
prompt, a rule (its facts appear in the messages) or the model changes the
key. Compile calls are not cached here: the rule cache covers them, and
"refresh" must really recompile.
"""

import hashlib
import json
from collections.abc import Sequence

import structlog
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import settings
from app.llm.client import Effort, LLMClient, StructuredResult, ToolTurn, Usage
from app.models import LLMResponse

logger = structlog.get_logger("app.llm")

CACHED_CALLS = frozenset({"parse_query", "resolve_facts", "document_profile", "charge_catalogue"})


def with_response_cache(
    client: LLMClient, session_factory: async_sessionmaker[AsyncSession]
) -> LLMClient:
    """`client`, answering CACHED_CALLS from the database when LLM_RESPONSE_CACHE
    is on. Never for the fake provider, whose answers tests script per call."""
    if not settings.llm_response_cache or settings.llm_provider == "fake":
        return client
    return CachingLLMClient(client, session_factory)


class CachingLLMClient:
    def __init__(
        self,
        inner: LLMClient,
        session_factory: async_sessionmaker[AsyncSession],
        names: frozenset[str] = CACHED_CALLS,
    ) -> None:
        self._inner = inner
        self._sessions = session_factory
        self._names = names
        self.model_name = inner.model_name

    async def generate_structured[ModelT: BaseModel](
        self,
        schema: type[ModelT],
        messages: Sequence[BaseMessage],
        *,
        name: str,
        effort: Effort | None = None,
    ) -> StructuredResult[ModelT]:
        if name not in self._names:
            return await self._inner.generate_structured(schema, messages, name=name, effort=effort)
        key = _key(self.model_name, name, schema, effort, messages)
        async with self._sessions() as session:
            stored = await session.get(LLMResponse, key)
        if stored is not None:
            try:
                value = schema.model_validate(stored.response)
            except ValidationError:
                pass  # the schema changed shape without changing its JSON schema; ask again
            else:
                logger.info("llm_call_cached", name=name, model=self.model_name)
                return StructuredResult(
                    value=value, usage=Usage(), model=self.model_name, latency_ms=0
                )

        result = await self._inner.generate_structured(schema, messages, name=name, effort=effort)
        async with self._sessions() as session:
            await session.execute(
                insert(LLMResponse)
                .values(
                    key=key,
                    name=name,
                    model=self.model_name,
                    response=result.value.model_dump(mode="json"),
                )
                .on_conflict_do_nothing(index_elements=["key"])
            )
            await session.commit()
        return result

    async def generate_with_tools(
        self, messages: Sequence[BaseMessage], tools: Sequence[type[BaseModel]], *, name: str
    ) -> ToolTurn:
        return await self._inner.generate_with_tools(messages, tools, name=name)

    def shutdown(self) -> None:
        self._inner.shutdown()


def _key(
    model: str,
    name: str,
    schema: type[BaseModel],
    effort: Effort | None,
    messages: Sequence[BaseMessage],
) -> str:
    payload = {
        "model": model,
        "name": name,
        "schema": schema.model_json_schema(),
        "effort": effort,
        "messages": [[message.type, message.content] for message in messages],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()
