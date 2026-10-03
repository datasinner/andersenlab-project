from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.db import async_session_factory
from app.llm.cache import CachingLLMClient
from app.llm.client import FakeLLMClient


class Answer(BaseModel):
    value: str


def _messages(text: str) -> list:
    return [SystemMessage("Decide."), HumanMessage(text)]


async def test_an_identical_call_is_answered_from_the_database(db_session):
    inner = FakeLLMClient()
    inner.script("resolve_facts", {"value": "first"}, {"value": "second"})
    client = CachingLLMClient(inner, async_session_factory)

    first = await client.generate_structured(Answer, _messages("call A"), name="resolve_facts")
    again = await client.generate_structured(Answer, _messages("call A"), name="resolve_facts")
    assert first.value.value == again.value.value == "first"
    assert again.usage.prompt_tokens == again.usage.completion_tokens == 0
    assert len(inner.calls) == 1

    other = await client.generate_structured(Answer, _messages("call B"), name="resolve_facts")
    assert other.value.value == "second" and len(inner.calls) == 2


async def test_calls_outside_the_cached_set_always_reach_the_model(db_session):
    inner = FakeLLMClient()
    inner.script("extract_rule", {"value": "one"}, {"value": "two"})
    client = CachingLLMClient(inner, async_session_factory)

    answers = [
        (await client.generate_structured(Answer, _messages("same"), name="extract_rule")).value
        for _ in range(2)
    ]
    assert [answer.value for answer in answers] == ["one", "two"]
