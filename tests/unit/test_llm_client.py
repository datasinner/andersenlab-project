import json
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field

from app.config import settings
from app.llm.client import (
    FakeLLMClient,
    LLMOutputError,
    OpenAIClient,
    Usage,
    build_llm_client,
)
from app.llm.resilience import LLMConfigurationError, LLMUnavailableError, ResiliencePolicy

MESSAGES = [HumanMessage(content="Extract the fee")]


class Fee(BaseModel):
    name: str
    amount: str = Field(min_length=1)


class LookUp(BaseModel):
    """Look something up."""

    query: str


class RecordingChatModel(BaseChatModel):
    """Returns preset messages and records the keyword arguments of each call
    (response_format, tools), so tests see exactly what would be sent."""

    responses: list[AIMessage]
    calls: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording"

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any
    ) -> ChatResult:
        self.calls.append(kwargs)
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[tool.__name__ for tool in tools], **kwargs)


def _client(*responses: AIMessage) -> tuple[OpenAIClient, RecordingChatModel]:
    model = RecordingChatModel(responses=list(responses))
    policy = ResiliencePolicy(max_concurrency=1, timeout_seconds=5, max_retries=0)
    return OpenAIClient(policy, chat_model=model), model


def _answer(content: str, **extra: Any) -> AIMessage:
    usage = {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}
    return AIMessage(content=content, usage_metadata=usage, **extra)


# -- OpenAIClient --------------------------------------------------------------


async def test_structured_call_sends_a_strict_schema_and_validates_the_answer():
    client, model = _client(_answer(json.dumps({"name": "Pilotage", "amount": "4 250.00"})))

    result = await client.generate_structured(Fee, MESSAGES, name="extract")

    assert result.value == Fee(name="Pilotage", amount="4 250.00")
    assert result.usage == Usage(prompt_tokens=120, completion_tokens=30)
    assert result.model == settings.llm_model
    response_format = model.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Fee"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["required"] == ["name", "amount"]


async def test_answer_failing_validation_carries_the_errors():
    client, _ = _client(_answer(json.dumps({"name": "Pilotage", "amount": ""})))
    with pytest.raises(LLMOutputError, match="does not match Fee") as caught:
        await client.generate_structured(Fee, MESSAGES, name="extract")
    assert caught.value.errors == ["amount: String should have at least 1 character"]
    assert caught.value.raw == '{"name": "Pilotage", "amount": ""}'


async def test_refusal_is_an_output_error():
    client, _ = _client(_answer("", additional_kwargs={"refusal": "I can't help with that"}))
    with pytest.raises(LLMOutputError, match="refused"):
        await client.generate_structured(Fee, MESSAGES, name="extract")


async def test_responses_api_refusal_block_is_an_output_error():
    client, _ = _client(_answer([{"type": "refusal", "refusal": "No."}]))
    with pytest.raises(LLMOutputError, match="refused: No."):
        await client.generate_structured(Fee, MESSAGES, name="extract")


@pytest.mark.parametrize(
    "metadata",
    [
        {"finish_reason": "length"},
        {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
    ],
)
async def test_truncated_answer_is_an_output_error(metadata):
    client, _ = _client(_answer('{"name": "Pil', response_metadata=metadata))
    with pytest.raises(LLMOutputError, match="cut off"):
        await client.generate_structured(Fee, MESSAGES, name="extract")


async def test_responses_api_content_blocks_are_read_as_text():
    blocks = [{"type": "text", "text": json.dumps({"name": "Pilotage", "amount": "1"})}]
    client, _ = _client(_answer(blocks))
    result = await client.generate_structured(Fee, MESSAGES, name="extract")
    assert result.value.name == "Pilotage"


async def test_tool_turn_binds_the_tools_and_returns_the_message():
    call = {"name": "LookUp", "args": {"query": "pilotage"}, "id": "call_1"}
    client, model = _client(_answer("", tool_calls=[call]))

    turn = await client.generate_with_tools(MESSAGES, [LookUp], name="research")

    assert turn.message.tool_calls[0]["args"] == {"query": "pilotage"}
    assert turn.usage.prompt_tokens == 120
    assert model.calls[0]["tools"] == ["LookUp"]


def test_openai_client_requires_an_api_key(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None)
    policy = ResiliencePolicy.from_settings()
    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY is not set"):
        OpenAIClient(policy)


def test_openai_client_passes_model_options(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "llm_temperature", None)
    monkeypatch.setattr(settings, "llm_reasoning_effort", "low")
    chat_model = OpenAIClient._build_chat_model()
    assert chat_model.model_name == settings.llm_model
    assert chat_model.temperature is None
    assert chat_model.reasoning_effort == "low"
    assert chat_model.max_retries == 0
    assert chat_model.use_responses_api is True


def test_build_llm_client_selects_the_provider(monkeypatch):
    policy = ResiliencePolicy.from_settings()
    assert isinstance(build_llm_client(policy), FakeLLMClient)
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    assert isinstance(build_llm_client(policy), OpenAIClient)


# -- FakeLLMClient ---------------------------------------------------------------


async def test_fake_replays_scripted_answers_in_order_and_records_calls():
    fake = FakeLLMClient()
    fake.script("extract", {"name": "First", "amount": "1"}, Fee(name="Second", amount="2"))

    first = await fake.generate_structured(Fee, MESSAGES, name="extract")
    second = await fake.generate_structured(Fee, MESSAGES, name="extract")

    assert [first.value.name, second.value.name] == ["First", "Second"]
    assert [call.name for call in fake.calls] == ["extract", "extract"]
    assert fake.calls[0].schema is Fee


async def test_fake_validates_like_the_real_client():
    fake = FakeLLMClient()
    fake.script("extract", {"name": "No amount"})
    with pytest.raises(LLMOutputError) as caught:
        await fake.generate_structured(Fee, MESSAGES, name="extract")
    assert caught.value.errors == ["amount: Field required"]


async def test_fake_accepts_other_models_callables_and_exceptions():
    class Other(BaseModel):
        name: str
        amount: str

    fake = FakeLLMClient()
    fake.script(
        "extract",
        Other(name="Converted", amount="3"),
        lambda messages: {"name": f"Saw {len(messages)} message(s)", "amount": "4"},
        LLMUnavailableError("down"),
    )
    assert (await fake.generate_structured(Fee, MESSAGES, name="extract")).value.name == (
        "Converted"
    )
    assert (await fake.generate_structured(Fee, MESSAGES, name="extract")).value.name == (
        "Saw 1 message(s)"
    )
    with pytest.raises(LLMUnavailableError, match="down"):
        await fake.generate_structured(Fee, MESSAGES, name="extract")


async def test_fake_without_a_script_fails_like_an_unavailable_provider():
    with pytest.raises(LLMUnavailableError, match="no scripted response for 'extract'"):
        await FakeLLMClient().generate_structured(Fee, MESSAGES, name="extract")


async def test_fake_rejects_unstructured_scripts():
    fake = FakeLLMClient()
    fake.script("extract", AIMessage(content="not structured"))
    with pytest.raises(TypeError, match="not structured data"):
        await fake.generate_structured(Fee, MESSAGES, name="extract")

    fake.script("research", {"name": "not a message"})
    with pytest.raises(TypeError, match="must be an AIMessage"):
        await fake.generate_with_tools(MESSAGES, [LookUp], name="research")


async def test_fake_tool_turn():
    fake = FakeLLMClient()
    fake.script("research", AIMessage(content="done"))
    turn = await fake.generate_with_tools(MESSAGES, [LookUp], name="research")
    assert turn.message.content == "done"
    assert fake.calls[0].tools == [LookUp]
    fake.shutdown()
