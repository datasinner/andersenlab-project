import math

import pytest

from app.config import settings
from app.llm.embeddings import FakeEmbedder, OpenAIEmbedder, build_embedder
from app.llm.resilience import LLMConfigurationError, ResiliencePolicy


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


async def test_fake_embeddings_are_deterministic_unit_vectors():
    embedder = FakeEmbedder()
    first, again = await embedder.embed(["Pilotage fee per service", "Pilotage fee per service"])
    assert first == again
    assert len(first) == settings.embedding_dimensions
    assert math.isclose(math.sqrt(sum(x * x for x in first)), 1.0)


async def test_fake_embeddings_place_texts_sharing_words_closer():
    embedder = FakeEmbedder(dimensions=256)
    query, related, unrelated = await embedder.embed(
        ["towage fee per tug service", "tug service fee by tonnage", "fresh water supply"]
    )
    assert _cosine(query, related) > _cosine(query, unrelated)


async def test_fake_embedding_of_text_without_words_is_still_a_unit_vector():
    (vector,) = await FakeEmbedder(dimensions=8).embed(["…  ---"])
    assert vector == [1.0, 0, 0, 0, 0, 0, 0, 0]


class _StubOpenAIEmbeddings:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text))] for text in texts]


async def test_openai_embedder_batches_through_the_policy():
    stub = _StubOpenAIEmbeddings()
    policy = ResiliencePolicy(max_concurrency=1, timeout_seconds=5, max_retries=0)
    embedder = OpenAIEmbedder(policy, client=stub)

    assert await embedder.embed(["ab", "abcd"]) == [[2.0], [4.0]]
    assert await embedder.embed([]) == []
    assert stub.calls == [["ab", "abcd"]]


def test_openai_embedder_requires_an_api_key(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None)
    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY is not set"):
        OpenAIEmbedder(ResiliencePolicy.from_settings())


def test_build_embedder_selects_the_provider(monkeypatch):
    policy = ResiliencePolicy.from_settings()
    assert isinstance(build_embedder(policy), FakeEmbedder)
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    embedder = build_embedder(policy)
    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.dimensions == settings.embedding_dimensions
