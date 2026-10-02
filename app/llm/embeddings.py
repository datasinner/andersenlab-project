"""Text embeddings: one protocol, an OpenAI implementation and a fake."""

import hashlib
import math
import re
import time
from collections.abc import Sequence
from typing import Protocol

import structlog
from langchain_openai import OpenAIEmbeddings

from app.config import settings
from app.llm.resilience import LLMConfigurationError, ResiliencePolicy

logger = structlog.get_logger("app.llm")

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    model_name: str
    dimensions: int

    async def embed(self, texts: Sequence[str], *, name: str = "embed") -> list[list[float]]: ...


class OpenAIEmbedder:
    def __init__(self, policy: ResiliencePolicy, client: OpenAIEmbeddings | None = None) -> None:
        self._policy = policy
        self.model_name = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self._client = client or self._build_client()

    @staticmethod
    def _build_client() -> OpenAIEmbeddings:
        if not settings.openai_api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is not set. Set it, or use LLM_PROVIDER=fake to run without one."
            )
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            api_key=settings.openai_api_key,
            max_retries=0,
            timeout=settings.llm_timeout_seconds + 5,
            # Token-length checking would download tiktoken data at runtime;
            # chunks are far below the model's input limit anyway.
            check_embedding_ctx_length=False,
        )

    async def embed(self, texts: Sequence[str], *, name: str = "embed") -> list[list[float]]:
        if not texts:
            return []
        started = time.monotonic()
        vectors = await self._policy.run(
            lambda: self._client.aembed_documents(list(texts)), name=name
        )
        logger.info(
            "embedding_call",
            name=name,
            model=self.model_name,
            texts=len(texts),
            latency_ms=int((time.monotonic() - started) * 1000),
        )
        return vectors


class FakeEmbedder:
    """Deterministic, offline embeddings by feature hashing: each word adds
    ±1 to a hashed dimension, then the vector is L2-normalised. Texts that
    share words end up close together, which is enough for retrieval tests
    to be meaningful without a network."""

    model_name = "fake-embedding"

    def __init__(self, dimensions: int | None = None) -> None:
        self.dimensions = dimensions or settings.embedding_dimensions

    async def embed(self, texts: Sequence[str], *, name: str = "embed") -> list[list[float]]:
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            sign = 1.0 if value & 1 else -1.0
            vector[(value >> 1) % self.dimensions] += sign
        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0:
            # Cosine distance is undefined for a zero vector.
            vector[0] = 1.0
            return vector
        return [component / norm for component in vector]


def build_embedder(policy: ResiliencePolicy) -> Embedder:
    if settings.llm_provider == "fake":
        return FakeEmbedder()
    return OpenAIEmbedder(policy)
