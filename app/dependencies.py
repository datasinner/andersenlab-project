from fastapi import Request

from app.llm.client import LLMClient
from app.llm.embeddings import Embedder
from app.services.rulebook import RulebookService


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_rulebook(request: Request) -> RulebookService:
    return request.app.state.rulebook
