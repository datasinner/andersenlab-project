"""Fixtures for tests that need Postgres.

A throwaway `test` database is created (and migrated with Alembic) once per
session on the Postgres server from compose.override.yaml, and every table
is truncated before each test. Unit tests live outside this directory and
never touch the database.
"""

import subprocess
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from pathlib import Path

import asyncpg
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

TEST_DB_NAME = "test"
ADMIN_DSN = "postgresql://tariff:tariff@localhost:5432/postgres"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _test_database() -> AsyncIterator[None]:
    conn = await asyncpg.connect(dsn=ADMIN_DSN)
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}" WITH (FORCE)')
        await conn.execute(f'CREATE DATABASE "{TEST_DB_NAME}"')
    finally:
        await conn.close()

    subprocess.run(
        ["uv", "run", "alembic", "upgrade", "head"],
        cwd=PROJECT_ROOT,
        check=True,
    )

    yield

    from app.db import engine as app_engine

    await app_engine.dispose()

    conn = await asyncpg.connect(dsn=ADMIN_DSN)
    try:
        await conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DB_NAME}" WITH (FORCE)')
    finally:
        await conn.close()


@pytest_asyncio.fixture(autouse=True)
async def _clean_tables(_test_database: None) -> AsyncIterator[None]:
    from app.db import async_session_factory
    from app.models import Base

    table_names = ", ".join(table.name for table in Base.metadata.sorted_tables)
    if table_names:
        async with async_session_factory() as session:
            await session.execute(text(f"TRUNCATE TABLE {table_names} RESTART IDENTITY CASCADE"))
            await session.commit()
    yield


@pytest_asyncio.fixture
async def app_instance(_clean_tables: None):
    from app.main import create_app

    app = create_app()
    async with AsyncExitStack() as stack:
        await stack.enter_async_context(app.router.lifespan_context(app))
        yield app


@pytest_asyncio.fixture
async def client(app_instance) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app_instance)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest_asyncio.fixture
async def db_session(_clean_tables: None):
    from app.db import async_session_factory

    async with async_session_factory() as session:
        yield session


TNPA_PDF = PROJECT_ROOT / "data" / "tariffs" / "tnpa_tariff_book_2024_25.pdf"

TNPA_PROFILE = {
    "title": "Tariff Book April 2024 - March 2025",
    "authority": "Transnet National Ports Authority",
    "currency": "zar",
    "vat_percent": "15",
    "effective_from": "2024-04-01",
    "effective_to": "2025-03-31",
    "ports": [{"name": "Durban", "aliases": ["Port of Durban"]}],
}

TNPA_CATALOGUE = {
    "charges": [
        {
            "charge_id": "light_dues",
            "name": "Light dues",
            "section_refs": ["1.1.1"],
            "payer": "vessel",
            "trigger": "per_call",
            "description": "Per 100 tons of gross tonnage.",
        },
        {
            "charge_id": "towage",
            "name": "Tug assistance",
            "section_refs": ["3.6"],
            "payer": "vessel",
            "trigger": "per_service",
            "description": "Per service by tonnage band.",
        },
        {
            "charge_id": "dry_bulk_cargo_dues",
            "name": "Dry bulk cargo dues",
            "section_refs": ["7.2"],
            "payer": "cargo",
            "trigger": "per_service",
            "description": "Per ton of cargo.",
        },
    ]
}


class _MemoisedParser:
    """Parses each PDF once per test session. The cleaner mutates what the
    parser returns, so every caller gets its own deep copy."""

    _cache: dict[bytes, object] = {}

    def parse(self, content: bytes):
        import copy
        import hashlib

        from app.ingestion.parser import PyMuPdfParser

        key = hashlib.sha256(content).digest()
        if key not in self._cache:
            self._cache[key] = PyMuPdfParser().parse(content)
        return copy.deepcopy(self._cache[key])


def tnpa_pipeline(*, profile=None, catalogue=None):
    """An ingestion pipeline with the fake embedder and scripted LLM answers."""
    from app.db import async_session_factory
    from app.ingestion.pipeline import IngestionPipeline
    from app.llm.client import FakeLLMClient
    from app.llm.embeddings import FakeEmbedder

    llm = FakeLLMClient()
    llm.script("document_profile", *(profile or (TNPA_PROFILE,)))
    llm.script("charge_catalogue", *(catalogue or (TNPA_CATALOGUE,)))
    pipeline = IngestionPipeline(
        async_session_factory, llm, FakeEmbedder(), parser=_MemoisedParser()
    )
    return pipeline, llm


@pytest_asyncio.fixture
async def ingested_tnpa(_clean_tables: None):
    pipeline, _ = tnpa_pipeline()
    return await pipeline.ingest(TNPA_PDF.read_bytes(), TNPA_PDF.name)
