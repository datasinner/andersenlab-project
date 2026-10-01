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
