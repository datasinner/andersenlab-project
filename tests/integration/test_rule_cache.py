"""Which cached rule a charge uses after a prompt change."""

from sqlalchemy import delete

from app.config import settings
from app.db import async_session_factory
from app.llm.client import FakeLLMClient
from app.llm.embeddings import FakeEmbedder
from app.models import CompiledRule
from app.rules.dsl import RULE_SCHEMA_VERSION
from app.services.rulebook import RulebookService, prompt_version
from tests.integration.golden import golden_rule


async def _cache(document_id, rule: dict | None, prompts: str) -> int:
    """Cache an outcome, replacing one under the same prompts (as compiling does)."""
    async with async_session_factory() as session:
        await session.execute(
            delete(CompiledRule).where(
                CompiledRule.document_id == document_id, CompiledRule.prompt_version == prompts
            )
        )
        row = CompiledRule(
            document_id=document_id,
            port_key="Durban",
            charge_id="light_dues",
            rule={**rule, "charge_id": "light_dues", "port_key": "Durban"} if rule else None,
            rule_schema_version=RULE_SCHEMA_VERSION,
            prompt_version=prompts,
            model="test",
            critic_verdict={"outcome": "approved" if rule else "failed"},
        )
        session.add(row)
        await session.commit()
        return row.id


async def test_rules_from_older_prompts_are_used_until_recompiled(ingested_tnpa, monkeypatch):
    document_id = ingested_tnpa.document_id
    rulebook = RulebookService(async_session_factory, FakeLLMClient(), FakeEmbedder())
    rule = await golden_rule("light_dues", "1.1.1")

    async def in_use() -> int | None:
        row = await rulebook.cached_rule(document_id, "Durban", "light_dues")
        return row.id if row else None

    older = await _cache(document_id, rule, "extract_rule@0")
    assert await in_use() == older

    # A failed recompile under the current prompts doesn't replace a working rule...
    failed = await _cache(document_id, None, prompt_version())
    assert await in_use() == older
    # ...unless rules from older prompts may not be reused.
    monkeypatch.setattr(settings, "rules_reuse_older_prompts", False)
    assert await in_use() == failed
    monkeypatch.setattr(settings, "rules_reuse_older_prompts", True)

    current = await _cache(document_id, rule, prompt_version())
    assert await in_use() == current
    assert [row.id for row in await rulebook.cached_rules(document_id, "Durban")] == [current]
