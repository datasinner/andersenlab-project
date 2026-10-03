"""The hand-written golden Durban rules, made usable against an ingested
TNPA document: each citation is pointed at the real chunk that holds its
quote (fixture chunk ids are invented)."""

import json
from pathlib import Path

from sqlalchemy import select

from app.db import async_session_factory
from app.llm.prompts import versions
from app.models import Chunk, CompiledRule, DocumentSection
from app.rules.dsl import RULE_SCHEMA_VERSION
from app.services.rulebook import COMPILE_PROMPTS

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "rules" / "durban"


async def section_chunks(*refs: str) -> dict[int, str]:
    async with async_session_factory() as session:
        rows = await session.execute(
            select(Chunk.id, Chunk.content)
            .join(DocumentSection, Chunk.section_id == DocumentSection.id)
            .where(DocumentSection.ref.in_(refs))
        )
        return dict(rows.all())


async def golden_rule(name: str, *section_refs: str) -> dict:
    """The golden rule `name`, citations retargeted to chunks of `section_refs`."""
    rule = json.loads((GOLDEN / f"{name}.json").read_text())
    chunks = await section_chunks(*section_refs)

    def retarget(citation: dict | None) -> None:
        if citation is None:
            return
        quote = " ".join(citation["quote"].replace("<br>", " ").split()).casefold()
        for chunk_id, text in chunks.items():
            if quote in " ".join(text.replace("<br>", " ").split()).casefold():
                citation["chunk_id"] = chunk_id
                return
        raise AssertionError(f"quote not found in {section_refs}: {citation['quote']}")

    for citation in rule["citations"]:
        retarget(citation)
    for exemption in rule.get("exemptions", []):
        retarget(exemption.get("citation"))
    for adjustment in rule.get("adjustments", []):
        retarget(adjustment.get("citation"))
    return rule


async def seed_compiled_rule(document_id, charge_id: str, rule: dict, port_key: str = "Durban"):
    """Put a rule in the compiled-rule cache, as an approved compilation would."""
    rule = {**rule, "charge_id": charge_id, "port_key": port_key}
    async with async_session_factory() as session:
        session.add(
            CompiledRule(
                document_id=document_id,
                port_key=port_key,
                charge_id=charge_id,
                rule=rule,
                rule_schema_version=RULE_SCHEMA_VERSION,
                prompt_version=versions(*COMPILE_PROMPTS),
                model="golden",
                critic_verdict={"outcome": "approved", "issues": [], "review_notes": []},
            )
        )
        await session.commit()
