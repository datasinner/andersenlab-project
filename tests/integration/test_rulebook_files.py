"""Rulebook files: export from one database state, import into another where
the same PDF was ingested again (new chunk ids, a different catalogue)."""

import copy

from sqlalchemy import select

from app.db import async_session_factory
from app.models import ChargeCatalogueEntry, Chunk, CompiledRule, TariffDocument
from app.rules.dsl import ChargeRule
from app.rules.grounding import check_grounding
from app.services.rulebook_files import export_rulebook, import_rulebook
from tests.integration.conftest import TNPA_CATALOGUE, TNPA_PDF, tnpa_pipeline
from tests.integration.golden import golden_rule, seed_compiled_rule

OTHER_CATALOGUE = {
    "charges": [
        {
            "charge_id": "lights",
            "name": "Lights",
            "section_refs": ["1.1.1"],
            "payer": "vessel",
            "trigger": "per_call",
            "description": "Named differently by another ingestion run.",
        }
    ]
}


async def _export(document_id) -> dict:
    async with async_session_factory() as session:
        document = await session.get(TariffDocument, document_id)
        return await export_rulebook(session, document)


async def _rebuild_with_other_catalogue():
    pipeline, _ = tnpa_pipeline(catalogue=(OTHER_CATALOGUE,))
    return await pipeline.ingest(TNPA_PDF.read_bytes(), TNPA_PDF.name, force=True)


async def _exported_golden_rulebook(ingested_tnpa) -> dict:
    document_id = ingested_tnpa.document_id
    await seed_compiled_rule(document_id, "light_dues", await golden_rule("light_dues", "1.1.1"))
    await seed_compiled_rule(document_id, "towage", await golden_rule("towage_dues", "3.6"))
    return await _export(document_id)


async def test_a_rulebook_moves_to_a_database_with_new_chunk_ids(ingested_tnpa):
    data = await _exported_golden_rulebook(ingested_tnpa)
    assert sorted(entry["charge_id"] for entry in data["ports"]["Durban"]) == [
        "light_dues",
        "towage",
    ]
    old_chunk_ids = {int(chunk_id) for chunk_id in data["chunk_ordinals"]}

    rebuilt = await _rebuild_with_other_catalogue()
    result = await import_rulebook(async_session_factory, data)
    assert result.imported and result.rules == 2 and result.left_out == []

    async with async_session_factory() as session:
        catalogue = await session.scalars(
            select(ChargeCatalogueEntry.charge_id).where(
                ChargeCatalogueEntry.document_id == rebuilt.document_id
            )
        )
        assert sorted(catalogue) == sorted(c["charge_id"] for c in TNPA_CATALOGUE["charges"])
        rows = list(await session.scalars(select(CompiledRule)))
        chunks = dict((await session.execute(select(Chunk.id, Chunk.content))).all())
    assert {row.port_key for row in rows} == {"Durban"}
    for row in rows:
        rule = ChargeRule.model_validate(row.rule)
        assert check_grounding(rule, chunks) == []
        assert not {citation.chunk_id for citation in rule.citations} & old_chunk_ids


async def test_import_is_skipped_when_the_rules_could_not_be_used(ingested_tnpa):
    data = await _exported_golden_rulebook(ingested_tnpa)

    already = await import_rulebook(async_session_factory, data)
    assert not already.imported and "already has compiled rules" in already.message

    stale = {**data, "prompt_version": "extract_rule@0"}
    result = await import_rulebook(async_session_factory, stale)
    assert not result.imported and "recompile" in result.message

    other_pdf = copy.deepcopy(data)
    other_pdf["document"]["checksum"] = "0" * 64
    result = await import_rulebook(async_session_factory, other_pdf)
    assert not result.imported and "not ingested" in result.message


async def test_a_rule_whose_quote_is_gone_is_left_out(ingested_tnpa):
    data = await _exported_golden_rulebook(ingested_tnpa)
    towage = next(e for e in data["ports"]["Durban"] if e["charge_id"] == "towage")
    towage["rule"]["citations"][0]["quote"] = "words that appear nowhere in the tariff"

    await _rebuild_with_other_catalogue()
    result = await import_rulebook(async_session_factory, data)
    assert result.imported and result.rules == 1 and result.left_out == ["towage"]
