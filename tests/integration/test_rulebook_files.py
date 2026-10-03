"""Rulebook files: export from one database state, import into another where
the same PDF was ingested again (new chunk ids, a different catalogue)."""

import copy
import json

from sqlalchemy import select

from app.config import settings
from app.db import async_session_factory
from app.ingestion.pipeline import IngestionPipeline
from app.llm.client import FakeLLMClient
from app.llm.embeddings import FakeEmbedder
from app.models import ChargeCatalogueEntry, Chunk, CompiledRule, TariffDocument
from app.rules.dsl import ChargeRule
from app.rules.grounding import check_grounding
from app.services.rulebook import RulebookService
from app.services.rulebook_files import export_rulebook, import_rulebook
from tests.integration.conftest import TNPA_CATALOGUE, TNPA_PDF, _MemoisedParser, tnpa_pipeline
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


async def test_import_is_skipped_when_the_rules_could_not_be_used(ingested_tnpa, monkeypatch):
    data = await _exported_golden_rulebook(ingested_tnpa)

    already = await import_rulebook(async_session_factory, data)
    assert not already.imported and "already has compiled rules" in already.message

    other_schema = {**data, "rule_schema_version": 1}
    result = await import_rulebook(async_session_factory, other_schema)
    assert not result.imported and "rule schema 1" in result.message

    older_prompts = _with_prompts(data, "extract_rule@0")
    monkeypatch.setattr(settings, "rules_reuse_older_prompts", False)
    result = await import_rulebook(async_session_factory, older_prompts)
    assert not result.imported and "RULES_REUSE_OLDER_PROMPTS is off" in result.message

    other_pdf = copy.deepcopy(data)
    other_pdf["document"]["checksum"] = "0" * 64
    result = await import_rulebook(async_session_factory, other_pdf)
    assert not result.imported and "not ingested" in result.message


async def test_rules_from_older_prompts_are_imported_and_used(ingested_tnpa):
    data = _with_prompts(await _exported_golden_rulebook(ingested_tnpa), "extract_rule@0")
    rebuilt = await _rebuild_with_other_catalogue()

    result = await import_rulebook(async_session_factory, data)
    assert result.imported and result.rules == 2

    rulebook = RulebookService(async_session_factory, FakeLLMClient(), FakeEmbedder())
    rows = await rulebook.cached_rules(rebuilt.document_id, "Durban")
    assert {row.prompt_version for row in rows} == {"extract_rule@0"}


async def test_ingestion_takes_profile_and_catalogue_from_a_rulebook_file(ingested_tnpa, tmp_path):
    data = await _exported_golden_rulebook(ingested_tnpa)
    (tmp_path / "tnpa.json").write_text(json.dumps(data))
    pipeline = IngestionPipeline(
        async_session_factory,
        FakeLLMClient(),  # nothing scripted: any model call would fail the ingestion
        FakeEmbedder(),
        parser=_MemoisedParser(),
        rulebooks_dir=str(tmp_path),
    )

    result = await pipeline.ingest(TNPA_PDF.read_bytes(), TNPA_PDF.name, force=True)
    assert result.status == "ready" and result.charges == len(TNPA_CATALOGUE["charges"])
    async with async_session_factory() as session:
        document = await session.get(TariffDocument, result.document_id)
        assert document.ports == data["document"]["ports"]
        assert document.currency == data["document"]["currency"]


def _with_prompts(data: dict, version: str) -> dict:
    data = copy.deepcopy(data)
    for entries in data["ports"].values():
        for entry in entries:
            entry["prompt_version"] = version
    return data


async def test_a_rule_whose_quote_is_gone_is_left_out(ingested_tnpa):
    data = await _exported_golden_rulebook(ingested_tnpa)
    towage = next(e for e in data["ports"]["Durban"] if e["charge_id"] == "towage")
    towage["rule"]["citations"][0]["quote"] = "words that appear nowhere in the tariff"

    await _rebuild_with_other_catalogue()
    result = await import_rulebook(async_session_factory, data)
    assert result.imported and result.rules == 1 and result.left_out == ["towage"]
