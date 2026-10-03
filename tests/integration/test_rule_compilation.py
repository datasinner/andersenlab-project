"""The rule-compilation agent on the ingested TNPA document, with scripted
model answers: real graph, tools, retrieval, grounding and database; no
network. The scripted rule is the hand-written golden light-dues rule, with
its citations pointed at the real chunk ids."""

import copy
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import func, select

from app.agent.state import ChargeSpec
from app.agent.tools import ToolExecutor
from app.config import settings
from app.db import async_session_factory
from app.domain.vessel import VesselCall, resolve_quantities
from app.llm.client import FakeLLMClient
from app.llm.embeddings import FakeEmbedder
from app.llm.resilience import LLMUnavailableError
from app.models import CompiledRule, TariffDocument
from app.retrieval.search import TariffSearch
from app.rules.engine import evaluate_rule
from app.services.documents import (
    DocumentNotReadyError,
    PortNotCoveredError,
    resolve_document_for_port,
)
from app.services.rulebook import RulebookService, UnknownChargeError
from tests.integration.golden import golden_rule

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
LIGHT_DUES = ChargeSpec(
    charge_id="light_dues",
    name="Light dues",
    description="Per 100 tons of gross tonnage.",
    section_refs=["1.1.1"],
    payer="vessel",
    trigger="per_call",
)
APPROVED = {"issues": []}


async def _golden_light_dues(rate: str = "117.08") -> dict:
    rule = await golden_rule("light_dues", "1.1.1")
    rule["components"][1]["rate"] = rate
    return rule


def _tool_call(name: str, call_id: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _submit(chunk_ids: list[int], notes: str = "Use the rate for all other vessels.") -> AIMessage:
    return _tool_call("SubmitEvidence", "submit", chunk_ids=chunk_ids, notes=notes)


async def _document() -> TariffDocument:
    async with async_session_factory() as session:
        return await session.scalar(select(TariffDocument))


def _service(llm: FakeLLMClient) -> RulebookService:
    return RulebookService(async_session_factory, llm, FakeEmbedder())


async def _cached_rows() -> int:
    async with async_session_factory() as session:
        return await session.scalar(select(func.count()).select_from(CompiledRule))


def _prompt(llm: FakeLLMClient, name: str, occurrence: int) -> str:
    calls = [call for call in llm.calls if call.name == name]
    return calls[occurrence].messages[1].content


# -- the happy path ------------------------------------------------------------------


async def test_compiles_researches_validates_reviews_and_caches(ingested_tnpa):
    rule = await _golden_light_dues()
    llm = FakeLLMClient()
    llm.script(
        "research",
        _tool_call("SearchTariff", "s1", query="definition of tonnage"),
        _submit([]),
    )
    llm.script("extract_rule", rule)
    llm.script("critique", {"issues": [{"severity": "minor", "problem": "Mention the 60 days."}]})
    service = _service(llm)
    document = await _document()

    outcome = await service.compile_charge(document, "Durban", LIGHT_DUES)

    assert outcome.status == "approved" and not outcome.from_cache
    assert outcome.revisions == 0
    assert outcome.review_notes == ["Mention the 60 days."]
    assert outcome.sections_read == ["1.1.1"]
    assert outcome.rule.port_key == "Durban" and outcome.rule.currency == "ZAR"
    tools_used = [step.tool for step in outcome.steps if step.tool]
    assert tools_used == ["ReadSection", "SearchTariff", "SubmitEvidence"]
    nodes = [step.node for step in outcome.steps if step.node != "research"]
    assert nodes == ["extract_rule", "validate_rule", "critique"]
    # The extraction prompt carried the research notes and the opened section.
    extract_prompt = _prompt(llm, "extract_rule", 0)
    assert "Use the rate for all other vessels." in extract_prompt
    assert 'section="1.1.1"' in extract_prompt

    # Cached: the next compile makes no model calls.
    calls_before = len(llm.calls)
    cached = await service.compile_charge(document, "Durban", LIGHT_DUES)
    assert cached.from_cache and cached.status == "approved"
    assert cached.review_notes == ["Mention the 60 days."]
    assert len(llm.calls) == calls_before
    assert await _cached_rows() == 1

    # And the compiled rule prices SUDESTADA like the golden rule does.
    profile = json.loads((FIXTURES / "vessels" / "sudestada.json").read_text())
    quantities = resolve_quantities(VesselCall.from_profile(profile, port="Durban"))
    assert evaluate_rule(cached.rule, quantities).amount == Decimal("60062.04")


# -- revision loops -------------------------------------------------------------------


async def test_ungrounded_rule_goes_back_with_the_problem(ingested_tnpa):
    llm = FakeLLMClient()
    llm.script("research", _submit([]))
    llm.script("extract_rule", await _golden_light_dues(rate="117.80"), await _golden_light_dues())
    llm.script("critique", APPROVED)

    outcome = await _service(llm).compile_charge(await _document(), "Durban", LIGHT_DUES)

    assert outcome.status == "approved"
    assert outcome.revisions == 1
    retry_prompt = _prompt(llm, "extract_rule", 1)
    assert "components[1].rate: 117.8 does not appear in the cited text" in retry_prompt
    assert "Previous answer:" in retry_prompt


async def test_blocking_review_triggers_a_revision(ingested_tnpa):
    llm = FakeLLMClient()
    llm.script("research", _submit([]))
    llm.script("extract_rule", await _golden_light_dues(), await _golden_light_dues())
    llm.script(
        "critique",
        {"issues": [{"severity": "blocking", "problem": "Charge per 100 tons, not per ton."}]},
        APPROVED,
    )

    outcome = await _service(llm).compile_charge(await _document(), "Durban", LIGHT_DUES)

    assert outcome.status == "approved" and outcome.revisions == 1
    assert "Charge per 100 tons, not per ton." in _prompt(llm, "extract_rule", 1)


async def test_unresolved_review_keeps_the_grounded_rule_flagged(ingested_tnpa):
    blocking = {"issues": [{"severity": "blocking", "problem": "Still wrong."}]}
    llm = FakeLLMClient()
    llm.script("research", _submit([]))
    llm.script("extract_rule", *[await _golden_light_dues() for _ in range(3)])
    llm.script("critique", blocking, blocking, blocking)

    outcome = await _service(llm).compile_charge(await _document(), "Durban", LIGHT_DUES)

    assert settings.agent_max_revisions == 2
    assert outcome.status == "low_confidence"
    assert outcome.issues == ["Still wrong."]
    assert outcome.rule is not None
    assert await _cached_rows() == 1


async def test_when_revisions_run_out_the_best_reviewed_version_is_kept(ingested_tnpa):
    def review(*problems: str) -> dict:
        return {"issues": [{"severity": "blocking", "problem": p} for p in problems]}

    llm = FakeLLMClient()
    llm.script("research", _submit([]))
    llm.script(
        "extract_rule",
        await _golden_light_dues(),
        await _golden_light_dues(),
        await _golden_light_dues(),
    )
    llm.script("critique", review("a", "b"), review("c"), review("d", "e", "f"))

    outcome = await _service(llm).compile_charge(await _document(), "Durban", LIGHT_DUES)

    assert outcome.status == "low_confidence"
    assert outcome.issues == ["c"]  # the second version, with one blocking issue


async def test_a_rule_that_never_validates_fails_and_the_failure_is_cached(ingested_tnpa):
    llm = FakeLLMClient()
    llm.script("research", _submit([]))
    llm.script("extract_rule", *[await _golden_light_dues(rate="999.99") for _ in range(3)])
    service = _service(llm)
    document = await _document()

    outcome = await service.compile_charge(document, "Durban", LIGHT_DUES)

    assert outcome.status == "failed"
    assert outcome.rule is None
    assert outcome.issues == ["components[1].rate: 999.99 does not appear in the cited text"]
    assert await _cached_rows() == 1

    # A calculation doesn't retry it: the failure comes from the cache.
    calls_before = len(llm.calls)
    cached = await service.compile_charge(document, "Durban", LIGHT_DUES)
    assert (cached.status, cached.from_cache, cached.rule) == ("failed", True, None)
    assert cached.issues == outcome.issues
    assert len(llm.calls) == calls_before


async def test_unreadable_extraction_is_fed_back(ingested_tnpa):
    broken = copy.deepcopy(await _golden_light_dues())
    del broken["components"][0]["kind"]
    llm = FakeLLMClient()
    llm.script("research", _submit([]))
    llm.script("extract_rule", broken, await _golden_light_dues())
    llm.script("critique", APPROVED)

    outcome = await _service(llm).compile_charge(await _document(), "Durban", LIGHT_DUES)

    assert outcome.status == "approved" and outcome.revisions == 1
    assert "components.0" in _prompt(llm, "extract_rule", 1)


# -- research ---------------------------------------------------------------------------


async def test_research_without_submitting_uses_everything_seen(ingested_tnpa):
    llm = FakeLLMClient()
    llm.script(
        "research",
        _tool_call("ReadSection", "r1", ref="3.1"),
        AIMessage(content="I have what I need."),
    )
    llm.script("extract_rule", await _golden_light_dues())
    llm.script("critique", APPROVED)

    outcome = await _service(llm).compile_charge(await _document(), "Durban", LIGHT_DUES)

    assert outcome.sections_read == ["1.1.1", "3.1"]
    extract_prompt = _prompt(llm, "extract_rule", 0)
    assert 'section="3.1"' in extract_prompt
    assert "research ended without submitting evidence" in extract_prompt


async def test_research_stops_at_the_tool_budget(ingested_tnpa, monkeypatch):
    monkeypatch.setattr(settings, "agent_max_tool_calls", 1)
    llm = FakeLLMClient()
    llm.script("research", _tool_call("ListSections", "l1", prefix="3."))
    llm.script("extract_rule", await _golden_light_dues())
    llm.script("critique", APPROVED)

    outcome = await _service(llm).compile_charge(await _document(), "Durban", LIGHT_DUES)

    assert outcome.status == "approved"
    assert len([call for call in llm.calls if call.name == "research"]) == 1


async def test_tools_report_bad_requests_as_text(ingested_tnpa):
    search = TariffSearch(async_session_factory, FakeEmbedder())
    tools = ToolExecutor(async_session_factory, search, ingested_tnpa.document_id)

    assert await tools.execute("Teleport", {}) == "Unknown tool Teleport."
    assert (await tools.execute("ReadSection", {})).startswith("Invalid arguments for ReadSection")
    assert "No section with ref '99'" in await tools.execute("ReadSection", {"ref": "99"})
    assert "No definition mentioning 'zzz'" in await tools.execute(
        "LookupDefinition", {"term": "zzz"}
    )
    outline = await tools.execute("ListSections", {"prefix": "3.1"})
    assert outline.splitlines()[0].startswith("3.1 | 3.1 GENERAL TERMS AND CONDITIONS")
    definitions = await tools.execute("LookupDefinition", {"term": "tonnage"})
    assert "<tariff_excerpt chunk_id=" in definitions
    assert tools.seen  # every excerpt shown is remembered for the evidence


async def test_an_excerpt_is_shown_in_full_once_then_referenced(ingested_tnpa):
    search = TariffSearch(async_session_factory, FakeEmbedder())
    tools = ToolExecutor(async_session_factory, search, ingested_tnpa.document_id)

    first = await tools.execute("ReadSection", {"ref": "3.6"})
    again = await tools.execute("ReadSection", {"ref": "3.6"})
    assert "(shown above)" not in first
    assert "(shown above)" in again and len(again) < len(first) / 3
    assert first.count("<tariff_excerpt") == again.count("<tariff_excerpt")


# -- the rulebook ---------------------------------------------------------------------------


async def test_compile_port_picks_the_charges_a_vessel_routinely_pays(ingested_tnpa):
    llm = FakeLLMClient()
    for _ in range(2):
        llm.script("research", _submit([]))
        llm.script("extract_rule", await _golden_light_dues())
        llm.script("critique", APPROVED)
    service = _service(llm)
    document = await _document()

    report = await service.compile_port(document, "Durban")

    # light dues and towage are vessel charges; dry bulk cargo dues are paid by cargo.
    assert sorted(outcome.charge_id for outcome in report.outcomes) == ["light_dues", "towage"]
    assert report.usage.prompt_tokens > 0

    with pytest.raises(UnknownChargeError, match="not_a_charge"):
        await service.compile_port(document, "Durban", charge_ids=["not_a_charge"])


async def test_provider_failure_fails_the_charge_without_caching(ingested_tnpa):
    llm = FakeLLMClient()
    llm.script("research", LLMUnavailableError("provider down"))

    outcome = await _service(llm).compile_charge(await _document(), "Durban", LIGHT_DUES)

    assert outcome.status == "failed"
    assert outcome.error == "LLMUnavailableError: provider down"
    assert await _cached_rows() == 0


async def test_resolving_the_document_for_a_port(ingested_tnpa):
    async with async_session_factory() as session:
        document, port_key = await resolve_document_for_port(session, "port of durban")
        assert (document.id, port_key) == (ingested_tnpa.document_id, "Durban")

        with pytest.raises(PortNotCoveredError):
            await resolve_document_for_port(session, "Amsterdam")
        with pytest.raises(PortNotCoveredError):
            await resolve_document_for_port(session, "Durban", on_date=date(2026, 1, 1))
        assert await resolve_document_for_port(session, "Durban", on_date=date(2024, 11, 15))

        document.status = "indexing"
        await session.commit()
        with pytest.raises(DocumentNotReadyError):
            await resolve_document_for_port(session, "Durban", document_id=document.id)


# -- the API ----------------------------------------------------------------------------------


async def test_compile_and_read_rules_over_http(client, app_instance, ingested_tnpa):
    llm: FakeLLMClient = app_instance.state.llm_client
    llm.script("research", _submit([]))
    llm.script("extract_rule", await _golden_light_dues())
    llm.script("critique", APPROVED)

    compiled = await client.post(
        "/v1/rules/compile", json={"port": "Durban", "charge_ids": ["light_dues"]}
    )
    assert compiled.status_code == 200, compiled.text
    body = compiled.json()
    assert body["port"] == "Durban"
    (rule,) = body["rules"]
    assert (rule["charge_id"], rule["status"], rule["from_cache"]) == (
        "light_dues",
        "approved",
        False,
    )
    assert rule["rule"]["components"][1]["rate"] == "117.08"

    rulebook = (await client.get("/v1/rules", params={"port": "Durban"})).json()
    assert [r["charge_id"] for r in rulebook["rules"]] == ["light_dues"]
    assert rulebook["rules"][0]["from_cache"] is True

    unknown = await client.post("/v1/rules/compile", json={"port": "Amsterdam"})
    assert unknown.status_code == 422
    assert unknown.json()["error"]["code"] == "PORT_NOT_COVERED"
