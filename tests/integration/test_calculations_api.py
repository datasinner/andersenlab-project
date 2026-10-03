"""POST /v1/calculations end to end, offline: the real graph, engine and
database; rules come from the compiled-rule cache (seeded with the golden
light-dues and towage rules), model answers are scripted."""

import json
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.config import settings
from app.llm.client import FakeLLMClient
from app.llm.resilience import LLMUnavailableError
from app.models import AgentStep, Calculation
from tests.integration.golden import golden_rule, seed_compiled_rule

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SUDESTADA = json.loads((FIXTURES / "vessels" / "sudestada.json").read_text())

FACTS = {
    "facts": [
        {
            "charge_id": "light_dues",
            "fact": "is_licensed_vessel_at_registered_port",
            "value": "false",
            "source": "vessel_data",
            "reason": "A Malta-flagged bulk carrier is not at its registered port.",
        },
        {
            "charge_id": "towage",
            "fact": "vessel_without_own_power",
            "value": "false",
            "source": "vessel_data",
            "reason": "A self-propelled bulk carrier.",
        },
    ]
}
QUERY_ANSWER = {
    "port": "Durban",
    "vessel_name": "SUDESTADA",
    "vessel_type": "Bulk Carrier",
    "flag": "Malta",
    "built_year": 2010,
    "gross_tonnage": "51300",
    "net_tonnage": "31192",
    "deadweight": "93274",
    "loa_m": "229.2",
    "beam_m": "38",
    "draft_m": "14.9",
    "passengers": None,
    "arrival": "2024-11-15T10:12:00",
    "departure": "2024-11-22T13:00:00",
    "days_alongside": "3.39",
    "activity": "Loading iron ore for export",
    "cargo_tonnes": "40,000",
    "num_services": None,
    "statements": [],
}


@pytest.fixture
async def seeded(ingested_tnpa):
    await seed_compiled_rule(
        ingested_tnpa.document_id, "light_dues", await golden_rule("light_dues", "1.1.1")
    )
    await seed_compiled_rule(
        ingested_tnpa.document_id, "towage", await golden_rule("towage_dues", "3.6")
    )
    return ingested_tnpa


def _llm(app_instance) -> FakeLLMClient:
    return app_instance.state.llm_client


async def _post(client, **body):
    return await client.post("/v1/calculations", json=body)


async def test_prices_a_vessel_profile_and_records_the_calculation(client, app_instance, seeded):
    _llm(app_instance).script("resolve_facts", FACTS)

    response = await _post(client, port="Durban", vessel=SUDESTADA)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["port"] == "Durban" and body["currency"] == "ZAR"
    assert body["vat"] == {"rate": "0.1500", "included": False}
    amounts = {item["charge_id"]: item["amount"] for item in body["line_items"]}
    assert amounts == {"light_dues": "60062.04", "towage": "147074.38"}
    assert body["total"] == "207136.42"

    towage = next(item for item in body["line_items"] if item["charge_id"] == "towage")
    assert towage["section_refs"] == ["3.6"]
    assert towage["confidence"] == "high" and towage["rule_source"] == "cache"
    assert "73,118.07 + 419.12 = 73,537.19" in towage["formula"][0]
    assert {"name": "vessel_without_own_power", "value": "false"}.items() <= {
        **towage["facts"][0]
    }.items()
    assert any("entering and leaving" in text for text in towage["assumptions"])
    assert towage["citations"][0]["section_ref"] == "3.6"

    assert [c["charge_id"] for c in body["excluded"]] == ["dry_bulk_cargo_dues"]
    assert body["excluded"][0]["reason"] == "Payable by the cargo owner, not the vessel."
    assert body["vessel"]["vessel"]["name"] == "SUDESTADA"

    record = (await client.get(f"/v1/calculations/{body['calculation_id']}")).json()
    assert record["status"] == "success" and record["result"]["total"] == "207136.42"
    assert [step["node"] for step in record["steps"]] == ["screen_charges", "resolve_facts"]

    listing = (await client.get("/v1/calculations", params={"port": "Durban"})).json()
    assert listing[0]["vessel_name"] == "SUDESTADA" and listing[0]["total"] == "207136.42"


async def test_a_plain_language_query_gives_the_same_amounts(client, app_instance, seeded):
    llm = _llm(app_instance)
    llm.script("parse_query", QUERY_ANSWER)
    llm.script("resolve_facts", FACTS)

    response = await _post(client, query="SUDESTADA at Durban, 51,300 GT, 3.39 days alongside")

    body = response.json()
    assert response.status_code == 200, response.text
    amounts = {item["charge_id"]: item["amount"] for item in body["line_items"]}
    assert amounts == {"light_dues": "60062.04", "towage": "147074.38"}
    assert body["vessel"]["call"]["cargo_tonnes"] == "40000"


async def test_profile_fields_win_over_the_query(client, app_instance, seeded):
    llm = _llm(app_instance)
    llm.script("parse_query", {**QUERY_ANSWER, "gross_tonnage": "1000", "statements": ["coaster"]})
    llm.script("resolve_facts", FACTS)

    body = (await _post(client, query="...", vessel=SUDESTADA)).json()

    assert body["vessel"]["vessel"]["gross_tonnage"] == "51300"
    assert body["vessel"]["call"]["additional"]["statements"] == ["coaster"]


async def test_overrides_win_over_the_agent(client, app_instance, seeded):
    _llm(app_instance).script("resolve_facts", FACTS)

    body = (
        await _post(
            client,
            port="Durban",
            vessel=SUDESTADA,
            charge_ids=["towage"],
            overrides={
                "num_services": 4,
                "facts": {"tug_service_outside_ordinary_working_hours": True},
            },
        )
    ).json()

    (towage,) = body["line_items"]
    # 73,537.19 × 4 services = 294,148.76, plus the 25% outside-hours surcharge
    assert towage["amount"] == "367685.95"
    assert towage["adjustments"][0]["percent"] == "25"
    sources = {fact["name"]: fact["source"] for fact in towage["facts"]}
    assert sources["tug_service_outside_ordinary_working_hours"] == "override"


async def test_fact_resolution_failure_falls_back_to_defaults(client, app_instance, seeded):
    _llm(app_instance).script("resolve_facts", LLMUnavailableError("provider down"))

    body = (await _post(client, port="Durban", vessel=SUDESTADA)).json()

    assert body["status"] == "success"
    assert body["total"] == "207136.42"
    assert body["warnings"][0].startswith("Facts for light_dues, towage could not be resolved")


async def test_a_charge_that_cannot_compile_makes_the_calculation_partial(
    client, app_instance, ingested_tnpa
):
    await seed_compiled_rule(
        ingested_tnpa.document_id, "light_dues", await golden_rule("light_dues", "1.1.1")
    )
    llm = _llm(app_instance)
    llm.script("research", LLMUnavailableError("provider down"))  # towage is not cached
    llm.script("resolve_facts", FACTS)

    body = (await _post(client, port="Durban", vessel=SUDESTADA)).json()

    assert body["status"] == "partial"
    assert [item["charge_id"] for item in body["line_items"]] == ["light_dues"]
    assert body["failed"][0]["charge_id"] == "towage"
    assert "provider down" in body["failed"][0]["error"]


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({"vessel": SUDESTADA}, 422, "PORT_REQUIRED"),
        ({"port": "Amsterdam", "vessel": SUDESTADA}, 422, "PORT_NOT_COVERED"),
        (
            {"port": "Durban", "vessel": {"vessel": {"name": "No tonnage"}}},
            422,
            "INSUFFICIENT_VESSEL_DATA",
        ),
        ({"port": "Durban", "vessel": SUDESTADA, "charge_ids": ["x"]}, 422, "UNKNOWN_CHARGE"),
    ],
)
async def test_bad_requests_are_rejected_and_recorded(
    client, seeded, db_session, body, status, code
):
    response = await _post(client, **body)

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    calculation = await db_session.scalar(select(Calculation))
    assert (calculation.status, calculation.error_code) == ("failed", code)


async def test_a_request_needs_a_vessel_or_a_query(client):
    response = await _post(client, port="Durban")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


async def test_model_outage_is_a_503(client, app_instance, seeded):
    _llm(app_instance).script("parse_query", LLMUnavailableError("provider down"))
    response = await _post(client, query="SUDESTADA at Durban")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "LLM_UNAVAILABLE"


async def test_slow_calculations_time_out(client, seeded, monkeypatch, db_session):
    monkeypatch.setattr(settings, "calculation_timeout_seconds", 0.000001)
    response = await _post(client, port="Durban", vessel=SUDESTADA)
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "CALCULATION_TIMEOUT"
    assert await db_session.scalar(select(func.count()).select_from(AgentStep)) == 0


async def test_unknown_calculation_is_404(client):
    response = await client.get("/v1/calculations/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "CALCULATION_NOT_FOUND"
