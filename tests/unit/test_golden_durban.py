"""Golden test: hand-written TNPA rules for Durban, evaluated for SUDESTADA.

The rule files under tests/fixtures/rules/durban are test data written by a
human from the TNPA Tariff Book 2024/25 (BUILD_PLAN §13). They pin the
engine's arithmetic independently of the LLM: later phases must compile
rules that produce the same amounts.
"""

import json
from decimal import Decimal
from pathlib import Path

import pytest

from app.domain.vessel import VesselCall, resolve_quantities
from app.rules.dsl import ChargeRule
from app.rules.engine import LineItemStatus, evaluate_rule
from app.rules.grounding import check_grounding

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
RULE_FILES = sorted((FIXTURES / "rules" / "durban").glob("*.json"))
FACTS = {"is_cargo_working": True}

# What the engine must compute from the tariff as written (BUILD_PLAN §13).
EXPECTED = {
    "light_dues": Decimal("60062.04"),
    "port_dues": Decimal("199371.35"),
    "towage_dues": Decimal("147074.38"),
    "vts_dues": Decimal("33345.00"),
    "pilotage_dues": Decimal("47189.94"),
    "berthing_services": Decimal("19639.50"),
    "running_of_vessel_lines": Decimal("3309.12"),
}

# The reference values from the brief, keyed to the charge they correspond
# to. "Running Lines" in the brief equals §3.8 berthing services.
REFERENCE = {
    "light_dues": Decimal("60062.04"),
    "port_dues": Decimal("199549.22"),
    "towage_dues": Decimal("147074.38"),
    "vts_dues": Decimal("33315.75"),
    "pilotage_dues": Decimal("47189.94"),
    "berthing_services": Decimal("19639.50"),
}


def _excerpts() -> dict[int, str]:
    raw = json.loads((FIXTURES / "tnpa_excerpts.json").read_text())
    return {int(chunk_id): excerpt["text"] for chunk_id, excerpt in raw.items()}


def _rules() -> dict[str, ChargeRule]:
    rules = [ChargeRule.model_validate_json(path.read_text()) for path in RULE_FILES]
    return {rule.charge_id: rule for rule in rules}


def _line_items():
    profile = json.loads((FIXTURES / "vessels" / "sudestada.json").read_text())
    quantities = resolve_quantities(VesselCall.from_profile(profile, port="Durban"))
    return {
        charge_id: evaluate_rule(rule, quantities, FACTS) for charge_id, rule in _rules().items()
    }


def test_every_golden_rule_is_present():
    assert set(_rules()) == set(EXPECTED)


@pytest.mark.parametrize("path", RULE_FILES, ids=lambda path: path.stem)
def test_golden_rule_is_grounded_in_the_tariff_text(path):
    rule = ChargeRule.model_validate_json(path.read_text())
    assert check_grounding(rule, _excerpts()) == []


def test_engine_reproduces_the_hand_computed_amounts():
    items = _line_items()
    amounts = {charge_id: item.amount for charge_id, item in items.items()}
    assert amounts == EXPECTED
    assert all(item.status == LineItemStatus.CHARGED for item in items.values())


def test_amounts_are_within_one_percent_of_the_reference():
    items = _line_items()
    for charge_id, reference in REFERENCE.items():
        relative_error = abs(items[charge_id].amount - reference) / reference
        assert relative_error < Decimal("0.01"), charge_id


def test_towage_trace_shows_the_band_and_per_service_multiplier():
    formula = _line_items()["towage_dues"].formula
    assert formula[0].endswith("73,118.07 + 419.12 = 73,537.19")
    assert "× 2 (num_services) = 147,074.38" in formula


def test_port_dues_explain_the_time_in_port_assumption():
    assumptions = _line_items()["port_dues"].assumptions
    assert any("3.39 days alongside" in assumption for assumption in assumptions)
