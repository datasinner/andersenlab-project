import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.rules.dsl import ChargeRule, Condition, FixedFee
from tests.unit.rule_builders import flag, make_rule, make_rule_data, per_unit


def test_amounts_accept_printed_numbers_and_serialize_as_strings():
    rule = make_rule(
        components=[{"kind": "fixed", "id": "flat", "label": "Flat", "amount": "30 960.46"}],
        minimum="1 000",
    )
    component = rule.components[0]
    assert isinstance(component, FixedFee)
    assert component.amount == Decimal("30960.46")

    dumped = json.loads(rule.model_dump_json())
    assert dumped["components"][0]["amount"] == "30960.46"
    assert dumped["minimum"] == "1000"
    assert ChargeRule.model_validate_json(rule.model_dump_json()) == rule


def test_amounts_accept_json_numbers():
    assert make_rule(minimum=5).minimum == Decimal(5)


def test_amounts_reject_unparseable_values():
    with pytest.raises(ValidationError):
        make_rule(components=[{"kind": "fixed", "id": "flat", "label": "Flat", "amount": "lots"}])


def test_condition_values_keep_their_json_type():
    assert Condition(fact="x", op="eq", value=True).value is True
    assert Condition(fact="x", op="lt", value="12").value == "12"
    assert Condition(fact="x", op="in", value=["a", "b"]).value == ["a", "b"]


def test_in_needs_a_list_and_only_in_takes_one():
    with pytest.raises(ValidationError, match="'in' needs a list"):
        Condition(fact="x", op="in", value="a")
    with pytest.raises(ValidationError, match="only 'in' takes a list"):
        Condition(fact="x", op="eq", value=["a"])


def test_conditions_must_name_a_declared_fact_or_a_quantity():
    make_rule(applies_when=[{"fact": "time_in_port_hours", "op": "lt", "value": "12"}])
    make_rule(
        applies_when=[{"fact": "is_coaster", "op": "eq", "value": True}],
        facts=[flag("is_coaster")],
    )
    with pytest.raises(ValidationError, match="neither a declared fact nor a Basis quantity"):
        make_rule(applies_when=[{"fact": "is_coaster", "op": "eq", "value": True}])


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"components": []}, "a priced rule needs at least one component"),
        ({"minimum": "10", "maximum": "5"}, "maximum is below minimum"),
        ({"components": [per_unit("a"), per_unit("a")]}, "component ids must be unique"),
        ({"facts": [flag("f"), flag("f")]}, "fact names must be unique"),
        ({"facts": [flag("gross_tonnage")]}, "may not reuse the name of a Basis quantity"),
        (
            {
                "adjustments": [
                    {
                        "id": "cut",
                        "kind": "reduction",
                        "description": "cut",
                        "percent": "10",
                        "applies_to": ["missing"],
                        "when": [{"fact": "gross_tonnage", "op": "gt", "value": "0"}],
                    }
                ]
            },
            "targets unknown components",
        ),
        ({"citations": []}, "at least 1 item"),
        ({"charge_id": "Not An Id"}, "should match pattern"),
        ({"currency": "RAND"}, "at most 3 characters"),
    ],
)
def test_inconsistent_rules_are_rejected(fields, message):
    with pytest.raises(ValidationError, match=message):
        make_rule(**fields)


def test_deductions_must_name_a_number():
    component = per_unit(basis="time_in_port_hours", unit_size="24")
    component["units"]["less"] = "is_working"
    with pytest.raises(ValidationError, match="neither a number fact nor a Basis quantity"):
        make_rule(components=[component], facts=[flag("is_working")])
    component["units"]["less"] = "call_window_days"
    assert make_rule(components=[component]).components[0].units.less == "call_window_days"


def test_an_unpriced_case_needs_a_condition():
    with pytest.raises(ValidationError):
        make_rule(components=[per_unit(), {"kind": "unpriced", "id": "u", "label": "U"}])


def test_unpriced_rules_need_no_components():
    rule = make_rule(status="on_application", components=[])
    assert rule.components == []


@pytest.mark.parametrize(
    ("bands", "message"),
    [
        (
            [{"lower": "100", "upper": "50", "base_fee": "1"}],
            "band upper is below its lower bound",
        ),
        (
            [
                {"lower": "100", "upper": "200", "base_fee": "1"},
                {"lower": "0", "upper": "100", "base_fee": "1"},
            ],
            "ascending lower",
        ),
    ],
)
def test_bands_must_be_ordered(bands, message):
    component = {"kind": "banded", "id": "b", "label": "B", "basis": "gross_tonnage"}
    with pytest.raises(ValidationError, match=message):
        make_rule(components=[{**component, "bands": bands}])


@pytest.mark.parametrize(
    ("tiers", "message"),
    [
        (
            [
                {"up_to": None, "rate": "1", "rounding": "ceil"},
                {"up_to": "100", "rate": "1", "rounding": "ceil"},
            ],
            "only the last tier may be unbounded",
        ),
        (
            [
                {"up_to": "200", "rate": "1", "rounding": "ceil"},
                {"up_to": "100", "rate": "1", "rounding": "ceil"},
            ],
            "must strictly increase",
        ),
        (
            [
                {"up_to": "100", "rate": "1", "rounding": "ceil"},
                {"up_to": "100", "rate": "1", "rounding": "ceil"},
            ],
            "must strictly increase",
        ),
        (
            [
                {"up_to": "100", "rate": "1", "rounding": "ceil"},
                {"up_to": "50", "width": "50", "rate": "1", "rounding": "ceil"},
            ],
            "either up_to or width, not both",
        ),
        (
            [
                {"up_to": "100", "rate": "1", "rounding": "ceil"},
                {"up_to": None, "width": "50", "rate": "1", "rounding": "ceil"},
                {"up_to": "120", "rate": "1", "rounding": "ceil"},
            ],
            "must strictly increase",
        ),
    ],
)
def test_tiers_must_be_ordered(tiers, message):
    component = {"kind": "tiered", "id": "t", "label": "T", "basis": "gross_tonnage"}
    with pytest.raises(ValidationError, match=message):
        make_rule(components=[{**component, "tiers": tiers}])


def test_all_conditions_and_fact_spec_lookup():
    data = make_rule_data(
        applies_when=[{"fact": "a", "op": "eq", "value": True}],
        exemptions=[{"description": "ex", "when": [{"fact": "b", "op": "eq", "value": True}]}],
        components=[per_unit(when=[{"fact": "c", "op": "eq", "value": True}])],
        adjustments=[
            {
                "id": "adj",
                "kind": "surcharge",
                "description": "adj",
                "percent": "5",
                "when": [{"fact": "d", "op": "eq", "value": True}],
            }
        ],
        facts=[flag("a"), flag("b"), flag("c"), flag("d")],
    )
    rule = ChargeRule.model_validate(data)
    assert [condition.fact for condition in rule.all_conditions()] == ["a", "b", "c", "d"]
    assert rule.fact_spec("c").name == "c"
    assert rule.fact_spec("missing") is None


def test_any_of_groups_are_validated_and_opened_up():
    group = {
        "any_of": [
            {"fact": "a", "op": "eq", "value": True},
            {"fact": "gross_tonnage", "op": "gt", "value": "50"},
        ]
    }
    rule = make_rule(applies_when=[group], facts=[flag("a")])
    assert [condition.fact for condition in rule.all_conditions()] == ["a", "gross_tonnage"]

    with pytest.raises(ValidationError, match="neither a declared fact"):
        make_rule(applies_when=[group])
    with pytest.raises(ValidationError):
        make_rule(applies_when=[{"any_of": group["any_of"][:1]}], facts=[flag("a")])
