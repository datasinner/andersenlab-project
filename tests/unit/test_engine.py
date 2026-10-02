from decimal import Decimal

import pytest

from app.domain.vessel import Basis
from app.rules.engine import (
    LineItemStatus,
    MissingFactError,
    MissingQuantityError,
    RuleEvaluationError,
    evaluate_rule,
)
from tests.unit.rule_builders import CITATION, flag, make_rule, per_unit, quantities

GT_1050 = quantities(gross_tonnage=1050)


def _adjustment(adjustment_id: str, kind: str, percent: str, **extra) -> dict:
    return {
        "id": adjustment_id,
        "kind": kind,
        "description": adjustment_id.replace("_", " "),
        "percent": percent,
        "when": [{"fact": "gross_tonnage", "op": "gt", "value": "0"}],
        **extra,
    }


# -- components ------------------------------------------------------------


def test_fixed_fee():
    item = evaluate_rule(make_rule(), GT_1050)
    assert item.status == LineItemStatus.CHARGED
    assert item.amount == Decimal("100.00")
    assert item.formula == ["Flat fee: 100", "Amount (rounded to cents): 100.00"]


@pytest.mark.parametrize(
    ("rounding", "expected", "expression"),
    [
        ("ceil", "22.00", "ceil(1,050 / 100) = 11"),  # "or part thereof"
        ("floor", "20.00", "floor(1,050 / 100) = 10"),
        ("pro_rata", "21.00", "1,050 / 100 = 10.5"),
    ],
)
def test_per_unit_rounding(rounding, expected, expression):
    item = evaluate_rule(make_rule(components=[per_unit(rounding=rounding)]), GT_1050)
    assert item.amount == Decimal(expected)
    assert expression in item.formula[0]


def test_quantity_used_as_is_gets_no_redundant_step():
    component = per_unit(rate="0.65", unit_size="1", rounding="pro_rata")
    item = evaluate_rule(make_rule(components=[component]), GT_1050)
    assert item.formula[0] == "per_ton fee: 1,050 × 0.65 = 682.5"


def test_per_unit_with_time_pro_rata():
    component = per_unit(
        rate="10",
        per_time={"basis": "time_in_port_hours", "unit_size": "24", "rounding": "pro_rata"},
    )
    item = evaluate_rule(
        make_rule(components=[component]), quantities(gross_tonnage=1000, time_in_port_hours=36)
    )
    assert item.amount == Decimal("150.00")  # 10 units × 10 × 1.5 days
    assert "36 / 24 = 1.5" in item.formula[0]


def test_units_can_deduct_a_number_fact():
    """ "Per 24 hours of the time in port less the hours worked"."""
    component = per_unit(
        rate="100",
        basis="time_in_port_hours",
        unit_size="24",
        rounding="ceil",
    )
    component["units"]["less"] = "hours_worked"
    rule = make_rule(
        components=[component],
        facts=[{"name": "hours_worked", "type": "number", "description": "Hours worked"}],
    )
    worked_half = evaluate_rule(rule, quantities(time_in_port_hours=81), {"hours_worked": "40"})
    assert worked_half.amount == Decimal("200.00")  # ceil((81 − 40) / 24) = 2
    assert "ceil((81 − 40 hours_worked) / 24) = 2" in worked_half.formula[0]
    worked_all = evaluate_rule(rule, quantities(time_in_port_hours=81), {"hours_worked": "81"})
    assert worked_all.amount == Decimal("0.00")


def test_units_above_an_offset_never_go_negative():
    component = per_unit()
    component["units"]["above"] = "5 000"
    item = evaluate_rule(make_rule(components=[component]), quantities(gross_tonnage=3000))
    assert item.amount == Decimal("0.00")


BANDED = {
    "kind": "banded",
    "id": "banded",
    "label": "Fee by band",
    "basis": "gross_tonnage",
    "bands": [
        {"lower": "0", "upper": "2000", "base_fee": "100"},
        {
            "lower": "2001",
            "upper": "10000",
            "base_fee": "200",
            "increment": {
                "rate": "3",
                "units": {
                    "basis": "gross_tonnage",
                    "unit_size": "100",
                    "rounding": "ceil",
                    "above": "2000",
                },
            },
        },
        {"lower": "10000", "upper": "50000", "base_fee": "500"},
        {"lower": "50000", "upper": None, "base_fee": "900"},
    ],
}


@pytest.mark.parametrize(
    ("gross_tonnage", "expected"),
    [
        (2000, "100.00"),
        (2001, "203.00"),  # 200 + ceil(1 / 100) × 3
        (2150, "206.00"),  # 200 + ceil(150 / 100) × 3
        (10000, "440.00"),  # overlapping bound: the first listed band wins
        (10001, "500.00"),
        (250000, "900.00"),  # unbounded top band
    ],
)
def test_banded_fee_picks_the_first_band_containing_the_quantity(gross_tonnage, expected):
    item = evaluate_rule(make_rule(components=[BANDED]), quantities(gross_tonnage=gross_tonnage))
    assert item.amount == Decimal(expected)


def test_banded_fee_explains_the_band():
    item = evaluate_rule(make_rule(components=[BANDED]), quantities(gross_tonnage=2150))
    assert "gross_tonnage 2,150 is in band 2,001–10,000" in item.formula[0]
    assert "ceil((2,150 − 2,000) / 100) = 2" in item.formula[0]
    unbounded = evaluate_rule(make_rule(components=[BANDED]), quantities(gross_tonnage=60000))
    assert "50,000–∞" in unbounded.formula[0]


def test_banded_fee_without_a_matching_band_is_an_error():
    component = {**BANDED, "bands": [{"lower": "100", "upper": "200", "base_fee": "1"}]}
    with pytest.raises(RuleEvaluationError, match="no band contains gross_tonnage = 50"):
        evaluate_rule(make_rule(components=[component]), quantities(gross_tonnage=50))


TIERED = {
    "kind": "tiered",
    "id": "tiered",
    "label": "Marginal tiers",
    "basis": "gross_tonnage",
    "tiers": [
        {"up_to": "17700", "rate": "50.56", "unit_size": "100", "rounding": "ceil"},
        {"up_to": "35300", "rate": "33.45", "unit_size": "100", "rounding": "ceil"},
        {"up_to": "53000", "rate": "16.82", "unit_size": "100", "rounding": "ceil"},
        {"up_to": None, "rate": "0", "unit_size": "100", "rounding": "ceil"},
    ],
}


@pytest.mark.parametrize(
    ("gross_tonnage", "expected"),
    [
        (10000, "5056.00"),  # 100 × 50.56, first tier only
        (51300, "17527.52"),  # 177 × 50.56 + 176 × 33.45 + 160 × 16.82
        (60000, "17813.46"),  # 177 × 50.56 + 176 × 33.45 + 177 × 16.82 + 70 × 0
    ],
)
def test_tiered_fee_charges_each_slice_at_its_own_rate(gross_tonnage, expected):
    item = evaluate_rule(make_rule(components=[TIERED]), quantities(gross_tonnage=gross_tonnage))
    assert item.amount == Decimal(expected)


def test_tiers_can_be_given_by_width():
    """ "Free for 30 days, the next 90 days at 2.82, the following 90 days at
    5.56, thereafter 11.14" per metre per day: widths, not printed bounds."""
    component = {
        "kind": "tiered",
        "id": "stay",
        "label": "Stay by days",
        "basis": "time_in_port_days",
        "tiers": [
            {"up_to": "30", "rate": "0", "rounding": "ceil"},
            {"up_to": None, "width": "90", "rate": "2.82", "rounding": "ceil"},
            {"up_to": None, "width": "90", "rate": "5.56", "rounding": "ceil"},
            {"up_to": None, "rate": "11.14", "rounding": "ceil"},
        ],
    }
    item = evaluate_rule(make_rule(components=[component]), quantities(time_in_port_days=250))
    # 30 free, 90 × 2.82, 90 × 5.56, 40 × 11.14
    assert item.amount == Decimal("1199.80")


def test_tiered_fee_with_a_bounded_last_tier_ignores_the_excess():
    component = {**TIERED, "tiers": TIERED["tiers"][:1]}
    item = evaluate_rule(make_rule(components=[component]), quantities(gross_tonnage=20000))
    assert item.amount == Decimal("8949.12")  # 177 × 50.56; nothing above 17 700


# -- totals ------------------------------------------------------------------


def test_components_are_summed():
    rule = make_rule(
        components=[{"kind": "fixed", "id": "a", "label": "A", "amount": "10"}, per_unit()]
    )
    item = evaluate_rule(rule, GT_1050)
    assert item.amount == Decimal("32.00")
    assert "Subtotal: 32" in item.formula


@pytest.mark.parametrize(
    ("fields", "expected", "line"),
    [
        ({"minimum": "50"}, "50.00", "Minimum fee applies: 50"),
        ({"maximum": "15"}, "15.00", "Maximum fee applies: 15"),
        ({"minimum": "10", "maximum": "30"}, "22.00", None),
    ],
)
def test_minimum_and_maximum_clamp(fields, expected, line):
    item = evaluate_rule(make_rule(components=[per_unit()], **fields), GT_1050)
    assert item.amount == Decimal(expected)
    if line:
        assert line in item.formula


def test_multiplier_applies_after_the_minimum():
    rule = make_rule(
        components=[per_unit()],
        minimum="50",
        multiplier={"basis": "num_services", "unit_size": "1", "rounding": "ceil"},
    )
    item = evaluate_rule(rule, quantities(gross_tonnage=1050, num_services=2))
    assert item.amount == Decimal("100.00")
    assert "× 2 (num_services) = 100" in item.formula


def test_result_is_rounded_half_up_to_cents():
    rule = make_rule(components=[{"kind": "fixed", "id": "f", "label": "F", "amount": "0.125"}])
    assert evaluate_rule(rule, GT_1050).amount == Decimal("0.13")


# -- applicability -----------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected_status", "reason"),
    [
        ("on_application", LineItemStatus.ON_APPLICATION, "Quoted on application."),
        (
            "not_priced_in_document",
            LineItemStatus.NOT_PRICED,
            "The document does not state a rate for this charge.",
        ),
        ("not_applicable_at_port", LineItemStatus.NOT_APPLICABLE, "Not charged at Exampleville."),
    ],
)
def test_unpriced_rules_pass_their_status_through(status, expected_status, reason):
    item = evaluate_rule(make_rule(status=status, components=[]), GT_1050)
    assert item.status == expected_status
    assert item.amount is None
    assert item.reason == reason


def test_unpriced_rule_reason_uses_its_notes():
    rule = make_rule(status="on_application", components=[], notes=["Ask the port authority."])
    assert evaluate_rule(rule, GT_1050).reason == "Ask the port authority."


def test_an_unpriced_case_reports_the_charge_not_priced_for_that_call():
    rule = make_rule(
        components=[
            per_unit(),
            {
                "kind": "unpriced",
                "id": "coasters",
                "label": "Coasters pay under a special agreement",
                "when": [{"fact": "is_coaster", "op": "eq", "value": True}],
            },
        ],
        facts=[flag("is_coaster")],
    )
    coaster = evaluate_rule(rule, GT_1050, {"is_coaster": True})
    assert coaster.status == LineItemStatus.NOT_PRICED
    assert coaster.reason == (
        "Coasters pay under a special agreement: the document gives no rate for this case."
    )
    other = evaluate_rule(rule, GT_1050, {"is_coaster": False})
    assert other.amount == Decimal("22.00")


def test_matching_exemption_makes_the_charge_not_applicable():
    exemption_citation = {**CITATION, "quote": "Naval vessels are exempt"}
    rule = make_rule(
        exemptions=[
            {
                "description": "Naval vessels",
                "when": [{"fact": "is_naval", "op": "eq", "value": True}],
                "citation": exemption_citation,
            },
            {
                "description": "Tiny vessels",
                "when": [{"fact": "gross_tonnage", "op": "lt", "value": "10"}],
            },
        ],
        facts=[flag("is_naval")],
    )
    exempt = evaluate_rule(rule, GT_1050, {"is_naval": True})
    assert exempt.status == LineItemStatus.NOT_APPLICABLE
    assert exempt.reason == "Exempt: Naval vessels"
    assert exempt.citations[0].quote == "Naval vessels are exempt"

    tiny = evaluate_rule(rule, quantities(gross_tonnage=5), {"is_naval": False})
    assert tiny.reason == "Exempt: Tiny vessels"
    assert tiny.citations == rule.citations

    assert evaluate_rule(rule, GT_1050, {"is_naval": False}).status == LineItemStatus.CHARGED


def test_failed_applies_when_condition_explains_itself():
    rule = make_rule(
        applies_when=[{"fact": "is_tanker", "op": "eq", "value": True}],
        facts=[flag("is_tanker")],
    )
    item = evaluate_rule(rule, GT_1050, {"is_tanker": False})
    assert item.status == LineItemStatus.NOT_APPLICABLE
    assert item.reason == "Not applicable: requires is_tanker applies = yes"


def test_components_with_unmet_conditions_are_skipped():
    rule = make_rule(
        components=[
            per_unit(
                "registered", rate="5", when=[{"fact": "registered", "op": "eq", "value": True}]
            ),
            per_unit(
                "visiting", rate="2", when=[{"fact": "registered", "op": "eq", "value": False}]
            ),
        ],
        facts=[flag("registered")],
    )
    item = evaluate_rule(rule, GT_1050, {"registered": False})
    assert item.amount == Decimal("22.00")
    assert item.formula[0] == "registered fee: not applicable to this call"


def test_no_applicable_component_means_not_applicable():
    rule = make_rule(
        components=[per_unit(when=[{"fact": "registered", "op": "eq", "value": True}])],
        facts=[flag("registered")],
    )
    item = evaluate_rule(rule, GT_1050)
    assert item.status == LineItemStatus.NOT_APPLICABLE
    assert item.reason == "No pricing component of this charge applies to the vessel call"


# -- adjustments -------------------------------------------------------------


def test_reduction_on_the_whole_charge():
    rule = make_rule(components=[per_unit()], adjustments=[_adjustment("cut", "reduction", "35")])
    item = evaluate_rule(rule, GT_1050)
    assert item.amount == Decimal("14.30")  # 22 − 35%
    assert item.adjustments[0].amount == Decimal("-7.70")
    assert "Reduction 35% (cut): −7.7" in item.formula


def test_surcharge_on_one_component_scales_with_the_multiplier():
    rule = make_rule(
        components=[{"kind": "fixed", "id": "base", "label": "Base", "amount": "100"}, per_unit()],
        multiplier={"basis": "num_services", "unit_size": "1", "rounding": "ceil"},
        adjustments=[_adjustment("late", "surcharge", "50", applies_to=["per_ton"])],
    )
    item = evaluate_rule(rule, quantities(gross_tonnage=1050, num_services=2))
    # (100 + 22) × 2 = 244, plus 50% of (22 × 2)
    assert item.amount == Decimal("266.00")
    assert "Surcharge 50% (late): +22" in item.formula


def test_only_the_largest_adjustment_in_an_exclusive_group_applies():
    rule = make_rule(
        adjustments=[
            _adjustment("coaster", "reduction", "35", exclusive_group="base"),
            _adjustment("bunkers_only", "reduction", "60", exclusive_group="base"),
            _adjustment("passenger", "reduction", "35", exclusive_group="base"),
            _adjustment("short_stay", "reduction", "15"),
        ]
    )
    item = evaluate_rule(rule, GT_1050)
    assert [adjustment.id for adjustment in item.adjustments] == ["bunkers_only", "short_stay"]
    assert item.amount == Decimal("25.00")  # 100 − 60% − 15%


def test_adjustments_never_make_a_charge_negative():
    rule = make_rule(
        adjustments=[
            _adjustment("full", "reduction", "100"),
            _adjustment("extra", "reduction", "15"),
        ]
    )
    assert evaluate_rule(rule, GT_1050).amount == Decimal("0.00")


def test_adjustment_conditions_can_use_quantities():
    short_stay = _adjustment("short_stay", "reduction", "15")
    short_stay["when"] = [{"fact": "time_in_port_hours", "op": "lt", "value": "12"}]
    rule = make_rule(adjustments=[short_stay])

    assert evaluate_rule(rule, quantities(gross_tonnage=1, time_in_port_hours=6)).amount == Decimal(
        "85.00"
    )
    assert evaluate_rule(
        rule, quantities(gross_tonnage=1, time_in_port_hours=12)
    ).amount == Decimal("100.00")


# -- facts and quantities ------------------------------------------------------


def test_missing_quantity_is_an_error():
    with pytest.raises(MissingQuantityError, match="no value for 'loa_m'"):
        evaluate_rule(make_rule(components=[per_unit(basis="loa_m")]), GT_1050)


def test_fact_without_value_or_default_is_an_error():
    rule = make_rule(
        applies_when=[{"fact": "is_tanker", "op": "eq", "value": True}],
        facts=[flag("is_tanker", default=None)],
    )
    with pytest.raises(MissingFactError, match="is_tanker"):
        evaluate_rule(rule, GT_1050)


def test_fact_default_is_used_and_recorded_as_an_assumption():
    rule = make_rule(
        applies_when=[{"fact": "is_tanker", "op": "eq", "value": False}],
        facts=[flag("is_tanker", default=False)],
    )
    item = evaluate_rule(rule, GT_1050)
    assert item.status == LineItemStatus.CHARGED
    assert item.assumptions == ["is_tanker applies: assumed no."]


def test_numeric_fact_default_is_described_as_a_number():
    rule = make_rule(
        applies_when=[{"fact": "passengers_on_board", "op": "le", "value": "12"}],
        facts=[
            {
                "name": "passengers_on_board",
                "type": "number",
                "description": "Passengers on board",
                "default_value": "0",
            }
        ],
    )
    item = evaluate_rule(rule, GT_1050)
    assert item.assumptions == ["Passengers on board: assumed 0."]


def test_quantity_assumptions_are_reported_only_when_used():
    assumptions = {Basis.NUM_SERVICES: "2 movements assumed."}
    values = quantities(assumptions=assumptions, gross_tonnage=1050, num_services=2)

    assert evaluate_rule(make_rule(), values).assumptions == []
    with_multiplier = make_rule(
        multiplier={"basis": "num_services", "unit_size": "1", "rounding": "ceil"}
    )
    assert evaluate_rule(with_multiplier, values).assumptions == ["2 movements assumed."]


@pytest.mark.parametrize(
    ("fact_type", "condition", "supplied", "holds"),
    [
        ("bool", {"op": "eq", "value": True}, "yes", True),
        ("bool", {"op": "ne", "value": "true"}, False, True),
        ("bool", {"op": "eq", "value": False}, "No", True),
        ("number", {"op": "ge", "value": "12"}, "12", True),
        ("number", {"op": "le", "value": "12"}, "13", False),
        ("number", {"op": "gt", "value": "1 000"}, 1001, True),
        ("number", {"op": "eq", "value": "5"}, Decimal("5.0"), True),
        ("number", {"op": "ne", "value": "5"}, "6", True),
        ("number", {"op": "in", "value": ["1", "2"]}, "2", True),
        ("text", {"op": "eq", "value": "Tanker"}, "tanker", True),
        ("text", {"op": "ne", "value": "tanker"}, "bulk carrier", True),
        ("text", {"op": "in", "value": ["Tanker", "Gas Carrier"]}, "gas carrier", True),
    ],
)
def test_conditions_coerce_facts_by_their_declared_type(fact_type, condition, supplied, holds):
    rule = make_rule(
        applies_when=[{"fact": "subject", **condition}],
        facts=[{"name": "subject", "type": fact_type, "description": "subject"}],
    )
    item = evaluate_rule(rule, GT_1050, {"subject": supplied})
    assert (item.status == LineItemStatus.CHARGED) is holds


@pytest.mark.parametrize(
    ("fact_type", "condition", "supplied", "message"),
    [
        ("bool", {"op": "lt", "value": "1"}, True, "is yes/no; it can't be 'lt'"),
        ("bool", {"op": "eq", "value": True}, "maybe", "needs a yes/no value"),
        ("number", {"op": "eq", "value": "1"}, "lots", "needs a number"),
        ("number", {"op": "eq", "value": "1"}, True, "needs a number"),
        ("text", {"op": "gt", "value": "a"}, "b", "is text; it can't be 'gt'"),
    ],
)
def test_conditions_reject_mismatched_types(fact_type, condition, supplied, message):
    rule = make_rule(
        applies_when=[{"fact": "subject", **condition}],
        facts=[{"name": "subject", "type": fact_type, "description": "subject"}],
    )
    with pytest.raises(RuleEvaluationError, match=message):
        evaluate_rule(rule, GT_1050, {"subject": supplied})


def test_quantity_conditions_support_in():
    rule = make_rule(applies_when=[{"fact": "num_services", "op": "in", "value": ["1", "3"]}])
    assert evaluate_rule(rule, quantities(gross_tonnage=1, num_services=3)).amount is not None
    not_listed = evaluate_rule(rule, quantities(gross_tonnage=1, num_services=2))
    assert not_listed.reason == "Not applicable: requires num_services in 1, 3"
