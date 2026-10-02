from app.rules.grounding import check_grounding
from tests.unit.rule_builders import CITATION, make_rule, per_unit

CHUNKS = {
    1: "Example fee per 100 tons or part<br>thereof ……… 12 345.67\nMinimum fee 50",
    2: "A surcharge of 25% applies to vessels staying longer than 48 hours.",
}


def _messages(rule, chunks=CHUNKS) -> list[tuple[str, str]]:
    return [(issue.path, issue.message) for issue in check_grounding(rule, chunks)]


def test_rule_whose_numbers_are_all_printed_is_grounded():
    rule = make_rule(components=[per_unit(rate="12 345.67")], minimum="50")
    assert _messages(rule) == []


def test_invented_rate_is_reported_with_its_path():
    rule = make_rule(components=[per_unit(rate="12 345.76")])
    assert _messages(rule) == [
        ("components[0].rate", "12,345.76 does not appear in the cited text"),
    ]


def test_zero_and_one_are_structural_and_need_not_be_printed():
    rule = make_rule(
        components=[per_unit(rate="12345.67", unit_size="1")],
        multiplier={"basis": "num_services", "unit_size": "1", "rounding": "ceil"},
    )
    assert _messages(rule) == []


def test_quotes_must_appear_in_their_chunk_ignoring_whitespace_and_line_breaks():
    rule = make_rule(
        citations=[{**CITATION, "quote": "per 100 tons or part thereof"}],
        components=[per_unit(rate="12345.67")],
    )
    assert _messages(rule) == []

    paraphrased = make_rule(
        citations=[{**CITATION, "quote": "per hundred tons"}],
        components=[per_unit(rate="12345.67")],
    )
    assert _messages(paraphrased) == [("citations[0]", "quote not found in chunk 1")]


def test_unknown_chunks_are_reported_and_contribute_no_numbers():
    rule = make_rule(
        citations=[{**CITATION, "chunk_id": 99}],
        components=[per_unit(rate="12345.67")],
    )
    assert _messages(rule) == [
        ("citations[0]", "cites unknown chunk 99"),
        ("components[0].rate", "12,345.67 does not appear in the cited text"),
        ("components[0].units.unit_size", "100 does not appear in the cited text"),
    ]


def test_adjustment_and_exemption_citations_add_their_text_and_conditions_are_checked():
    surcharge_citation = {**CITATION, "chunk_id": 2, "quote": "A surcharge of 25% applies"}
    rule = make_rule(
        components=[per_unit(rate="12345.67")],
        adjustments=[
            {
                "id": "long_stay",
                "kind": "surcharge",
                "description": "Long stay",
                "percent": "25",
                "when": [{"fact": "time_in_port_hours", "op": "gt", "value": "48"}],
                "citation": surcharge_citation,
            }
        ],
        exemptions=[
            {
                "description": "Short stays",
                "when": [{"fact": "time_in_port_hours", "op": "lt", "value": "6"}],
                "citation": surcharge_citation,
            }
        ],
    )
    assert _messages(rule) == [
        ("exemptions[0].when[0].value", "6 does not appear in the cited text"),
    ]


def test_every_kind_of_number_is_checked():
    rule = make_rule(
        components=[
            {
                "kind": "banded",
                "id": "banded",
                "label": "Banded",
                "basis": "gross_tonnage",
                "when": [{"fact": "loa_m", "op": "in", "value": ["7", "Long"]}],
                "bands": [
                    {
                        "lower": "2",
                        "upper": "3",
                        "base_fee": "4",
                        "increment": {
                            "rate": "5",
                            "units": {"basis": "gross_tonnage", "rounding": "ceil", "above": "6"},
                        },
                    }
                ],
            },
            {
                "kind": "tiered",
                "id": "tiered",
                "label": "Tiered",
                "basis": "gross_tonnage",
                "tiers": [
                    {"up_to": "8", "rate": "9", "unit_size": "10", "rounding": "ceil"},
                    {"up_to": None, "width": "17", "rate": "11", "rounding": "ceil"},
                    {"up_to": None, "rate": "18", "rounding": "ceil"},
                ],
            },
            per_unit(
                "timed",
                rate="12",
                unit_size="13",
                per_time={"basis": "time_in_port_hours", "unit_size": "14", "rounding": "pro_rata"},
            ),
        ],
        maximum="15",
        applies_when=[{"fact": "num_services", "op": "ge", "value": "16"}],
    )
    flagged = sorted(int(message.split()[0]) for _, message in _messages(rule, {1: "Example fee"}))
    assert flagged == list(range(2, 19))
