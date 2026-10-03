"""Agent node logic that needs no database: citation repair, how facts are
presented to the resolver, and fact batching."""

from decimal import Decimal

from app.agent.nodes.facts import FACTS_PER_BATCH, fact_usages, resolve_facts
from app.agent.nodes.validate import repair_citations
from app.agent.state import Excerpt
from app.domain.vessel import Basis, VesselCall, resolve_quantities
from app.llm.client import FakeLLMClient
from app.llm.resilience import LLMUnavailableError
from app.rules.dsl import ChargeRule
from tests.unit.rule_builders import CITATION, flag, make_rule, per_unit

EVIDENCE = [
    Excerpt(1, "9.9", 1, None, "Example fee per 100 tons"),
    Excerpt(2, "9.9", 2, "12", "A surcharge of 25% applies at night."),
]


def test_a_quote_in_another_evidence_chunk_gets_its_pointer_fixed():
    rule = make_rule(
        citations=[{**CITATION, "quote": "A surcharge of 25% applies"}],
        adjustments=[
            {
                "id": "night",
                "kind": "surcharge",
                "description": "night",
                "percent": "25",
                "when": [{"fact": "gross_tonnage", "op": "gt", "value": "0"}],
                "citation": {**CITATION, "quote": "a surcharge of 25 % applies at night"},
            }
        ],
    )
    repaired = repair_citations(rule, EVIDENCE)
    assert (repaired.citations[0].chunk_id, repaired.citations[0].page) == (2, 2)
    assert repaired.adjustments[0].citation.chunk_id == 2


def test_correct_and_unfound_quotes_are_left_alone():
    rule = make_rule(
        citations=[
            {**CITATION, "quote": "Example fee"},
            {**CITATION, "quote": "a sentence nowhere in the evidence"},
        ]
    )
    repaired = repair_citations(rule, EVIDENCE)
    assert [citation.chunk_id for citation in repaired.citations] == [1, 1]


def _lighting_rule() -> ChargeRule:
    return make_rule(
        applies_when=[{"fact": "enters_port", "op": "eq", "value": True}],
        exemptions=[
            {"description": "Naval vessels", "when": [{"fact": "naval", "op": "eq", "value": True}]}
        ],
        components=[
            per_unit("by_length", when=[{"fact": "home_port", "op": "eq", "value": True}]),
            per_unit("by_tonnage", when=[{"fact": "home_port", "op": "eq", "value": False}]),
        ],
        adjustments=[
            {
                "id": "short",
                "kind": "reduction",
                "description": "short stay",
                "percent": "15",
                "when": [{"fact": "naval", "op": "ne", "value": True}],
            }
        ],
        facts=[flag("enters_port", True), flag("naval"), flag("home_port")],
    )


def test_fact_usages_say_what_each_value_selects():
    rule = _lighting_rule()
    assert fact_usages(rule, "home_port") == [
        "rate 'by_length fee' is used when home_port eq true",
        "rate 'by_tonnage fee' is used when home_port eq false",
    ]
    assert fact_usages(rule, "naval") == [
        "exempt (Naval vessels) when naval eq true",
        "reduction of 15% (short stay) when naval ne true",
    ]
    assert fact_usages(rule, "enters_port") == ["the charge applies only when enters_port eq true"]


def test_fact_usages_show_the_whole_any_of_group():
    rule = make_rule(
        applies_when=[
            {
                "any_of": [
                    {"fact": "loa_m", "op": "gt", "value": "90"},
                    {"fact": "requested", "op": "eq", "value": True},
                ]
            }
        ],
        facts=[flag("requested")],
    )
    assert fact_usages(rule, "requested") == [
        "the charge applies only when loa_m gt 90 or requested eq true"
    ]


def _call():
    vessel_call = VesselCall.from_profile({"vessel": {"gross_tonnage": 1000}}, port="Exampleville")
    return vessel_call, resolve_quantities(vessel_call)


async def test_facts_are_resolved_in_concurrent_batches_with_overrides_winning():
    rules = {
        f"charge_{n}": make_rule(
            charge_id=f"charge_{n}",
            facts=[flag(f"f{n}_{i}") for i in range(5)],
        )
        for n in range(4)
    }
    llm = FakeLLMClient()

    def decide(messages):
        prompt = messages[1].content
        decisions = []
        for charge_id, rule in rules.items():
            if f"[{charge_id}]" in prompt:
                decisions.extend(
                    {
                        "charge_id": charge_id,
                        "fact": spec.name,
                        "value": "true",
                        "source": "presumed",
                        "reason": "typical",
                    }
                    for spec in rule.facts
                )
        return {"facts": decisions}

    llm.script("resolve_facts", decide, decide)
    vessel_call, quantities = _call()

    result = await resolve_facts(
        llm=llm,
        port="Exampleville",
        vessel_call=vessel_call,
        quantities=quantities,
        rules=rules,
        overrides={"f0_0": "false"},
    )

    # 19 facts to decide, at most FACTS_PER_BATCH per call, rules kept whole.
    assert FACTS_PER_BATCH == 12
    assert len(llm.calls) == 2
    facts = {(fact.charge_id, fact.fact): (fact.value, fact.source) for fact in result["facts"]}
    assert facts[("charge_0", "f0_0")] == ("false", "override")
    assert facts[("charge_3", "f3_4")] == ("true", "presumed")
    assert len(facts) == 20


async def test_a_failed_batch_keeps_defaults_and_warns():
    rules = {"charge_0": make_rule(facts=[flag("f0")])}
    llm = FakeLLMClient()
    llm.script("resolve_facts", LLMUnavailableError("down"))
    vessel_call, quantities = _call()

    result = await resolve_facts(
        llm=llm, port="P", vessel_call=vessel_call, quantities=quantities, rules=rules, overrides={}
    )

    assert result["facts"] == []
    assert result["warnings"] == [
        "Facts for charge_0 could not be resolved (down); they took their rule defaults."
    ]


def test_passengers_default_to_none_with_an_assumption():
    _, quantities = _call()
    assert quantities.values[Basis.PASSENGERS] == Decimal(0)
    assert quantities.assumptions[Basis.PASSENGERS] == "No passengers stated; none assumed."
