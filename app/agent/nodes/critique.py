"""critique: a second model call reviews the rule against its sources.

The critic also sees what the engine computes for one illustrative vessel
(fixed, made-up particulars; defaults for every fact), which makes
structural mistakes visible: a missing multiplier, the wrong band, a
surcharge that always applies.
"""

from decimal import Decimal

from app.agent.state import AgentStep, CompileState, Critique, render_excerpts
from app.domain.vessel import Basis, ResolvedQuantities
from app.llm.client import LLMClient, LLMOutputError
from app.llm.prompts import PROMPTS
from app.llm.prompts.rule_semantics import quantity_glossary
from app.rules.dsl import ChargeRule
from app.rules.engine import RuleEvaluationError, evaluate_rule

# Illustrative particulars only: they let the critic see the engine at work.
PROBE = ResolvedQuantities(
    values={
        Basis.GROSS_TONNAGE: Decimal("25000"),
        Basis.NET_TONNAGE: Decimal("12000"),
        Basis.DEADWEIGHT: Decimal("40000"),
        Basis.LOA_M: Decimal("190"),
        Basis.BEAM_M: Decimal("32"),
        Basis.DRAFT_M: Decimal("11"),
        Basis.CARGO_TONNES: Decimal("30000"),
        Basis.TIME_IN_PORT_HOURS: Decimal("50"),
        Basis.TIME_IN_PORT_DAYS: Decimal("50") / Decimal("24"),
        Basis.CALL_WINDOW_DAYS: Decimal("3"),
        Basis.NUM_SERVICES: Decimal("2"),
        Basis.PASSENGERS: Decimal("0"),
    }
)
PROBE_DESCRIPTION = (
    "GT 25,000, NT 12,000, LOA 190 m, 30,000 t of cargo, 50 hours in port, 2 services, "
    "every fact at its default"
)


async def critique(state: CompileState, *, llm: LLMClient) -> dict:
    charge = state["charge"]
    rule = state["rule"]
    assert rule is not None
    messages = PROMPTS["critique"].render(
        quantities=quantity_glossary(),
        port=state["port"],
        charge_name=charge.name,
        rule_json=rule.model_dump_json(indent=1, exclude_defaults=True),
        probe_description=PROBE_DESCRIPTION,
        example=example_evaluation(rule),
        excerpts=render_excerpts(state["evidence"]),
    )
    try:
        result = await llm.generate_structured(Critique, messages, name="critique")
    except LLMOutputError as exc:
        # An unreadable review is no review: keep the rule, flagged.
        step = AgentStep(node="critique", charge_id=charge.charge_id, output={"error": str(exc)})
        return {
            "outcome": "low_confidence",
            "review_notes": [f"The review could not be read: {exc}"],
            "steps": [step],
        }

    review = result.value
    step = AgentStep(
        node="critique",
        charge_id=charge.charge_id,
        output={"blocking": review.blocking, "minor": review.minor},
        latency_ms=result.latency_ms,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
    )
    update: dict = {"critique": review, "review_notes": review.minor, "steps": [step]}
    if review.blocking:
        update.update(feedback=review.blocking, revisions=state.get("revisions", 0) + 1)
    else:
        update.update(outcome="approved", feedback=[])
    return update


def example_evaluation(rule: ChargeRule) -> str:
    try:
        item = evaluate_rule(rule, PROBE)
    except RuleEvaluationError as exc:
        return f"The engine could not evaluate the rule: {exc}"
    lines = [f"status: {item.status.value}"]
    if item.amount is not None:
        lines.append(f"amount: {item.amount} {item.currency}")
    if item.reason:
        lines.append(f"reason: {item.reason}")
    lines.extend(f"  {line}" for line in item.formula)
    return "\n".join(lines)
