"""resolve_facts: the qualitative facts every rule depends on.

The facts of all compiled rules for the call are decided by structured LLM
calls over batches of rules, run concurrently. (Low reasoning effort was
tried; it misread compound descriptions such as "self-propelled vessels
... at their registered port", so these calls keep the default.) The model
sees the vessel call and each fact's description in the tariff's words, and
answers only the facts that differ from their defaults (most facts of an
ordinary call take their defaults, so this keeps answers short); the rest are
filled in with their defaults here. If a batch fails, its facts keep their
rule defaults and a warning says so; pricing still happens.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from app.agent.state import AgentStep
from app.domain.numbers import format_number
from app.domain.vessel import ResolvedQuantities, VesselCall
from app.llm.client import LLMClient
from app.llm.prompts import PROMPTS
from app.llm.resilience import LLMError
from app.rules.dsl import AnyOf, ChargeRule, Condition, Conditions, FactSpec, flatten


class FactDecision(BaseModel):
    charge_id: str
    fact: str
    value: str
    source: Literal["vessel_data", "presumed", "default"]
    reason: str


class FactDecisions(BaseModel):
    facts: list[FactDecision]


@dataclass(frozen=True)
class ResolvedFact:
    charge_id: str
    fact: str
    value: str
    source: str  # vessel_data | presumed | default | override
    reason: str


# Facts per resolution call (AGENT_FACTS_PER_BATCH). Every call repeats the
# instructions and the call description (about 1.3k tokens), so bigger batches
# cost fewer tokens; smaller ones answer sooner, since batches run
# concurrently. A rule's facts stay together (one rule with more facts gets a
# batch of its own).
FACTS_PER_BATCH = 40


async def resolve_facts(
    *,
    llm: LLMClient,
    port: str,
    vessel_call: VesselCall,
    quantities: ResolvedQuantities,
    rules: dict[str, ChargeRule],
    overrides: dict[str, str],
    facts_per_batch: int = FACTS_PER_BATCH,
) -> dict:
    """{"facts": [ResolvedFact], "warnings": [...], "steps": [...]}"""
    resolved = [
        ResolvedFact(
            charge_id, spec.name, overrides[spec.name], "override", "Given in the request."
        )
        for charge_id, rule in rules.items()
        for spec in rule.facts
        if spec.name in overrides
    ]
    pending = {
        charge_id: [spec for spec in rule.facts if spec.name not in overrides]
        for charge_id, rule in rules.items()
    }
    pending = {charge_id: specs for charge_id, specs in pending.items() if specs}
    if not pending:
        return {"facts": resolved, "warnings": [], "steps": []}

    batches: list[dict[str, tuple[ChargeRule, list[FactSpec]]]] = [{}]
    size = 0
    for charge_id, specs in pending.items():
        if batches[-1] and size + len(specs) > facts_per_batch:
            batches.append({})
            size = 0
        batches[-1][charge_id] = (rules[charge_id], specs)
        size += len(specs)
    call_text = describe_call(vessel_call, quantities)
    results = await asyncio.gather(
        *(_resolve_batch(llm, port, call_text, batch) for batch in batches),
        return_exceptions=True,
    )

    warnings: list[str] = []
    steps: list[AgentStep] = []
    for batch, result in zip(batches, results, strict=True):
        if isinstance(result, LLMError):
            warnings.append(
                f"Facts for {', '.join(batch)} could not be resolved ({result}); "
                "they took their rule defaults."
            )
            steps.append(AgentStep(node="resolve_facts", output={"error": str(result)}))
            continue
        if isinstance(result, BaseException):
            raise result
        facts, step = result
        resolved.extend(facts)
        steps.append(step)
    return {"facts": resolved, "warnings": warnings, "steps": steps}


async def _resolve_batch(
    llm: LLMClient,
    port: str,
    call_text: str,
    batch: dict[str, tuple[ChargeRule, list[FactSpec]]],
) -> tuple[list[ResolvedFact], AgentStep]:
    lines = []
    for charge_id, (rule, specs) in batch.items():
        lines.append(f"{rule.name} [{charge_id}]:")
        for spec in specs:
            default = "none" if spec.default_value is None else str(spec.default_value).lower()
            lines.append(f"- {spec.name} ({spec.type}, default {default}): {spec.description}")
            lines.extend(f"    {usage}" for usage in fact_usages(rule, spec.name))
    messages = PROMPTS["resolve_facts"].render(
        port=port, vessel_call=call_text, facts="\n".join(lines)
    )
    started = time.monotonic()
    result = await llm.generate_structured(FactDecisions, messages, name="resolve_facts")
    known = {(charge_id, spec.name) for charge_id, (_, specs) in batch.items() for spec in specs}
    facts = [
        ResolvedFact(d.charge_id, d.fact, d.value, d.source, d.reason)
        for d in result.value.facts
        if (d.charge_id, d.fact) in known
    ]
    answered = {(fact.charge_id, fact.fact) for fact in facts}
    facts += [
        ResolvedFact(charge_id, spec.name, _text(spec.default_value), "default", "")
        for charge_id, (_, specs) in batch.items()
        for spec in specs
        if (charge_id, spec.name) not in answered and spec.default_value is not None
    ]
    step = AgentStep(
        node="resolve_facts",
        input_summary=f"{len(known)} fact(s) for {', '.join(batch)}",
        output={
            "decided_from_the_call": [
                f"{f.charge_id}.{f.fact}={f.value} ({f.source})"
                for f in facts
                if f.source != "default"
            ]
        },
        latency_ms=int((time.monotonic() - started) * 1000),
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
    )
    return facts, step


def _text(value: bool | str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def fact_usages(rule: ChargeRule, fact: str) -> list[str]:
    """How the rule uses a fact, in words: which rate, condition, exemption or
    adjustment depends on which value. Context for deciding the fact, since a
    description quoted from a tariff can be ambiguous on its own."""
    usages = []
    for item in _mentioning(rule.applies_when, fact):
        usages.append(f"the charge applies only when {_requires(item)}")
    for exemption in rule.exemptions:
        for item in _mentioning(exemption.when, fact):
            usages.append(f"exempt ({exemption.description}) when {_requires(item)}")
    for component in rule.components:
        for item in _mentioning(component.when, fact):
            usages.append(f"rate '{component.label}' is used when {_requires(item)}")
    for adjustment in rule.adjustments:
        for item in _mentioning(adjustment.when, fact):
            usages.append(
                f"{adjustment.kind} of {adjustment.percent}% ({adjustment.description}) "
                f"when {_requires(item)}"
            )
    return usages


def _mentioning(items: Conditions, fact: str) -> list[Condition | AnyOf]:
    return [item for item in items if any(c.fact == fact for c in flatten([item]))]


def _requires(item: Condition | AnyOf) -> str:
    if isinstance(item, AnyOf):
        return " or ".join(_requires(condition) for condition in item.any_of)
    condition = item
    value = condition.value
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif isinstance(value, list):
        value = "one of " + ", ".join(value)
    return f"{condition.fact} {condition.op.value} {value}"


def describe_call(vessel_call: VesselCall, quantities: ResolvedQuantities) -> str:
    """The call as the fact resolver sees it: the data as given, plus the
    quantities the engine will use."""
    given = vessel_call.model_dump(mode="json", exclude_none=True)
    lines = [json.dumps(given, indent=1, ensure_ascii=False), "", "Quantities:"]
    for basis, value in quantities.values.items():
        lines.append(f"- {basis.value}: {format_number(value)}")
    return "\n".join(lines)
