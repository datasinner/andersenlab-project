"""The calculation graph: one vessel call → priced line items.

normalize_input → resolve_document → screen_charges ─Send()→ compile_charge (one per charge,
                                                              in parallel; cached rules return
                                                              at once, others run the
                                                              compile graph)
                                   → resolve_facts (one batched call) → compute → end
"""

import operator
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.nodes.facts import ResolvedFact, resolve_facts
from app.agent.nodes.normalize import normalize_input
from app.agent.nodes.screen import ScreenedCharge, screen_charges
from app.agent.state import AgentStep, ChargeSpec
from app.config import settings
from app.domain.vessel import ResolvedQuantities, VesselCall
from app.llm.client import LLMClient
from app.models import ChargeCatalogueEntry, TariffDocument
from app.rules.engine import LineItem, LineItemStatus, RuleEvaluationError, evaluate_rule
from app.services.documents import resolve_document_for_port
from app.services.rulebook import CompileOutcome, RulebookService, UnknownChargeError


@dataclass(frozen=True)
class CalculationInput:
    port: str | None = None
    vessel: dict[str, Any] | None = None
    query: str | None = None
    document_id: uuid.UUID | None = None
    num_services: int | None = None
    fact_overrides: dict[str, str] = field(default_factory=dict)
    charge_ids: list[str] | None = None  # price only these charges
    requested_charge_ids: list[str] = field(default_factory=list)  # on-request charges to include
    refresh_rules: bool = False


@dataclass(frozen=True)
class PricedCharge:
    outcome: CompileOutcome
    item: LineItem
    facts: list[ResolvedFact]


@dataclass(frozen=True)
class FailedCharge:
    charge_id: str
    name: str
    error: str


class CalculationState(TypedDict, total=False):
    request: CalculationInput
    port: str
    vessel_call: VesselCall
    quantities: ResolvedQuantities
    document: TariffDocument
    port_key: str
    candidates: list[ChargeSpec]
    screened: list[ScreenedCharge]
    compiled: Annotated[list[CompileOutcome], operator.add]
    facts: list[ResolvedFact]
    priced: list[PricedCharge]
    failed: list[FailedCharge]
    warnings: Annotated[list[str], operator.add]
    steps: Annotated[list[AgentStep], operator.add]


def build_calculation_graph(
    llm: LLMClient,
    rulebook: RulebookService,
    session_factory: async_sessionmaker[AsyncSession],
) -> CompiledStateGraph:
    async def normalize_node(state: CalculationState) -> dict:
        request = state["request"]
        return await normalize_input(
            llm=llm,
            port=request.port,
            vessel=request.vessel,
            query=request.query,
            num_services=request.num_services,
        )

    async def resolve_document_node(state: CalculationState) -> dict:
        arrival = state["vessel_call"].call.arrival
        async with session_factory() as session:
            document, port_key = await resolve_document_for_port(
                session,
                state["port"],
                document_id=state["request"].document_id,
                on_date=arrival.date() if arrival else None,
            )
        return {"document": document, "port_key": port_key}

    async def screen_node(state: CalculationState) -> dict:
        request = state["request"]
        async with session_factory() as session:
            entries = list(
                await session.scalars(
                    select(ChargeCatalogueEntry)
                    .where(ChargeCatalogueEntry.document_id == state["document"].id)
                    .order_by(ChargeCatalogueEntry.id)
                )
            )
        known = {entry.charge_id for entry in entries}
        asked = (request.charge_ids or []) + request.requested_charge_ids
        unknown = [charge_id for charge_id in asked if charge_id not in known]
        if unknown:
            raise UnknownChargeError(f"Not in this document's catalogue: {', '.join(unknown)}")
        candidates, screened = screen_charges(
            entries, only=request.charge_ids, requested=request.requested_charge_ids
        )
        step = AgentStep(
            node="screen_charges",
            output={"candidates": [c.charge_id for c in candidates], "screened_out": len(screened)},
        )
        return {"candidates": candidates, "screened": screened, "steps": [step]}

    def fan_out(state: CalculationState) -> list[Send] | str:
        if not state["candidates"]:
            return "resolve_facts"
        return [
            Send(
                "compile_charge",
                {
                    "charge": charge,
                    "document": state["document"],
                    "port_key": state["port_key"],
                    "refresh": state["request"].refresh_rules,
                },
            )
            for charge in state["candidates"]
        ]

    async def compile_charge_node(payload: dict) -> dict:
        outcome = await rulebook.compile_charge(
            payload["document"], payload["port_key"], payload["charge"], refresh=payload["refresh"]
        )
        return {"compiled": [outcome], "steps": outcome.steps}

    async def resolve_facts_node(state: CalculationState) -> dict:
        rules = {o.charge_id: o.rule for o in state.get("compiled", []) if o.rule is not None}
        return await resolve_facts(
            llm=llm,
            port=state["port_key"],
            vessel_call=state["vessel_call"],
            quantities=state["quantities"],
            rules=rules,
            overrides=state["request"].fact_overrides,
            facts_per_batch=settings.agent_facts_per_batch,
        )

    graph = StateGraph(CalculationState)
    graph.add_node("normalize_input", normalize_node)
    graph.add_node("resolve_document", resolve_document_node)
    graph.add_node("screen_charges", screen_node)
    graph.add_node("compile_charge", compile_charge_node)
    graph.add_node("resolve_facts", resolve_facts_node)
    graph.add_node("compute", compute)
    graph.add_edge(START, "normalize_input")
    graph.add_edge("normalize_input", "resolve_document")
    graph.add_edge("resolve_document", "screen_charges")
    graph.add_conditional_edges("screen_charges", fan_out, ["compile_charge", "resolve_facts"])
    graph.add_edge("compile_charge", "resolve_facts")
    graph.add_edge("resolve_facts", "compute")
    graph.add_edge("compute", END)
    return graph.compile()


def compute(state: CalculationState) -> dict:
    """Price every compiled rule with the engine. A charge whose rule failed
    to compile or to evaluate is reported, never silently dropped."""
    facts_by_charge: dict[str, list[ResolvedFact]] = {}
    for fact in state.get("facts", []):
        facts_by_charge.setdefault(fact.charge_id, []).append(fact)

    priced: list[PricedCharge] = []
    failed: list[FailedCharge] = []
    outcomes = sorted(state.get("compiled", []), key=lambda outcome: _section_order(outcome))
    for outcome in outcomes:
        if outcome.rule is None:
            error = outcome.error or "; ".join(outcome.issues) or "The rule could not be compiled."
            failed.append(FailedCharge(outcome.charge_id, outcome.name, error))
            continue
        facts = facts_by_charge.get(outcome.charge_id, [])
        try:
            item = evaluate_rule(
                outcome.rule, state["quantities"], {fact.fact: fact.value for fact in facts}
            )
        except RuleEvaluationError as exc:
            failed.append(FailedCharge(outcome.charge_id, outcome.name, str(exc)))
            continue
        if item.status == LineItemStatus.CHARGED and item.amount == 0:
            item = item.model_copy(
                update={
                    "status": LineItemStatus.NOT_APPLICABLE,
                    "reason": "Computes to zero for this call.",
                }
            )
        priced.append(PricedCharge(outcome, item, facts))
    return {"priced": priced, "failed": failed}


def _section_order(outcome: CompileOutcome) -> tuple:
    refs = outcome.rule.section_refs if outcome.rule else []
    first = refs[0] if refs else "~"
    return tuple(int(part) if part.isdigit() else 0 for part in first.split(".")), outcome.charge_id
