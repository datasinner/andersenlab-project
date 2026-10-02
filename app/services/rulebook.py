"""The rulebook: compiled ChargeRules per (document, port, charge), cached.

compile_port resolves the port against the document, picks the charges to
compile (by default those a vessel pays on an ordinary call: payer vessel,
triggered per call, per service or per period), and compiles them
concurrently. A cached rule is reused unless refresh is asked for; the cache
key includes the rule schema version and the prompt versions, so changing
either recompiles.

Rules that pass validation are cached, including those the critic still had
issues with after the last revision (flagged low_confidence, with the issues
kept), so a calculation never silently recompiles and changes its answer.
Rules that never passed validation are not cached.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.compile import build_compile_graph
from app.agent.state import AgentStep, ChargeSpec, CompileState
from app.agent.tools import ToolExecutor
from app.config import settings
from app.errors import AppError
from app.llm.client import LLMClient, Usage
from app.llm.embeddings import Embedder
from app.llm.prompts import versions
from app.llm.resilience import LLMError
from app.models import ChargeCatalogueEntry, CompiledRule, TariffDocument
from app.retrieval.search import TariffSearch
from app.rules.dsl import RULE_SCHEMA_VERSION, ChargeRule

logger = structlog.get_logger("app.rulebook")

COMPILE_PROMPTS = ("research", "extract_rule", "critique")
ROUTINE_TRIGGERS = {"per_call", "per_service", "per_period"}


class UnknownChargeError(AppError):
    status_code = 422
    code = "UNKNOWN_CHARGE"


@dataclass(frozen=True)
class CompileOutcome:
    charge_id: str
    name: str
    status: str  # approved | low_confidence | failed
    rule: ChargeRule | None
    from_cache: bool
    issues: list[str] = field(default_factory=list)  # open blocking problems
    review_notes: list[str] = field(default_factory=list)  # minor points from the review
    revisions: int = 0
    sections_read: list[str] = field(default_factory=list)
    research_notes: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    usage: Usage = Usage()
    latency_ms: int = 0
    model: str | None = None
    compiled_at: datetime | None = None
    error: str | None = None


@dataclass(frozen=True)
class CompileReport:
    document_id: uuid.UUID
    port_key: str
    prompt_version: str
    outcomes: list[CompileOutcome]

    @property
    def usage(self) -> Usage:
        total = Usage()
        for outcome in self.outcomes:
            total = total + outcome.usage
        return total


def prompt_version() -> str:
    return versions(*COMPILE_PROMPTS)


class RulebookService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        llm: LLMClient,
        embedder: Embedder,
    ) -> None:
        self._sessions = session_factory
        self._llm = llm
        self._search = TariffSearch(session_factory, embedder)
        self._graph = build_compile_graph(
            llm,
            max_tool_calls=settings.agent_max_tool_calls,
            max_revisions=settings.agent_max_revisions,
        )

    async def compile_port(
        self,
        document: TariffDocument,
        port_key: str,
        *,
        charge_ids: list[str] | None = None,
        include_all: bool = False,
        refresh: bool = False,
    ) -> CompileReport:
        charges = await self._select_charges(document.id, charge_ids, include_all)
        outcomes = await asyncio.gather(
            *(
                self.compile_charge(document, port_key, charge, refresh=refresh)
                for charge in charges
            )
        )
        return CompileReport(document.id, port_key, prompt_version(), list(outcomes))

    async def compile_charge(
        self, document: TariffDocument, port_key: str, charge: ChargeSpec, *, refresh: bool = False
    ) -> CompileOutcome:
        if not refresh:
            cached = await self.cached_rule(document.id, port_key, charge.charge_id)
            if cached is not None:
                return outcome_from_cache(cached, charge.name)

        log = logger.bind(charge_id=charge.charge_id, port=port_key)
        tools = ToolExecutor(self._sessions, self._search, document.id)
        initial: CompileState = {
            "charge": charge,
            "port": port_key,
            "currency": document.currency or "XXX",
            "tools": tools,
            "revisions": 0,
            "feedback": [],
            "steps": [],
        }
        started = time.monotonic()
        try:
            final = await self._graph.ainvoke(initial)
        except LLMError as exc:
            log.warning("rule_compile_failed", error=str(exc))
            return CompileOutcome(
                charge_id=charge.charge_id,
                name=charge.name,
                status="failed",
                rule=None,
                from_cache=False,
                sections_read=tools.sections_read,
                latency_ms=int((time.monotonic() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )

        steps: list[AgentStep] = final.get("steps", [])
        status = final.get("outcome", "failed")
        issues = [] if status == "approved" else list(final.get("feedback") or [])
        outcome = CompileOutcome(
            charge_id=charge.charge_id,
            name=charge.name,
            status=status,
            rule=final.get("rule"),
            from_cache=False,
            issues=issues,
            review_notes=list(final.get("review_notes") or []),
            revisions=final.get("revisions", 0),
            sections_read=tools.sections_read,
            research_notes=final.get("research_notes", ""),
            steps=steps,
            usage=_usage(steps),
            latency_ms=int((time.monotonic() - started) * 1000),
            model=self._llm.model_name,
        )
        log.info(
            "rule_compiled",
            status=status,
            revisions=outcome.revisions,
            latency_ms=outcome.latency_ms,
            prompt_tokens=outcome.usage.prompt_tokens,
            completion_tokens=outcome.usage.completion_tokens,
        )
        if outcome.rule is not None and status in ("approved", "low_confidence"):
            await self._store(document.id, port_key, outcome)
        return outcome

    async def cached_rule(
        self, document_id: uuid.UUID, port_key: str, charge_id: str
    ) -> CompiledRule | None:
        async with self._sessions() as session:
            return await session.scalar(
                _current_rules(document_id, port_key).where(CompiledRule.charge_id == charge_id)
            )

    async def cached_rules(self, document_id: uuid.UUID, port_key: str) -> list[CompiledRule]:
        async with self._sessions() as session:
            return list(await session.scalars(_current_rules(document_id, port_key)))

    async def _select_charges(
        self, document_id: uuid.UUID, charge_ids: list[str] | None, include_all: bool
    ) -> list[ChargeSpec]:
        async with self._sessions() as session:
            entries = list(
                await session.scalars(
                    select(ChargeCatalogueEntry)
                    .where(ChargeCatalogueEntry.document_id == document_id)
                    .order_by(ChargeCatalogueEntry.id)
                )
            )
        if charge_ids:
            by_id = {entry.charge_id: entry for entry in entries}
            unknown = [charge_id for charge_id in charge_ids if charge_id not in by_id]
            if unknown:
                raise UnknownChargeError(f"Not in this document's catalogue: {', '.join(unknown)}")
            return [ChargeSpec.from_catalogue(by_id[charge_id]) for charge_id in charge_ids]
        return [
            ChargeSpec.from_catalogue(entry)
            for entry in entries
            if include_all or (entry.payer == "vessel" and entry.trigger in ROUTINE_TRIGGERS)
        ]

    async def _store(self, document_id: uuid.UUID, port_key: str, outcome: CompileOutcome) -> None:
        assert outcome.rule is not None
        async with self._sessions() as session:
            await session.execute(
                delete(CompiledRule).where(
                    CompiledRule.document_id == document_id,
                    CompiledRule.port_key == port_key,
                    CompiledRule.charge_id == outcome.charge_id,
                    CompiledRule.rule_schema_version == RULE_SCHEMA_VERSION,
                    CompiledRule.prompt_version == prompt_version(),
                )
            )
            session.add(
                CompiledRule(
                    document_id=document_id,
                    port_key=port_key,
                    charge_id=outcome.charge_id,
                    rule=outcome.rule.model_dump(mode="json"),
                    rule_schema_version=RULE_SCHEMA_VERSION,
                    prompt_version=prompt_version(),
                    model=outcome.model or "unknown",
                    critic_verdict={
                        "outcome": outcome.status,
                        "issues": outcome.issues,
                        "review_notes": outcome.review_notes,
                        "revisions": outcome.revisions,
                        "sections_read": outcome.sections_read,
                        "research_notes": outcome.research_notes,
                    },
                )
            )
            await session.commit()


def _current_rules(document_id: uuid.UUID, port_key: str):
    return (
        select(CompiledRule)
        .where(
            CompiledRule.document_id == document_id,
            CompiledRule.port_key == port_key,
            CompiledRule.rule_schema_version == RULE_SCHEMA_VERSION,
            CompiledRule.prompt_version == prompt_version(),
        )
        .order_by(CompiledRule.id)
    )


def outcome_from_cache(row: CompiledRule, name: str | None = None) -> CompileOutcome:
    rule = ChargeRule.model_validate(row.rule)
    verdict = row.critic_verdict or {}
    return CompileOutcome(
        charge_id=row.charge_id,
        name=name or rule.name,
        status=verdict.get("outcome", "approved"),
        rule=rule,
        from_cache=True,
        issues=verdict.get("issues", []),
        review_notes=verdict.get("review_notes", []),
        revisions=verdict.get("revisions", 0),
        sections_read=verdict.get("sections_read", []),
        research_notes=verdict.get("research_notes", ""),
        model=row.model,
        compiled_at=row.created_at,
    )


def _usage(steps: list[AgentStep]) -> Usage:
    total = Usage()
    for step in steps:
        total = total + Usage(step.prompt_tokens or 0, step.completion_tokens or 0)
    return total
