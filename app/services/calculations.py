"""Run the calculation graph for a vessel call, persist it, and read it back.

Every calculation is written, failures included (with their error code), so
GET /v1/calculations shows what was asked and what went wrong. The agent's
steps (tool calls, extractions, reviews, fact decisions) are written as
agent_step rows: the audit trail of how each figure was found.
"""

import asyncio
import json
import time
import uuid
from decimal import Decimal
from typing import Any

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.graph import CalculationInput, CalculationState, build_calculation_graph
from app.agent.state import AgentStep
from app.config import settings
from app.domain.numbers import round_money
from app.errors import AppError
from app.llm.client import LLMClient
from app.llm.resilience import LLMError, LLMRateLimitedError, LLMTimeoutError
from app.models import AgentStep as AgentStepRow
from app.models import Calculation, CalculationStatus
from app.observability import DISABLED, Observability
from app.rules.engine import LineItemStatus
from app.schemas import (
    AdjustmentOut,
    CalculationOut,
    CitationOut,
    DocumentRef,
    FactOut,
    FailedChargeOut,
    LineItemOut,
    OtherChargeOut,
    VatOut,
)
from app.services.rulebook import RulebookService

logger = structlog.get_logger("app.calculations")


class CalculationNotFoundError(AppError):
    status_code = 404
    code = "CALCULATION_NOT_FOUND"


class CalculationTimeoutError(AppError):
    status_code = 504
    code = "CALCULATION_TIMEOUT"


class ModelUnavailableError(AppError):
    status_code = 503
    code = "LLM_UNAVAILABLE"


class ModelTimeoutError(AppError):
    status_code = 504
    code = "LLM_TIMEOUT"


class ModelRateLimitedError(AppError):
    status_code = 429
    code = "RATE_LIMITED"


class CalculationService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        llm: LLMClient,
        rulebook: RulebookService,
        observability: Observability = DISABLED,
    ) -> None:
        self._sessions = session_factory
        self._llm = llm
        self._observability = observability
        self._graph = build_calculation_graph(llm, rulebook, session_factory)

    async def calculate(self, request: CalculationInput, *, request_id: str) -> CalculationOut:
        calculation_id = uuid.uuid4()
        started = time.monotonic()
        trace = self._observability.trace(
            "calculate-port-dues",
            seed=str(calculation_id),
            metadata={
                "request_id": request_id,
                "calculation_id": str(calculation_id),
                "port": request.port or "",
            },
            tags=["calculation"],
            input_text=request.query,
        )
        try:
            with trace:
                async with asyncio.timeout(settings.calculation_timeout_seconds):
                    final: CalculationState = await self._graph.ainvoke(
                        {"request": request, "warnings": [], "steps": []}
                    )
        except TimeoutError as exc:
            error = CalculationTimeoutError(
                f"The calculation took longer than {settings.calculation_timeout_seconds} s"
            )
            await self._record_failure(calculation_id, request, request_id, started, error.code)
            raise error from exc
        except AppError as exc:
            await self._record_failure(calculation_id, request, request_id, started, exc.code)
            raise
        except LLMError as exc:
            error = _model_error(exc)
            await self._record_failure(calculation_id, request, request_id, started, error.code)
            raise error from exc

        latency_ms = int((time.monotonic() - started) * 1000)
        result = build_result(
            calculation_id, final, model=self._llm.model_name, latency_ms=latency_ms
        )
        await self._record(calculation_id, request, request_id, final, result)
        logger.info(
            "calculation_done",
            calculation_id=str(calculation_id),
            status=result.status,
            port=result.port,
            line_items=len(result.line_items),
            total=str(result.total),
            latency_ms=latency_ms,
        )
        return result

    async def get(self, calculation_id: uuid.UUID) -> tuple[Calculation, list[AgentStepRow]]:
        async with self._sessions() as session:
            calculation = await session.get(Calculation, calculation_id)
            if calculation is None:
                raise CalculationNotFoundError(f"No calculation with id {calculation_id}")
            steps = list(
                await session.scalars(
                    select(AgentStepRow)
                    .where(AgentStepRow.calculation_id == calculation_id)
                    .order_by(AgentStepRow.id)
                )
            )
        return calculation, steps

    async def list(
        self, *, port: str | None, status: str | None, limit: int, offset: int
    ) -> tuple[list[Calculation], int]:
        query = select(Calculation)
        if port:
            query = query.where(Calculation.port_key == port)
        if status:
            query = query.where(Calculation.status == status)
        async with self._sessions() as session:
            total = await session.scalar(select(func.count()).select_from(query.subquery()))
            rows = await session.scalars(
                query.order_by(Calculation.created_at.desc()).limit(limit).offset(offset)
            )
            return list(rows), total or 0

    async def _record(
        self,
        calculation_id: uuid.UUID,
        request: CalculationInput,
        request_id: str,
        final: CalculationState,
        result: CalculationOut,
    ) -> None:
        async with self._sessions() as session:
            session.add(
                Calculation(
                    id=calculation_id,
                    request_id=request_id,
                    document_id=result.document.id,
                    port_key=result.port,
                    vessel_call=result.vessel,
                    query_text=request.query,
                    status=result.status,
                    result=result.model_dump(mode="json"),
                    total=result.total,
                    model=result.model,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    latency_ms=result.latency_ms,
                )
            )
            await session.flush()
            session.add_all(_step_rows(calculation_id, final.get("steps", [])))
            await session.commit()

    async def _record_failure(
        self,
        calculation_id: uuid.UUID,
        request: CalculationInput,
        request_id: str,
        started: float,
        error_code: str,
    ) -> None:
        latency_ms = int((time.monotonic() - started) * 1000)
        logger.warning(
            "calculation_failed", calculation_id=str(calculation_id), error_code=error_code
        )
        async with self._sessions() as session:
            session.add(
                Calculation(
                    id=calculation_id,
                    request_id=request_id,
                    port_key=request.port,
                    vessel_call=_json_safe(request.vessel) if request.vessel else None,
                    query_text=request.query,
                    status=CalculationStatus.FAILED,
                    error_code=error_code,
                    model=self._llm.model_name,
                    latency_ms=latency_ms,
                )
            )
            await session.commit()


def build_result(
    calculation_id: uuid.UUID, final: CalculationState, *, model: str, latency_ms: int
) -> CalculationOut:
    document = final["document"]
    line_items: list[LineItemOut] = []
    not_applicable: list[OtherChargeOut] = []
    not_priced: list[OtherChargeOut] = []
    for priced in final.get("priced", []):
        item, outcome = priced.item, priced.outcome
        if item.status == LineItemStatus.CHARGED:
            line_items.append(
                LineItemOut(
                    charge_id=item.charge_id,
                    name=item.name,
                    section_refs=outcome.section_refs or item.section_refs,
                    amount=item.amount or Decimal(0),
                    currency=item.currency,
                    confidence="high" if outcome.status == "approved" else "low",
                    formula=item.formula,
                    adjustments=[
                        AdjustmentOut(**adjustment.model_dump()) for adjustment in item.adjustments
                    ],
                    assumptions=item.assumptions,
                    facts=[
                        FactOut(name=f.fact, value=f.value, source=f.source, reason=f.reason)
                        for f in priced.facts
                    ],
                    citations=[CitationOut(**citation.model_dump()) for citation in item.citations],
                    notes=item.notes,
                    review_notes=outcome.review_notes,
                    open_issues=outcome.issues,
                    rule_source="cache" if outcome.from_cache else "compiled",
                )
            )
        else:
            other = OtherChargeOut(
                charge_id=item.charge_id,
                name=item.name,
                section_refs=outcome.section_refs or item.section_refs,
                reason=item.reason or item.status.value,
                facts=[
                    FactOut(name=f.fact, value=f.value, source=f.source, reason=f.reason)
                    for f in priced.facts
                ],
            )
            if item.status == LineItemStatus.NOT_APPLICABLE:
                not_applicable.append(other)
            else:
                not_priced.append(other)

    screened = final.get("screened", [])
    failed = [FailedChargeOut(**vars(failure)) for failure in final.get("failed", [])]
    if not failed:
        status = CalculationStatus.SUCCESS
    elif line_items:
        status = CalculationStatus.PARTIAL
    else:
        status = CalculationStatus.FAILED

    prompt_tokens = sum(step.prompt_tokens or 0 for step in final.get("steps", []))
    completion_tokens = sum(step.completion_tokens or 0 for step in final.get("steps", []))
    total = round_money(sum((item.amount for item in line_items), Decimal(0)))
    return CalculationOut(
        calculation_id=calculation_id,
        status=status,
        port=final["port_key"],
        document=DocumentRef(id=document.id, title=document.title, authority=document.authority),
        vessel=final["vessel_call"].model_dump(mode="json", exclude_none=True),
        currency=document.currency,
        vat=VatOut(rate=document.vat_rate, included=False),
        line_items=line_items,
        total=total,
        not_applicable=not_applicable,
        not_priced=not_priced,
        on_request=[_other(charge) for charge in screened if charge.group == "on_request"],
        excluded=[_other(charge) for charge in screened if charge.group == "excluded"],
        failed=failed,
        warnings=list(final.get("warnings", [])),
        model=model,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _other(charge) -> OtherChargeOut:
    return OtherChargeOut(
        charge_id=charge.charge_id,
        name=charge.name,
        section_refs=charge.section_refs,
        reason=charge.reason,
    )


def _step_rows(calculation_id: uuid.UUID, steps: list[AgentStep]) -> list[AgentStepRow]:
    return [
        AgentStepRow(
            calculation_id=calculation_id,
            charge_id=step.charge_id,
            node=step.node,
            tool=step.tool,
            input_summary=step.input_summary,
            output=_json_safe(step.output) if step.output is not None else None,
            latency_ms=step.latency_ms,
            prompt_tokens=step.prompt_tokens,
            completion_tokens=step.completion_tokens,
        )
        for step in steps
    ]


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _model_error(exc: LLMError) -> AppError:
    if isinstance(exc, LLMTimeoutError):
        return ModelTimeoutError(str(exc))
    if isinstance(exc, LLMRateLimitedError):
        return ModelRateLimitedError(str(exc))
    return ModelUnavailableError(str(exc))
