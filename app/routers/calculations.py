import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, Query, Request

from app.agent.graph import CalculationInput
from app.config import settings
from app.dependencies import get_calculations
from app.schemas import (
    AgentStepOut,
    CalculationOut,
    CalculationRecordOut,
    CalculationRequest,
    CalculationSummaryOut,
)
from app.services.calculations import CalculationService

router = APIRouter(prefix="/v1/calculations", tags=["calculations"])


def _openapi_examples() -> dict[str, Any]:
    """Example requests for Swagger, from EXAMPLES_DIR (outside app/)."""
    path = Path(settings.examples_dir) / "calculation_requests.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


_EXAMPLES = _openapi_examples()


@router.post("", response_model=CalculationOut)
async def calculate(
    request: Request,
    body: CalculationRequest = Body(openapi_examples=_EXAMPLES),
    service: CalculationService = Depends(get_calculations),
) -> CalculationOut:
    """Price every charge the tariff sets for a vessel call.

    Give the vessel as a profile JSON, as a plain-language query, or both.
    Each line item comes with its formula, the tariff text it rests on, the
    facts decided for the call and the assumptions made. The first call for a
    port compiles its rules (minutes); later calls reuse them (seconds)."""
    calculation = CalculationInput(
        port=body.port,
        vessel=body.vessel,
        query=body.query,
        document_id=body.document_id,
        num_services=body.overrides.num_services,
        fact_overrides={name: _fact_text(value) for name, value in body.overrides.facts.items()},
        charge_ids=body.charge_ids,
        requested_charge_ids=body.requested_charges,
        refresh_rules=body.refresh_rules,
    )
    return await service.calculate(calculation, request_id=request.state.request_id)


@router.get("", response_model=list[CalculationSummaryOut])
async def list_calculations(
    port: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: CalculationService = Depends(get_calculations),
) -> list[CalculationSummaryOut]:
    rows, _ = await service.list(port=port, status=status, limit=limit, offset=offset)
    return [
        CalculationSummaryOut(
            calculation_id=row.id,
            created_at=row.created_at,
            status=row.status,
            port=row.port_key,
            vessel_name=(row.vessel_call or {}).get("vessel", {}).get("name"),
            total=row.total,
            latency_ms=row.latency_ms,
            error_code=row.error_code,
        )
        for row in rows
    ]


@router.get("/{calculation_id}", response_model=CalculationRecordOut)
async def get_calculation(
    calculation_id: uuid.UUID, service: CalculationService = Depends(get_calculations)
) -> CalculationRecordOut:
    """A stored calculation with the agent's full trace."""
    calculation, steps = await service.get(calculation_id)
    return CalculationRecordOut(
        calculation_id=calculation.id,
        created_at=calculation.created_at,
        status=calculation.status,
        error_code=calculation.error_code,
        port=calculation.port_key,
        query=calculation.query_text,
        latency_ms=calculation.latency_ms,
        result=calculation.result,
        steps=[
            AgentStepOut(
                charge_id=step.charge_id,
                node=step.node,
                tool=step.tool,
                input_summary=step.input_summary,
                output=step.output,
                latency_ms=step.latency_ms,
            )
            for step in steps
        ],
    )


def _fact_text(value: bool | int | float | str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
