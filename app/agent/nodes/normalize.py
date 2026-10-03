"""normalize_input: request → VesselCall + quantities.

A natural-language query is read by one structured LLM call; a profile JSON
is mapped deterministically. When both are given, the JSON wins field by
field and the query fills the gaps. Anything else the query says about the
call ("calls only for bunkers") is kept as statements for fact resolution.
"""

import time
from typing import Any

from pydantic import BaseModel, ValidationError

from app.agent.state import AgentStep
from app.domain.numbers import NumberFormatError, parse_number
from app.domain.vessel import VesselCall, profile_to_canonical, resolve_quantities
from app.errors import AppError
from app.llm.client import LLMClient
from app.llm.prompts import PROMPTS


class InsufficientVesselDataError(AppError):
    status_code = 422
    code = "INSUFFICIENT_VESSEL_DATA"


class PortRequiredError(AppError):
    status_code = 422
    code = "PORT_REQUIRED"


class ExtractedCall(BaseModel):
    """A vessel call as read from a natural-language request."""

    port: str | None
    vessel_name: str | None
    vessel_type: str | None
    flag: str | None
    built_year: int | None
    gross_tonnage: str | None
    net_tonnage: str | None
    deadweight: str | None
    loa_m: str | None
    beam_m: str | None
    draft_m: str | None
    passengers: int | None
    arrival: str | None
    departure: str | None
    days_alongside: str | None
    activity: str | None
    cargo_tonnes: str | None
    num_services: int | None
    statements: list[str]


_VESSEL_FIELDS = {
    "vessel_name": "name",
    "vessel_type": "vessel_type",
    "flag": "flag",
    "built_year": "built_year",
    "gross_tonnage": "gross_tonnage",
    "net_tonnage": "net_tonnage",
    "deadweight": "deadweight",
    "loa_m": "loa_m",
    "beam_m": "beam_m",
    "draft_m": "draft_m",
    "passengers": "passengers",
}
_CALL_FIELDS = (
    "arrival",
    "departure",
    "days_alongside",
    "activity",
    "cargo_tonnes",
    "num_services",
)
_NUMERIC = {"gross_tonnage", "net_tonnage", "deadweight", "loa_m", "beam_m", "draft_m"}
_NUMERIC |= {"days_alongside", "cargo_tonnes"}


async def normalize_input(
    *,
    llm: LLMClient,
    port: str | None,
    vessel: dict[str, Any] | None,
    query: str | None,
    num_services: int | None,
) -> dict:
    data: dict[str, dict[str, Any]] = {"vessel": {}, "call": {}}
    steps: list[AgentStep] = []
    warnings: list[str] = []
    statements: list[str] = []

    if query:
        started = time.monotonic()
        result = await llm.generate_structured(
            ExtractedCall, PROMPTS["parse_query"].render(query=query), name="parse_query"
        )
        extracted = result.value
        data, dropped = _canonical(extracted)
        statements = extracted.statements
        warnings.extend(
            f"Ignored unreadable {name} in the query: {value!r}" for name, value in dropped
        )
        steps.append(
            AgentStep(
                node="normalize_input",
                output=extracted.model_dump(exclude_none=True),
                latency_ms=int((time.monotonic() - started) * 1000),
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
            )
        )
        port = port or extracted.port

    if vessel:
        given = profile_to_canonical(vessel)
        for part in ("vessel", "call"):
            data[part].update(
                {key: value for key, value in given[part].items() if value is not None}
            )

    if not port:
        raise PortRequiredError("Say which port the vessel calls at (the port field or the query).")
    if num_services is not None:
        data["call"]["num_services"] = num_services
    if statements:
        additional = dict(data["call"].get("additional") or {})
        additional["statements"] = statements
        data["call"]["additional"] = additional
    data["call"]["port"] = port

    try:
        vessel_call = VesselCall.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise InsufficientVesselDataError(
            f"The vessel call is incomplete or invalid: {problems}"
        ) from exc

    return {
        "vessel_call": vessel_call,
        "quantities": resolve_quantities(vessel_call),
        "port": port,
        "warnings": warnings,
        "steps": steps,
    }


def _canonical(extracted: ExtractedCall) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str]]]:
    data: dict[str, dict[str, Any]] = {"vessel": {}, "call": {}}
    dropped: list[tuple[str, str]] = []
    values = extracted.model_dump()
    for source, target in _VESSEL_FIELDS.items():
        _put(data["vessel"], target, source, values[source], dropped)
    for name in _CALL_FIELDS:
        _put(data["call"], name, name, values[name], dropped)
    return data, dropped


def _put(target: dict, key: str, name: str, value: Any, dropped: list[tuple[str, str]]) -> None:
    if value is None:
        return
    if name in _NUMERIC:
        try:
            value = parse_number(str(value))
        except NumberFormatError:
            dropped.append((name, str(value)))
            return
    target[key] = value
