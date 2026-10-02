"""The canonical vessel call and the quantities a tariff rule may use.

Nothing here knows about any port or tariff. It describes a ship and a
port call, maps common input formats onto that description, and derives
the numeric quantities (the Basis vocabulary) that rules are written in.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.numbers import format_number

DEFAULT_NUM_SERVICES = 2
_SECONDS_PER_DAY = Decimal(86400)
_HOURS_PER_DAY = Decimal(24)


class Basis(StrEnum):
    """Numeric quantities a tariff rule may reference."""

    GROSS_TONNAGE = "gross_tonnage"
    NET_TONNAGE = "net_tonnage"
    DEADWEIGHT = "deadweight"
    LOA_M = "loa_m"
    BEAM_M = "beam_m"
    DRAFT_M = "draft_m"
    CARGO_TONNES = "cargo_tonnes"
    TIME_IN_PORT_HOURS = "time_in_port_hours"
    TIME_IN_PORT_DAYS = "time_in_port_days"
    CALL_WINDOW_DAYS = "call_window_days"
    NUM_SERVICES = "num_services"
    PASSENGERS = "passengers"


# Shown to the model whenever it has to pick a basis, so the meaning of each
# quantity is defined in exactly one place.
BASIS_DESCRIPTIONS: dict[Basis, str] = {
    Basis.GROSS_TONNAGE: "Gross tonnage (GT) from the vessel's tonnage certificate.",
    Basis.NET_TONNAGE: "Net tonnage (NT) from the vessel's tonnage certificate.",
    Basis.DEADWEIGHT: "Deadweight tonnage (DWT) in metric tons.",
    Basis.LOA_M: "Length overall in metres.",
    Basis.BEAM_M: "Beam (breadth) in metres.",
    Basis.DRAFT_M: "Maximum (summer) draft in metres.",
    Basis.CARGO_TONNES: "Cargo loaded or discharged on this call, in metric tons.",
    Basis.TIME_IN_PORT_HOURS: (
        "Hours the vessel spends inside port limits (alongside time when known)."
    ),
    Basis.TIME_IN_PORT_DAYS: (
        "Days the vessel spends inside port limits (alongside time when known)."
    ),
    Basis.CALL_WINDOW_DAYS: (
        "Days from reported arrival to departure; may include waiting at an outer anchorage."
    ),
    Basis.NUM_SERVICES: (
        "Number of marine-service movements on this call (entering + leaving = 2, plus shifts)."
    ),
    Basis.PASSENGERS: "Number of passengers carried.",
}


class VesselParticulars(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    vessel_type: str | None = None
    flag: str | None = None
    built_year: int | None = None
    gross_tonnage: Decimal = Field(gt=0)
    net_tonnage: Decimal | None = Field(default=None, gt=0)
    deadweight: Decimal | None = Field(default=None, gt=0)
    loa_m: Decimal | None = Field(default=None, gt=0)
    beam_m: Decimal | None = Field(default=None, gt=0)
    draft_m: Decimal | None = Field(default=None, gt=0)
    passengers: int | None = Field(default=None, ge=0)
    # Particulars with no slot above (classification society, Suez tonnage,
    # ...). Kept so the agent can still reason about them.
    additional: dict[str, Any] = Field(default_factory=dict)


class CallDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    port: str | None = None
    arrival: datetime | None = None
    departure: datetime | None = None
    days_alongside: Decimal | None = Field(default=None, ge=0)
    activity: str | None = None
    cargo_tonnes: Decimal | None = Field(default=None, ge=0)
    num_operations: int | None = Field(default=None, ge=0)
    num_holds: int | None = Field(default=None, ge=0)
    # Explicit number of marine-service movements; overrides the default.
    num_services: int | None = Field(default=None, ge=0)
    additional: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _departure_after_arrival(self) -> "CallDetails":
        if self.arrival and self.departure and self.departure < self.arrival:
            raise ValueError("departure must not be before arrival")
        return self


class VesselCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vessel: VesselParticulars
    call: CallDetails = Field(default_factory=CallDetails)

    @classmethod
    def from_profile(cls, data: Mapping[str, Any], port: str | None = None) -> "VesselCall":
        """Build a VesselCall from either the canonical shape
        ({"vessel": ..., "call": ...}) or the sectioned profile shape
        ({"vessel_metadata", "technical_specs", "operational_data"})."""
        sectioned_keys = {"vessel_metadata", "technical_specs", "operational_data"}
        if sectioned_keys & data.keys():
            vessel_call = cls._from_sectioned_profile(data)
        else:
            vessel_call = cls.model_validate(data)
        if port:
            vessel_call.call.port = port
        return vessel_call

    @classmethod
    def _from_sectioned_profile(cls, data: Mapping[str, Any]) -> "VesselCall":
        metadata = dict(data.get("vessel_metadata") or {})
        specs = dict(data.get("technical_specs") or {})
        operations = dict(data.get("operational_data") or {})

        vessel = {
            "name": metadata.pop("name", None),
            "flag": metadata.pop("flag", None),
            "built_year": metadata.pop("built_year", None),
            "vessel_type": specs.pop("type", None),
            "gross_tonnage": specs.pop("gross_tonnage", None),
            "net_tonnage": specs.pop("net_tonnage", None),
            "deadweight": specs.pop("dwt", None),
            "loa_m": specs.pop("loa_meters", None),
            "beam_m": specs.pop("beam_meters", None),
            "draft_m": _first_positive(specs.pop("draft_sw_s_w_t", None)),
        }
        call = {
            "arrival": operations.pop("arrival_time", None),
            "departure": operations.pop("departure_time", None),
            "days_alongside": operations.pop("days_alongside", None),
            "activity": operations.pop("activity", None),
            "cargo_tonnes": operations.pop("cargo_quantity_mt", None),
            "num_operations": operations.pop("num_operations", None),
            "num_holds": operations.pop("num_holds", None),
            "additional": _without_nulls(operations),
        }
        vessel["additional"] = _without_nulls(metadata | specs)
        return cls.model_validate({"vessel": vessel, "call": call})


def _first_positive(values: Any) -> Any:
    """Drafts often come as [summer, winter, tropical] with zeros for
    unknowns; the first positive value is the one to use."""
    if not isinstance(values, list):
        return values
    for value in values:
        if value:
            return value
    return None


def _without_nulls(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class ResolvedQuantities:
    """Quantities for one call, plus the assumption behind each derived one.

    The engine reports an assumption only when a rule actually used that
    quantity, so line items don't carry irrelevant caveats.
    """

    values: dict[Basis, Decimal]
    assumptions: dict[Basis, str] = field(default_factory=dict)


def resolve_quantities(vessel_call: VesselCall) -> ResolvedQuantities:
    vessel = vessel_call.vessel
    call = vessel_call.call
    values: dict[Basis, Decimal] = {}
    assumptions: dict[Basis, str] = {}

    direct = {
        Basis.GROSS_TONNAGE: vessel.gross_tonnage,
        Basis.NET_TONNAGE: vessel.net_tonnage,
        Basis.DEADWEIGHT: vessel.deadweight,
        Basis.LOA_M: vessel.loa_m,
        Basis.BEAM_M: vessel.beam_m,
        Basis.DRAFT_M: vessel.draft_m,
        Basis.CARGO_TONNES: call.cargo_tonnes,
        Basis.PASSENGERS: Decimal(vessel.passengers) if vessel.passengers is not None else None,
    }
    for basis, value in direct.items():
        if value is not None:
            values[basis] = value

    window_days: Decimal | None = None
    if call.arrival and call.departure:
        seconds = Decimal(int((call.departure - call.arrival).total_seconds()))
        window_days = seconds / _SECONDS_PER_DAY
        values[Basis.CALL_WINDOW_DAYS] = window_days

    time_in_port_days: Decimal | None = None
    time_in_port_note = ""
    if call.days_alongside is not None:
        time_in_port_days = call.days_alongside
        note = f"Time in port taken as the {format_number(call.days_alongside)} days alongside"
        if window_days is not None:
            note += (
                f"; the arrival-to-departure window of {format_number(round(window_days, 2))}"
                " days may include waiting outside port limits"
            )
        time_in_port_note = note + "."
    elif window_days is not None:
        time_in_port_days = window_days
        time_in_port_note = (
            "Time in port taken as the arrival-to-departure window of "
            f"{format_number(round(window_days, 2))} days (days alongside not given)."
        )
    if time_in_port_days is not None:
        values[Basis.TIME_IN_PORT_DAYS] = time_in_port_days
        values[Basis.TIME_IN_PORT_HOURS] = time_in_port_days * _HOURS_PER_DAY
        assumptions[Basis.TIME_IN_PORT_DAYS] = time_in_port_note
        assumptions[Basis.TIME_IN_PORT_HOURS] = time_in_port_note

    if call.num_services is not None:
        values[Basis.NUM_SERVICES] = Decimal(call.num_services)
    else:
        values[Basis.NUM_SERVICES] = Decimal(DEFAULT_NUM_SERVICES)
        assumptions[Basis.NUM_SERVICES] = (
            f"{DEFAULT_NUM_SERVICES} marine-service movements assumed (entering and leaving)."
        )

    return ResolvedQuantities(values=values, assumptions=assumptions)
