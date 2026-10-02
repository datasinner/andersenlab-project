"""screen_charges: which catalogue charges to price for a vessel call.

Deterministic, from the catalogue's payer and trigger:
- paid by cargo or by someone other than the vessel → excluded;
- licences and permits → excluded;
- raised only on request or after an incident → listed as on request, and
  priced only if the request names them;
- everything else is a candidate. Whether a candidate applies to this call
  is then decided by its own rule's conditions, against the call's facts.
"""

from dataclasses import dataclass
from typing import Literal

from app.agent.state import ChargeSpec
from app.models import ChargeCatalogueEntry


@dataclass(frozen=True)
class ScreenedCharge:
    charge_id: str
    name: str
    section_refs: list[str]
    group: Literal["excluded", "on_request"]
    reason: str


_PAYER_REASONS = {
    "cargo": "Payable by the cargo owner, not the vessel.",
    "other": "Not a charge on the vessel (a fee for other parties or services).",
}


def screen_charges(
    entries: list[ChargeCatalogueEntry],
    *,
    only: list[str] | None = None,
    requested: list[str] | None = None,
) -> tuple[list[ChargeSpec], list[ScreenedCharge]]:
    """(candidates, screened out). `only` restricts the charges considered;
    `requested` adds on-request charges the caller asked for."""
    requested_ids = set(requested or [])
    candidates: list[ChargeSpec] = []
    screened: list[ScreenedCharge] = []
    for entry in entries:
        if only is not None and entry.charge_id not in only:
            continue
        spec = ChargeSpec.from_catalogue(entry)
        if entry.payer != "vessel":
            screened.append(_screened(entry, "excluded", _PAYER_REASONS[entry.payer]))
        elif entry.trigger == "licence_or_permit":
            screened.append(
                _screened(entry, "excluded", "A licence or permit, not a charge on a vessel call.")
            )
        elif entry.trigger == "on_request" and entry.charge_id not in requested_ids:
            screened.append(
                _screened(
                    entry,
                    "on_request",
                    "Charged only when requested or after an incident; not included.",
                )
            )
        else:
            candidates.append(spec)
    return candidates, screened


def _screened(entry: ChargeCatalogueEntry, group, reason: str) -> ScreenedCharge:
    return ScreenedCharge(entry.charge_id, entry.name, list(entry.section_refs), group, reason)
