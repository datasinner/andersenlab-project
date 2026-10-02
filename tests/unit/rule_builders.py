"""Builders for small ChargeRules in unit tests. Uses an invented port and
currency so nothing here resembles a real tariff."""

from decimal import Decimal
from typing import Any

from app.domain.vessel import Basis, ResolvedQuantities
from app.rules.dsl import ChargeRule

CITATION = {"chunk_id": 1, "section_ref": "9.9", "page": 1, "quote": "Example fee"}


def make_rule_data(**fields: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "charge_id": "example_charge",
        "name": "Example charge",
        "section_refs": ["9.9"],
        "port_key": "Exampleville",
        "currency": "XTS",
        "payer": "vessel",
        "status": "priced",
        "components": [{"kind": "fixed", "id": "flat", "label": "Flat fee", "amount": "100"}],
        "citations": [CITATION],
    }
    data.update(fields)
    return data


def make_rule(**fields: Any) -> ChargeRule:
    return ChargeRule.model_validate(make_rule_data(**fields))


def per_unit(
    component_id: str = "per_ton",
    rate: str = "2",
    basis: str = "gross_tonnage",
    unit_size: str = "100",
    rounding: str = "ceil",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "kind": "per_unit",
        "id": component_id,
        "label": f"{component_id} fee",
        "rate": rate,
        "units": {"basis": basis, "unit_size": unit_size, "rounding": rounding},
        **extra,
    }


def flag(name: str, default: bool | None = False) -> dict[str, Any]:
    return {
        "name": name,
        "type": "bool",
        "description": f"{name} applies",
        "default_value": default,
    }


def quantities(
    assumptions: dict[Basis, str] | None = None, **values: str | int
) -> ResolvedQuantities:
    return ResolvedQuantities(
        values={Basis(name): Decimal(str(value)) for name, value in values.items()},
        assumptions=assumptions or {},
    )
