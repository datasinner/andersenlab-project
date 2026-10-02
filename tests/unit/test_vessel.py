import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.domain.vessel import Basis, VesselCall, resolve_quantities

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _sudestada() -> dict:
    return json.loads((FIXTURES / "vessels" / "sudestada.json").read_text())


def test_sectioned_profile_maps_onto_vessel_call():
    call = VesselCall.from_profile(_sudestada(), port="Durban")

    assert call.vessel.name == "SUDESTADA"
    assert call.vessel.vessel_type == "Bulk Carrier"
    assert call.vessel.gross_tonnage == Decimal("51300")
    assert call.vessel.deadweight == Decimal("93274")
    assert call.vessel.draft_m == Decimal("14.9")
    assert call.call.port == "Durban"
    assert call.call.arrival == datetime(2024, 11, 15, 10, 12)
    assert call.call.cargo_tonnes == Decimal("40000")
    assert call.call.activity == "Exporting Iron Ore"


def test_float_inputs_become_exact_decimals():
    call = VesselCall.from_profile(_sudestada())
    assert call.vessel.loa_m == Decimal("229.2")
    assert call.call.days_alongside == Decimal("3.39")


def test_unmapped_particulars_are_kept_without_nulls():
    call = VesselCall.from_profile(_sudestada())
    assert call.vessel.additional["suez_nt"] == 49069
    assert call.vessel.additional["classification_society"] == "Registro Italiano Navale"
    assert "call_sign" not in call.vessel.additional
    assert call.vessel.additional["lbp_meters"] == 222.0


def test_canonical_shape_is_accepted():
    call = VesselCall.from_profile(
        {"vessel": {"gross_tonnage": "1200"}, "call": {"days_alongside": "1.5"}},
        port="Exampleville",
    )
    assert call.vessel.gross_tonnage == Decimal("1200")
    assert call.call.port == "Exampleville"


def test_draft_list_without_a_positive_value_becomes_none():
    profile = _sudestada()
    profile["technical_specs"]["draft_sw_s_w_t"] = [0.0, 0.0]
    assert VesselCall.from_profile(profile).vessel.draft_m is None

    profile["technical_specs"]["draft_sw_s_w_t"] = 11.2
    assert VesselCall.from_profile(profile).vessel.draft_m == Decimal("11.2")


def test_gross_tonnage_is_required_and_positive():
    with pytest.raises(ValidationError):
        VesselCall.from_profile({"vessel": {"name": "No tonnage"}})
    with pytest.raises(ValidationError):
        VesselCall.from_profile({"vessel": {"gross_tonnage": 0}})


def test_departure_before_arrival_is_rejected():
    with pytest.raises(ValidationError, match="departure must not be before arrival"):
        VesselCall.from_profile(
            {
                "vessel": {"gross_tonnage": 1000},
                "call": {"arrival": "2024-11-22T00:00:00", "departure": "2024-11-15T00:00:00"},
            }
        )


def test_time_in_port_prefers_days_alongside_and_explains_why():
    quantities = resolve_quantities(VesselCall.from_profile(_sudestada()))

    assert quantities.values[Basis.TIME_IN_PORT_DAYS] == Decimal("3.39")
    assert quantities.values[Basis.TIME_IN_PORT_HOURS] == Decimal("81.36")
    assert round(quantities.values[Basis.CALL_WINDOW_DAYS], 4) == Decimal("7.1167")
    assert "3.39 days alongside" in quantities.assumptions[Basis.TIME_IN_PORT_DAYS]
    assert "7.12 days" in quantities.assumptions[Basis.TIME_IN_PORT_DAYS]


def test_time_in_port_falls_back_to_the_call_window():
    profile = _sudestada()
    del profile["operational_data"]["days_alongside"]
    quantities = resolve_quantities(VesselCall.from_profile(profile))

    assert round(quantities.values[Basis.TIME_IN_PORT_DAYS], 4) == Decimal("7.1167")
    assert "days alongside not given" in quantities.assumptions[Basis.TIME_IN_PORT_HOURS]


def test_time_in_port_without_window_has_no_window_note():
    call = VesselCall.from_profile(
        {"vessel": {"gross_tonnage": 1000}, "call": {"days_alongside": "2"}}
    )
    quantities = resolve_quantities(call)
    assert quantities.values[Basis.TIME_IN_PORT_HOURS] == Decimal("48")
    assert "window" not in quantities.assumptions[Basis.TIME_IN_PORT_DAYS]


def test_unknown_time_in_port_is_left_out():
    quantities = resolve_quantities(VesselCall.from_profile({"vessel": {"gross_tonnage": 1000}}))
    assert Basis.TIME_IN_PORT_DAYS not in quantities.values
    assert Basis.CALL_WINDOW_DAYS not in quantities.values


def test_num_services_defaults_to_two_with_an_assumption():
    quantities = resolve_quantities(VesselCall.from_profile(_sudestada()))
    assert quantities.values[Basis.NUM_SERVICES] == Decimal(2)
    assert "entering and leaving" in quantities.assumptions[Basis.NUM_SERVICES]


def test_explicit_num_services_and_passengers_are_used_as_given():
    call = VesselCall.from_profile(
        {"vessel": {"gross_tonnage": 1000, "passengers": 300}, "call": {"num_services": 3}}
    )
    quantities = resolve_quantities(call)
    assert quantities.values[Basis.NUM_SERVICES] == Decimal(3)
    assert quantities.values[Basis.PASSENGERS] == Decimal(300)
    assert Basis.NUM_SERVICES not in quantities.assumptions
