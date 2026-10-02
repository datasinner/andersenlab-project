from app.agent.nodes.screen import screen_charges
from app.models import ChargeCatalogueEntry


def _entry(charge_id: str, payer: str, trigger: str) -> ChargeCatalogueEntry:
    return ChargeCatalogueEntry(
        charge_id=charge_id,
        name=charge_id.replace("_", " "),
        section_refs=["9.9"],
        payer=payer,
        trigger=trigger,
        description="A charge.",
    )


CATALOGUE = [
    _entry("harbour_dues", "vessel", "per_call"),
    _entry("pilotage", "vessel", "per_service"),
    _entry("lay_up_dues", "vessel", "per_period"),
    _entry("fresh_water", "vessel", "on_request"),
    _entry("cargo_dues", "cargo", "per_service"),
    _entry("crane_licence", "other", "licence_or_permit"),
    _entry("agent_registration", "vessel", "licence_or_permit"),
]


def test_routine_vessel_charges_are_candidates_and_the_rest_are_explained():
    candidates, screened = screen_charges(CATALOGUE)

    assert [c.charge_id for c in candidates] == ["harbour_dues", "pilotage", "lay_up_dues"]
    reasons = {charge.charge_id: (charge.group, charge.reason) for charge in screened}
    assert reasons == {
        "fresh_water": (
            "on_request",
            "Charged only when requested or after an incident; not included.",
        ),
        "cargo_dues": ("excluded", "Payable by the cargo owner, not the vessel."),
        "crane_licence": (
            "excluded",
            "Not a charge on the vessel (a fee for other parties or services).",
        ),
        "agent_registration": (
            "excluded",
            "A licence or permit, not a charge on a vessel call.",
        ),
    }


def test_requested_on_request_charges_are_priced():
    candidates, screened = screen_charges(CATALOGUE, requested=["fresh_water"])
    assert "fresh_water" in [c.charge_id for c in candidates]
    assert "fresh_water" not in [c.charge_id for c in screened]


def test_only_limits_the_charges_considered():
    candidates, screened = screen_charges(CATALOGUE, only=["pilotage", "cargo_dues"])
    assert [c.charge_id for c in candidates] == ["pilotage"]
    assert [c.charge_id for c in screened] == ["cargo_dues"]
