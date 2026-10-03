"""No tariff knowledge in the application code (BUILD_PLAN §3).

Fails if any port, authority or vessel name from the test tariffs, or any
decimal rate from the golden rules or the synthetic tariff, appears anywhere
under app/ (code, docstrings, comments). Rules come from documents; examples
and test data live outside app/.
"""

import json
import re
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
NAMES = [
    "Durban",
    "Richards Bay",
    "Saldanha",
    "Cape Town",
    "Ngqura",
    "Mossel Bay",
    "East London",
    "Port Elizabeth",
    "Transnet",
    "TNPA",
    "SAMSA",
    "SUDESTADA",
    "Exampleville",
    "NORDIC TERN",
    # Distinctive wording from the TNPA document.
    "Tariff Book",
    "TUGS/VESSEL",
    "subject to VAT at 15",
    "Per service based on vessel",
]
_NUMERIC_FIELDS = {"amount", "rate", "base_fee", "minimum", "maximum", "percent", "up_to", "width"}
_DECIMAL = re.compile(r"\d[\d ]*\.\d+")


def _app_sources() -> dict[Path, str]:
    return {path: path.read_text() for path in APP.rglob("*.py")}


def _decimals(text: str) -> set[Decimal]:
    return {Decimal(match.replace(" ", "")) for match in _DECIMAL.findall(text)}


def _rule_numbers(node: object) -> set[Decimal]:
    """Values of the numeric fields of a rule (not section refs or ids)."""
    found: set[Decimal] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _NUMERIC_FIELDS and isinstance(value, str):
                found |= _decimals(value)
            else:
                found |= _rule_numbers(value)
    elif isinstance(node, list):
        for item in node:
            found |= _rule_numbers(item)
    return found


def _tariff_rates() -> set[Decimal]:
    """Decimal amounts from the golden TNPA rules and the synthetic tariff."""
    rates: set[Decimal] = set()
    for path in (ROOT / "tests" / "fixtures" / "rules" / "durban").glob("*.json"):
        rates |= _rule_numbers(json.loads(path.read_text()))
    synthetic = ROOT / "scripts" / "make_synthetic_tariff.py"
    if synthetic.exists():
        rates |= _decimals(synthetic.read_text())
    # Structural values that are not rates.
    return {rate for rate in rates if rate >= 1 and rate != rate.to_integral_value()}


def test_no_port_authority_or_vessel_names_in_app():
    found = [
        f"{path.relative_to(ROOT)}: {name}"
        for path, text in _app_sources().items()
        for name in NAMES
        if name.casefold() in text.casefold()
    ]
    assert not found, found


def test_no_tariff_rates_in_app():
    rates = _tariff_rates()
    assert len(rates) > 15  # the fixtures were found
    found = [
        f"{path.relative_to(ROOT)}: {value}"
        for path, text in _app_sources().items()
        for value in sorted(_decimals(text) & rates)
    ]
    assert not found, found


def test_swagger_examples_live_outside_app():
    examples = json.loads((ROOT / "examples" / "calculation_requests.json").read_text())
    assert examples["sudestada_profile"]["value"]["port"] == "Durban"
