"""Price a vessel call against a directory of ChargeRule JSON files.

No database, no LLM: this runs the deterministic engine only. Later phases
produce the rule files with the agent; for now they can be hand-written.

    uv run python scripts/calculate.py \\
        --rules tests/fixtures/rules/durban \\
        --vessel tests/fixtures/vessels/sudestada.json \\
        --port Durban --facts '{"is_cargo_working": true}' --explain
"""

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from app.domain.numbers import format_money
from app.domain.vessel import VesselCall, resolve_quantities
from app.rules.dsl import ChargeRule
from app.rules.engine import LineItem, LineItemStatus, RuleEvaluationError, evaluate_rule


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        vessel_data = json.loads(Path(args.vessel).read_text(), parse_float=Decimal)
        vessel_call = VesselCall.from_profile(vessel_data, port=args.port)
        facts = json.loads(args.facts) if args.facts else {}
        rules = _load_rules(Path(args.rules))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    quantities = resolve_quantities(vessel_call)
    items: list[LineItem] = []
    failures: list[str] = []
    for rule in rules:
        try:
            items.append(evaluate_rule(rule, quantities, facts))
        except RuleEvaluationError as exc:
            failures.append(f"{rule.charge_id}: {exc}")

    if args.json:
        payload = {
            "line_items": [item.model_dump(mode="json") for item in items],
            "failures": failures,
            "total": str(_total(items)),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        _print_report(vessel_call, items, failures, explain=args.explain)
    return 1 if failures else 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rules", required=True, help="directory of ChargeRule *.json files")
    parser.add_argument("--vessel", required=True, help="vessel profile JSON file")
    parser.add_argument("--port", help="port name, if not in the profile")
    parser.add_argument(
        "--facts", help="resolved facts as JSON, e.g. '{\"is_cargo_working\": true}'"
    )
    parser.add_argument("--explain", action="store_true", help="show formulas and assumptions")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser.parse_args(argv)


def _load_rules(directory: Path) -> list[ChargeRule]:
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise OSError(f"no *.json rule files in {directory}")
    return [ChargeRule.model_validate_json(path.read_text()) for path in paths]


def _total(items: list[LineItem]) -> Decimal:
    return sum((item.amount or Decimal(0) for item in items), Decimal(0))


def _print_report(
    vessel_call: VesselCall, items: list[LineItem], failures: list[str], *, explain: bool
) -> None:
    vessel = vessel_call.vessel.name or "Vessel"
    port = vessel_call.call.port or "unspecified port"
    print(f"{vessel} at {port}\n")
    print(f"{'Charge':<45} {'Status':<24} {'Amount':>14}")
    for item in items:
        amount = f"{format_money(item.amount)} {item.currency}" if item.amount is not None else "-"
        print(f"{item.name:<45} {item.status.value:<24} {amount:>14}")
        if explain:
            for line in item.formula:
                print(f"    {line}")
            if item.reason:
                print(f"    {item.reason}")
    charged = [item for item in items if item.status == LineItemStatus.CHARGED]
    currencies = {item.currency for item in charged}
    currency = currencies.pop() if len(currencies) == 1 else ""
    print(f"{'Total':<70} {format_money(_total(items)):>10} {currency}")

    if explain:
        assumptions: list[str] = []
        for item in items:
            for assumption in item.assumptions:
                if assumption not in assumptions:
                    assumptions.append(assumption)
        if assumptions:
            print("\nAssumptions:")
            for assumption in assumptions:
                print(f"  - {assumption}")
    if failures:
        print("\nCould not evaluate:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
