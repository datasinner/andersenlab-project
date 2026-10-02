"""End-to-end accuracy eval: price the reference vessel calls and compare.

Runs the real system in-process (database, OpenAI, cached or freshly
compiled rules) on every case in eval/cases. Each case is priced from its
vessel profile and, if it has one, from its plain-language query; both must
match the reference values within the case tolerance, and each other.

    uv run python eval/run_eval.py              # or: make eval
    uv run python eval/run_eval.py --refresh    # recompile every rule first (cold)

Expected values are matched to line items by tariff section, which is
stable across runs (catalogue ids are not). Writes eval/report.json.
"""

import argparse
import asyncio
import json
import sys
import time
import uuid
from decimal import Decimal
from pathlib import Path

from app.agent.graph import CalculationInput
from app.db import async_session_factory, engine
from app.domain.numbers import format_money
from app.errors import AppError
from app.llm.client import build_llm_clients
from app.llm.embeddings import build_embedder
from app.llm.resilience import LLMConfigurationError, ResiliencePolicy
from app.logging_conf import configure_logging
from app.schemas import CalculationOut
from app.services.calculations import CalculationService
from app.services.rulebook import RulebookService

ROOT = Path(__file__).resolve().parents[1]
CASES = Path(__file__).with_name("cases")
REPORT = Path(__file__).with_name("report.json")


async def main(refresh: bool, only_json: bool) -> int:
    configure_logging()
    policy = ResiliencePolicy.from_settings()
    try:
        llm, compile_llm = build_llm_clients(policy)
        embedder = build_embedder(policy)
    except LLMConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    service = CalculationService(
        async_session_factory, llm, RulebookService(async_session_factory, compile_llm, embedder)
    )

    report = []
    all_passed = True
    try:
        for path in sorted(CASES.glob("*.json")):
            case = json.loads(path.read_text())
            vessel = json.loads((ROOT / case["vessel_file"]).read_text())
            runs = [
                (
                    "profile JSON",
                    CalculationInput(port=case["port"], vessel=vessel, refresh_rules=refresh),
                )
            ]
            if case.get("query") and not only_json:
                runs.append(("plain-language query", CalculationInput(query=case["query"])))

            print(f"\n=== {case['name']}")
            amounts_by_run = []
            for label, request in runs:
                started = time.monotonic()
                try:
                    result = await service.calculate(request, request_id=f"eval-{uuid.uuid4()}")
                except AppError as exc:
                    print(f"\n{label}: FAILED {exc.code}: {exc.message}")
                    all_passed = False
                    continue
                seconds = time.monotonic() - started
                passed, rows = _compare(case, result)
                all_passed &= passed
                amounts_by_run.append({row["label"]: row["computed"] for row in rows})
                _print_run(label, result, rows, seconds)
                report.append(
                    {
                        "case": case["name"],
                        "input": label,
                        "seconds": round(seconds, 1),
                        "status": result.status,
                        "rows": rows,
                        "total": str(result.total),
                    }
                )
                refresh = False  # a cold run recompiles once, not per input
            if len(amounts_by_run) == 2:
                consistent = amounts_by_run[0] == amounts_by_run[1]
                all_passed &= consistent
                agree = "yes" if consistent else "NO"
                print(f"\nProfile JSON and plain-language query agree: {agree}")
    finally:
        await engine.dispose()

    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{'PASS' if all_passed else 'FAIL'} — report written to {REPORT.relative_to(ROOT)}")
    return 0 if all_passed else 1


def _compare(case: dict, result: CalculationOut) -> tuple[bool, list[dict]]:
    tolerance = Decimal(case["tolerance_pct"])
    rows = []
    passed = True
    for expected in case["expected"]:
        reference = Decimal(expected["amount"])
        item = next((i for i in result.line_items if expected["section"] in i.section_refs), None)
        computed = item.amount if item else None
        delta = (computed - reference) / reference * 100 if computed is not None else None
        ok = delta is not None and abs(delta) <= tolerance
        passed &= ok
        rows.append(
            {
                "label": expected["label"],
                "section": expected["section"],
                "expected": str(reference),
                "computed": str(computed) if computed is not None else None,
                "delta_pct": f"{delta:+.3f}" if delta is not None else None,
                "charge": item.name if item else None,
                "confidence": item.confidence if item else None,
                "pass": ok,
            }
        )
    return passed, rows


def _print_run(label: str, result: CalculationOut, rows: list[dict], seconds: float) -> None:
    print(
        f"\n{label}: {result.status}, {seconds:.1f} s, "
        f"{result.prompt_tokens + result.completion_tokens} tokens"
    )
    print(f"  {'Reference item':<15} {'§':<6} {'Expected':>12} {'Computed':>12} {'Δ %':>8}  Charge")
    for row in rows:
        computed = format_money(Decimal(row["computed"])) if row["computed"] else "missing"
        mark = "" if row["pass"] else "  ✗"
        expected = format_money(Decimal(row["expected"]))
        print(
            f"  {row['label']:<15} {row['section']:<6} {expected:>12} "
            f"{computed:>12} {row['delta_pct'] or '':>8}  {row['charge'] or ''}{mark}"
        )
    matched = {row["charge"] for row in rows}
    others = [item for item in result.line_items if item.name not in matched]
    for item in others:
        print(
            f"  {'(also charged)':<15} {item.section_refs[0]:<6} {'':>12} "
            f"{format_money(item.amount):>12} {'':>8}  {item.name}"
        )
    print(
        f"  Total {format_money(result.total)} {result.currency}; not applicable: "
        f"{len(result.not_applicable)}, on request: {len(result.on_request)}, "
        f"excluded: {len(result.excluded)}, failed: {len(result.failed)}"
    )
    for warning in result.warnings:
        print(f"  ! {warning}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--refresh", action="store_true", help="recompile every rule first")
    parser.add_argument("--json-only", action="store_true", help="skip the plain-language query")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.refresh, args.json_only)))
