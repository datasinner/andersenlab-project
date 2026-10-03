"""End-to-end accuracy eval: price the reference vessel calls and compare.

Runs the real system in-process (database, OpenAI) on every case in
eval/cases. For each case it makes sure the tariff document is ingested and
the port's rulebook compiled (with the compile model, outside any calculation
timeout; cached rules are reused unless --refresh), then prices the call from
its vessel profile and, if it has one, from its plain-language query. Both
must match the expected values within the case tolerance, and each other.

    uv run python eval/run_eval.py                  # or: make eval
    uv run python eval/run_eval.py --refresh        # recompile every rule first (cold)
    uv run python eval/run_eval.py --case exampleville

Expected values are matched to line items by tariff section ref, or by
section title for documents without numbered sections (their refs are
generated). Writes eval/report.json.
"""

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy import select

from app.agent.graph import CalculationInput
from app.config import settings
from app.db import async_session_factory, engine
from app.domain.numbers import format_money
from app.errors import AppError
from app.ingestion.pipeline import IngestionPipeline
from app.llm.cache import with_response_cache
from app.llm.client import build_llm_clients
from app.llm.embeddings import build_embedder
from app.llm.resilience import LLMConfigurationError, ResiliencePolicy
from app.logging_conf import configure_logging
from app.models import DocumentSection
from app.schemas import CalculationOut
from app.services.calculations import CalculationService
from app.services.documents import resolve_document_for_port
from app.services.rulebook import RulebookService

ROOT = Path(__file__).resolve().parents[1]
CASES = Path(__file__).with_name("cases")
REPORT = Path(__file__).with_name("report.json")


async def main(refresh: bool, only_json: bool, case_filter: str | None) -> int:
    configure_logging()
    policy = ResiliencePolicy.from_settings()
    try:
        llm, compile_llm = build_llm_clients(policy)
        llm = with_response_cache(llm, async_session_factory)
        compile_llm = with_response_cache(compile_llm, async_session_factory)
        embedder = build_embedder(policy)
    except LLMConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rulebook = RulebookService(async_session_factory, compile_llm, embedder)
    service = CalculationService(async_session_factory, llm, rulebook)
    ingestion = IngestionPipeline(
        async_session_factory, compile_llm, embedder, rulebooks_dir=settings.rulebooks_dir
    )

    report = []
    all_passed = True
    try:
        for path in sorted(CASES.glob("*.json")):
            if case_filter and case_filter not in path.stem:
                continue
            case = json.loads(path.read_text())
            vessel = json.loads((ROOT / case["vessel_file"]).read_text())
            print(f"\n=== {case['name']}")
            if not await _prepare(case, vessel, ingestion, rulebook, refresh):
                all_passed = False
                continue

            runs = [("profile JSON", CalculationInput(port=case["port"], vessel=vessel))]
            if case.get("query") and not only_json:
                runs.append(("plain-language query", CalculationInput(query=case["query"])))
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
                titles = await _section_titles(result.document.id)
                passed, rows = _compare(case, result, titles)
                all_passed &= passed
                amounts_by_run.append({row["label"]: row["computed"] for row in rows})
                _print_run(label, result, rows, seconds, titles, case.get("expected_total"))
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


async def _prepare(case: dict, vessel: dict, ingestion, rulebook, refresh: bool) -> bool:
    """Ingest the case's document if needed and compile the port's rulebook."""
    if case.get("document_file"):
        path = ROOT / case["document_file"]
        result = await ingestion.ingest(path.read_bytes(), path.name)
        if result.error:
            print(f"ingestion FAILED: {result.error}")
            return False
    arrival = (vessel.get("operational_data") or {}).get("arrival_time")
    on_date = datetime.fromisoformat(arrival).date() if arrival else None
    try:
        async with async_session_factory() as session:
            document, port_key = await resolve_document_for_port(
                session, case["port"], on_date=on_date
            )
    except AppError as exc:
        print(f"FAILED {exc.code}: {exc.message}")
        return False
    started = time.monotonic()
    report = await rulebook.compile_port(document, port_key, refresh=refresh)
    counts: dict[str, int] = {}
    for outcome in report.outcomes:
        counts[outcome.status] = counts.get(outcome.status, 0) + 1
    fresh = sum(not outcome.from_cache for outcome in report.outcomes)
    tokens = report.usage.prompt_tokens + report.usage.completion_tokens
    summary = ", ".join(f"{count} {status}" for status, count in sorted(counts.items()))
    print(
        f"rulebook for {port_key}: {len(report.outcomes)} rules, {fresh} compiled now "
        f"({tokens} tokens, {time.monotonic() - started:.0f} s): {summary}"
    )
    return True


async def _section_titles(document_id) -> dict[str, str]:
    async with async_session_factory() as session:
        rows = await session.execute(
            select(DocumentSection.ref, DocumentSection.title).where(
                DocumentSection.document_id == document_id
            )
        )
        return dict(rows.all())


def _matches(expected: dict, item, titles: dict[str, str]) -> bool:
    if "section" in expected:
        return expected["section"] in item.section_refs
    wanted = expected["section_title"].casefold()
    return any(titles.get(ref, "").casefold() == wanted for ref in item.section_refs)


def _compare(case: dict, result: CalculationOut, titles: dict[str, str]) -> tuple[bool, list[dict]]:
    tolerance = Decimal(case["tolerance_pct"])
    rows = []
    passed = True
    for expected in case["expected"]:
        reference = Decimal(expected["amount"])
        item = next((i for i in result.line_items if _matches(expected, i, titles)), None)
        computed = item.amount if item else None
        delta = (computed - reference) / reference * 100 if computed is not None else None
        ok = delta is not None and abs(delta) <= tolerance
        passed &= ok
        rows.append(
            {
                "label": expected["label"],
                "section": expected.get("section") or expected["section_title"],
                "expected": str(reference),
                "computed": str(computed) if computed is not None else None,
                "delta_pct": f"{delta:+.3f}" if delta is not None else None,
                "charge": item.name if item else None,
                "confidence": item.confidence if item else None,
                "pass": ok,
            }
        )
    expected_total = case.get("expected_total")
    if expected_total is not None and result.total != Decimal(expected_total):
        passed = False
    return passed, rows


def _print_run(
    label: str,
    result: CalculationOut,
    rows: list[dict],
    seconds: float,
    titles: dict[str, str],
    expected_total: str | None,
) -> None:
    print(
        f"\n{label}: {result.status}, {seconds:.1f} s, "
        f"{result.prompt_tokens + result.completion_tokens} tokens"
    )
    print(f"  {'Expected item':<20} {'Section':<20} {'Expected':>12} {'Computed':>12} {'Δ %':>8}")
    for row in rows:
        computed = format_money(Decimal(row["computed"])) if row["computed"] else "missing"
        mark = "" if row["pass"] else "  ✗"
        expected = format_money(Decimal(row["expected"]))
        print(
            f"  {row['label']:<20} {row['section'][:20]:<20} {expected:>12} "
            f"{computed:>12} {row['delta_pct'] or '':>8}{mark}"
        )
    matched = {row["charge"] for row in rows}
    for item in result.line_items:
        if item.name not in matched:
            section = titles.get(item.section_refs[0], item.section_refs[0])
            print(
                f"  {'(also charged)':<20} {section[:20]:<20} {'':>12} "
                f"{format_money(item.amount):>12}           {item.name}"
            )
    total_check = ""
    if expected_total is not None:
        ok = result.total == Decimal(expected_total)
        total_check = f" (expected {format_money(Decimal(expected_total))}{'' if ok else '  ✗'})"
    print(
        f"  Total {format_money(result.total)} {result.currency}{total_check}; not applicable: "
        f"{len(result.not_applicable)}, not priced: {len(result.not_priced)}, on request: "
        f"{len(result.on_request)}, excluded: {len(result.excluded)}, "
        f"failed: {len(result.failed)}"
    )
    for warning in result.warnings:
        print(f"  ! {warning}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--refresh", action="store_true", help="recompile every rule first")
    parser.add_argument("--json-only", action="store_true", help="skip the plain-language query")
    parser.add_argument("--case", help="run only cases whose file name contains this")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.refresh, args.json_only, args.case)))
