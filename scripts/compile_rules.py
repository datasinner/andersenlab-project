"""Compile the rulebook for a port: research, extract, validate and review each charge.

    uv run python scripts/compile_rules.py --port Durban
    uv run python scripts/compile_rules.py --port Durban --charge port_dues --refresh
    uv run python scripts/compile_rules.py --port Durban --out-dir build/rulebook/durban

By default compiles the charges a vessel pays on an ordinary call (payer
vessel; per call, per service or per period); --all compiles every charge in
the catalogue. Rules are cached; --refresh recompiles. --out-dir writes one
ChargeRule JSON per charge, which scripts/calculate.py can price directly.
"""

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

from app.db import async_session_factory, engine
from app.errors import AppError
from app.llm.client import build_llm_clients
from app.llm.embeddings import build_embedder
from app.llm.resilience import LLMConfigurationError, ResiliencePolicy
from app.logging_conf import configure_logging
from app.services.documents import resolve_document_for_port
from app.services.rulebook import RulebookService


async def main(args: argparse.Namespace) -> int:
    configure_logging()
    policy = ResiliencePolicy.from_settings()
    try:
        _, llm = build_llm_clients(policy)  # offline work uses the compile model
        embedder = build_embedder(policy)
    except LLMConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rulebook = RulebookService(async_session_factory, llm, embedder)
    try:
        async with async_session_factory() as session:
            document, port_key = await resolve_document_for_port(
                session, args.port, document_id=args.document_id
            )
        report = await rulebook.compile_port(
            document,
            port_key,
            charge_ids=args.charge or None,
            include_all=args.all,
            refresh=args.refresh,
        )
    except AppError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 2
    finally:
        await engine.dispose()

    print(f"\n{document.title} — {port_key} — prompts {report.prompt_version}\n")
    print(f"{'Charge':<44} {'Status':<15} {'Rev':>3} {'Tokens':>8} {'Secs':>5}  Sections read")
    for outcome in report.outcomes:
        status = outcome.status + (" (cached)" if outcome.from_cache else "")
        tokens = outcome.usage.prompt_tokens + outcome.usage.completion_tokens
        print(
            f"{outcome.charge_id:<44} {status:<15} {outcome.revisions:>3} {tokens:>8} "
            f"{outcome.latency_ms / 1000:>5.0f}  {', '.join(outcome.sections_read)}"
        )
        for issue in outcome.issues:
            print(f"{'':<46}! {issue}")
        for note in outcome.review_notes:
            print(f"{'':<46}- {note}"[:400])
        if args.verbose:
            for step in outcome.steps:
                detail = step.tool or ""
                if step.input_summary:
                    detail += f" {step.input_summary}"
                if step.output:
                    detail += f" -> {step.output}"
                print(f"{'':<6}[{step.node}] {detail.strip()}"[:400])
        if outcome.error:
            print(f"{'':<46}! {outcome.error}")
    usage = report.usage
    print(f"\nTotal tokens: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for outcome in report.outcomes:
            if outcome.rule is not None:
                path = out_dir / f"{outcome.charge_id}.json"
                path.write_text(
                    outcome.rule.model_dump_json(indent=2, exclude_defaults=True) + "\n"
                )
                written += 1
        print(f"Wrote {written} rule file(s) to {out_dir}")
    return 1 if any(outcome.status == "failed" for outcome in report.outcomes) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", required=True)
    parser.add_argument("--document-id", type=uuid.UUID)
    parser.add_argument("--charge", action="append", help="compile only this charge (repeatable)")
    parser.add_argument("--all", action="store_true", help="compile every charge in the catalogue")
    parser.add_argument("--refresh", action="store_true", help="recompile cached rules")
    parser.add_argument("--out-dir", help="write each compiled rule as JSON here")
    parser.add_argument("--verbose", action="store_true", help="print every agent step")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
