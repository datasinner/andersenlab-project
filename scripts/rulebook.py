"""Export a compiled rulebook to a file, or load rulebook files into the database.

    uv run python scripts/rulebook.py export --port Durban      # → data/rulebooks/<pdf name>.json
    uv run python scripts/rulebook.py import                    # every file in data/rulebooks
    uv run python scripts/rulebook.py import --file path/to/rulebook.json

A rulebook file holds what a document's rules were compiled against (its
profile and charge catalogue) and the rules themselves, so a fresh database
with the same PDF ingested is ready to price calls without recompiling (see
app/services/rulebook_files.py). Import is idempotent and needs no API key;
the container entrypoint runs it for RULEBOOKS_DIR after ingesting.

Exit code 0 when every file was loaded or skipped for a reason (stale,
document not ingested, already compiled), 1 if a file could not be read,
2 on a setup error.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.config import settings
from app.db import async_session_factory, engine
from app.errors import AppError
from app.logging_conf import configure_logging
from app.services.documents import resolve_document_for_port
from app.services.rulebook_files import export_rulebook, import_rulebook


async def export(port: str, out: Path | None) -> int:
    try:
        async with async_session_factory() as session:
            document, _ = await resolve_document_for_port(session, port)
            data = await export_rulebook(session, document)
    except AppError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 2
    rules = sum(len(entries) for entries in data["ports"].values())
    if not rules:
        print(f"error: no compiled rules for {document.source_filename}", file=sys.stderr)
        return 2
    path = out or Path(settings.rulebooks_dir) / f"{Path(document.source_filename).stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    ports = ", ".join(f"{port} ({len(entries)})" for port, entries in data["ports"].items())
    print(f"wrote {path}: {rules} rule(s) for {ports}")
    return 0


async def import_files(paths: list[Path]) -> int:
    unreadable = 0
    for path in paths:
        try:
            data = json.loads(path.read_text())
            result = await import_rulebook(async_session_factory, data)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"FAILED  {path.name}: {type(exc).__name__}: {exc}")
            unreadable += 1
            continue
        status = "loaded " if result.imported else "skipped"
        print(f"{status} {path.name}: {result.message}")
        if result.left_out:
            print(f"        left out (will compile on demand): {', '.join(result.left_out)}")
    return 1 if unreadable else 0


async def main(args: argparse.Namespace) -> int:
    configure_logging()
    try:
        if args.command == "export":
            return await export(args.port, args.out)
        if args.file:
            paths = [args.file]
        else:
            paths = sorted(Path(args.dir or settings.rulebooks_dir).glob("*.json"))
        if not paths:
            print("no rulebook files found")
            return 0
        return await import_files(paths)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export", help="write a document's compiled rules")
    export_parser.add_argument("--port", required=True, help="a port the document covers")
    export_parser.add_argument("--out", type=Path, help="file to write (default: RULEBOOKS_DIR)")
    import_parser = commands.add_parser("import", help="load rulebook files")
    import_parser.add_argument("--dir", help="directory of rulebook files (default: RULEBOOKS_DIR)")
    import_parser.add_argument("--file", type=Path, help="one rulebook file")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
