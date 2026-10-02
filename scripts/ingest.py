"""Ingest tariff PDFs into the database (idempotent: already-ingested files are skipped).

    uv run python scripts/ingest.py --dir data/tariffs
    uv run python scripts/ingest.py --file path/to/tariff.pdf
    uv run python scripts/ingest.py --dir data/tariffs --force   # rebuild ready documents

Exit code 0 if every file is ready, 1 if any failed, 2 on a setup error.
The container entrypoint runs this for TARIFFS_DIR before starting the API.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.db import async_session_factory, engine
from app.ingestion.pipeline import IngestionPipeline
from app.llm.client import build_llm_client
from app.llm.embeddings import build_embedder
from app.llm.resilience import LLMConfigurationError, ResiliencePolicy
from app.logging_conf import configure_logging
from app.models import DocumentStatus


async def main(paths: list[Path], force: bool) -> int:
    configure_logging()
    policy = ResiliencePolicy.from_settings()
    try:
        llm = build_llm_client(policy)
        embedder = build_embedder(policy)
    except LLMConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    pipeline = IngestionPipeline(async_session_factory, llm, embedder)
    failed = 0
    try:
        for path in paths:
            result = await pipeline.ingest(path.read_bytes(), path.name, force=force)
            if result.status == DocumentStatus.FAILED:
                failed += 1
                print(f"FAILED  {path.name}: {result.error}", file=sys.stderr)
            elif not result.created:
                print(f"skipped {path.name}: already ingested ({result.document_id})")
            else:
                print(
                    f"ready   {path.name}: {result.sections} sections, {result.chunks} chunks, "
                    f"{result.tables} tables, {result.charges} charges ({result.document_id})"
                )
    finally:
        await engine.dispose()
    return 1 if failed else 0


def _pdf_paths(args: argparse.Namespace) -> list[Path]:
    if args.file:
        return [Path(args.file)]
    directory = Path(args.dir)
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() == ".pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dir", help="ingest every PDF in this directory")
    source.add_argument("--file", help="ingest one PDF")
    parser.add_argument("--force", action="store_true", help="rebuild already-ingested documents")
    args = parser.parse_args()
    paths = _pdf_paths(args)
    if not paths:
        print("no PDF files to ingest")
        raise SystemExit(0)
    raise SystemExit(asyncio.run(main(paths, args.force)))
