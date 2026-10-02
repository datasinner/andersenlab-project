"""Retrieval eval: does hybrid search put the right section in the top k?

Runs against the ingested document named in eval/retrieval_cases.json, with
the configured embedder (real OpenAI embeddings by default).

    uv run python eval/run_retrieval_eval.py     # or: make eval-retrieval

A case passes when any of the top-k chunks belongs to an expected section
(by ref, or by title for sections without a printed number). Exits non-zero
when recall@k is below the threshold in the cases file.
"""

import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

from app.db import async_session_factory, engine
from app.llm.embeddings import build_embedder
from app.llm.resilience import ResiliencePolicy
from app.models import DocumentStatus, TariffDocument
from app.retrieval.search import TariffSearch

CASES = Path(__file__).with_name("retrieval_cases.json")


async def main() -> int:
    spec = json.loads(CASES.read_text())
    k = spec["k"]
    async with async_session_factory() as session:
        document_id = await session.scalar(
            select(TariffDocument.id).where(
                TariffDocument.source_filename == spec["document"],
                TariffDocument.status == DocumentStatus.READY,
            )
        )
    if document_id is None:
        print(
            f"error: {spec['document']} is not ingested; run `make ingest` first", file=sys.stderr
        )
        return 2

    search = TariffSearch(async_session_factory, build_embedder(ResiliencePolicy.from_settings()))
    passed = 0
    try:
        for case in spec["cases"]:
            hits = await search.search(document_id, case["query"], k=k)
            expected_refs = set(case.get("expected", []))
            expected_titles = set(case.get("expected_titles", []))
            positions = [
                index
                for index, hit in enumerate(hits, start=1)
                if hit.section_ref in expected_refs or hit.section_title in expected_titles
            ]
            ok = bool(positions)
            passed += ok
            found = f"rank {positions[0]}" if ok else "missed"
            top = ", ".join(hit.section_ref for hit in hits)
            print(f"{'PASS' if ok else 'FAIL'}  {found:8}  {case['query']}  [{top}]")
    finally:
        await engine.dispose()

    recall = passed / len(spec["cases"])
    print(
        f"\nrecall@{k}: {recall:.2f} ({passed}/{len(spec['cases'])}), threshold {spec['threshold']}"
    )
    return 0 if recall >= spec["threshold"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
