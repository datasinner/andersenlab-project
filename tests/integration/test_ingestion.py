"""Ingestion of the real TNPA tariff book (BUILD_PLAN §14), with the fake
embedder and a scripted document profile: no network, but the real PDF,
parser, structure, chunker and database."""

from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.db import async_session_factory
from app.ingestion.chunker import build_chunks
from app.ingestion.cleaner import clean
from app.ingestion.parser import PyMuPdfParser
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.structure import build_sections
from app.llm.client import FakeLLMClient
from app.llm.embeddings import FakeEmbedder
from app.llm.resilience import LLMUnavailableError
from app.models import (
    Chunk,
    ChunkKind,
    DocumentSection,
    DocumentStatus,
    SectionKind,
    TariffDocument,
)

PDF = Path(__file__).resolve().parents[2] / "data" / "tariffs" / "tnpa_tariff_book_2024_25.pdf"
PORT_COLUMNS = ["Richards", "Durban", "East", "London", "Elizabeth", "Mossel", "Cape", "Saldanha"]

PROFILE = {
    "title": "Tariff Book April 2024 - March 2025",
    "authority": "Transnet National Ports Authority",
    "currency": "zar",
    "vat_percent": "15",
    "effective_from": "2024-04-01",
    "effective_to": "2025-03-31",
    "ports": [{"name": "Durban", "aliases": ["Port of Durban"]}],
}


@pytest.fixture(scope="module")
def tnpa():
    parsed = clean(PyMuPdfParser().parse(PDF.read_bytes()))
    sections = build_sections(parsed)
    return parsed, sections, build_chunks(sections)


def _pipeline(*profile_responses) -> tuple[IngestionPipeline, FakeLLMClient]:
    llm = FakeLLMClient()
    llm.script("document_profile", *(profile_responses or (PROFILE,)))
    return IngestionPipeline(async_session_factory, llm, FakeEmbedder()), llm


# -- structure of the real document ---------------------------------------------


def test_outline_has_the_numbered_sections(tnpa):
    _, sections, _ = tnpa
    by_ref = {section.ref: section for section in sections}
    assert by_ref["3.6"].title == "3.6 TUGS/VESSEL ASSISTANCE AND/OR ATTENDANCE"
    assert by_ref["3.6"].parent.ref == "3"
    assert by_ref["4.1.1"].path.endswith("4.1 PORT FEES ON VESSELS > 4.1.1 PORT DUES")
    assert {"1.1.1", "2.1.1", "3.3", "3.8", "3.9", "4.1.2", "7.2", "8.6"} <= by_ref.keys()
    # The "2.1"-"2.5" items printed inside 5.2 don't collide with section 2.1.
    assert by_ref["2.1"].title == "2.1 VTS CHARGES ON VESSELS"
    assert by_ref["5.2.2.1"].parent.ref == "5.2"
    assert len(by_ref) == len(sections)


def test_towage_fee_table_is_one_chunk_with_every_port_column(tnpa):
    _, sections, chunks = tnpa
    towage = next(section for section in sections if section.ref == "3.6")
    tables = [
        chunk
        for chunk in chunks
        if chunk.section_ordinal == towage.ordinal and chunk.kind == ChunkKind.TABLE
    ]
    fee_table = next(chunk for chunk in tables if "73 118.07" in chunk.content)
    header = next(row for row in fee_table.content.splitlines() if row.startswith("|"))
    assert all(port in header for port in PORT_COLUMNS)
    assert "Per service based on vessel’s tonnage:" in fee_table.content
    assert "50 001 to 100 000" in fee_table.content and "93 548.13" in fee_table.content
    assert fee_table.printed_page == "15"


def test_running_headers_are_stripped_and_kept_for_the_profile(tnpa):
    parsed, _, chunks = tnpa
    assert "Tariffs subject to VAT at 15%: Tariffs in South African Rand" in parsed.running_text
    assert not any("Tariffs subject to VAT" in chunk.content for chunk in chunks)
    # The footer is gone from every page; only the cover still shows the title.
    assert [chunk.page for chunk in chunks if "Tariff Book April 2024" in chunk.content] == [1]


def test_contents_are_not_chunked_and_definitions_are_per_term(tnpa):
    _, sections, chunks = tnpa
    kinds = {section.kind for section in sections}
    assert SectionKind.CONTENTS in kinds
    contents = {s.ordinal for s in sections if s.kind == SectionKind.CONTENTS}
    assert not any(chunk.section_ordinal in contents for chunk in chunks)
    definitions = [chunk for chunk in chunks if chunk.kind == ChunkKind.DEFINITION]
    assert any("“Act” means the National Ports Act" in chunk.content for chunk in definitions)
    assert all(
        chunk.content.startswith(definitions[0].content.split("\n\n")[0]) for chunk in definitions
    )


# -- the pipeline ---------------------------------------------------------------------


async def test_pipeline_ingests_the_pdf_and_reads_its_profile(db_session):
    pipeline, llm = _pipeline()
    result = await pipeline.ingest(PDF.read_bytes(), PDF.name)

    assert result.status == DocumentStatus.READY and result.created
    assert result.sections > 90 and result.chunks > 100 and result.tables >= 10

    document = await db_session.get(TariffDocument, result.document_id)
    assert document.status == DocumentStatus.READY
    assert (document.currency, str(document.vat_rate), document.page_count) == ("ZAR", "0.1500", 27)
    assert document.ports == [{"name": "Durban", "aliases": ["Port of Durban"]}]

    chunk_count = await db_session.scalar(select(func.count()).select_from(Chunk))
    embedded = await db_session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.embedding.is_not(None))
    )
    assert chunk_count == embedded == result.chunks
    towage = await db_session.scalar(select(DocumentSection).where(DocumentSection.ref == "3.6"))
    assert towage.page_start == 8

    # The profile prompt saw the running text and the port table headers.
    prompt = llm.calls[0].messages[1].content
    assert "Tariffs subject to VAT at 15%" in prompt
    assert "Durban" in prompt and "Saldanha" in prompt


async def test_reingesting_the_same_bytes_is_a_no_op(db_session):
    pipeline, llm = _pipeline()
    first = await pipeline.ingest(PDF.read_bytes(), PDF.name)
    second = await pipeline.ingest(PDF.read_bytes(), "renamed.pdf")

    assert second.document_id == first.document_id
    assert not second.created and second.status == DocumentStatus.READY
    assert len(llm.calls) == 1
    assert await db_session.scalar(select(func.count()).select_from(TariffDocument)) == 1
    assert await db_session.scalar(select(func.count()).select_from(Chunk)) == first.chunks


async def test_failed_ingestion_is_recorded_and_retried(db_session):
    pipeline, _ = _pipeline(LLMUnavailableError("provider down"), PROFILE)

    failed = await pipeline.ingest(PDF.read_bytes(), PDF.name)
    assert failed.status == DocumentStatus.FAILED
    assert failed.error == "LLMUnavailableError: provider down"
    document = await db_session.get(TariffDocument, failed.document_id)
    assert (document.status, document.error) == (DocumentStatus.FAILED, failed.error)

    retried = await pipeline.ingest(PDF.read_bytes(), PDF.name)
    assert retried.document_id == failed.document_id
    assert retried.status == DocumentStatus.READY and retried.created
    db_session.expire_all()
    assert await db_session.scalar(select(func.count()).select_from(Chunk)) == retried.chunks
    assert await db_session.scalar(select(func.count()).select_from(TariffDocument)) == 1
