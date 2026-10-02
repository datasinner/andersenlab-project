"""Ingestion: PDF bytes → a ready, searchable tariff document.

register (sha256, idempotent) → parse → clean → structure → chunk → store
[parsing] → embed [indexing] → document profile + charge catalogue, two
concurrent LLM calls [cataloguing] → ready.

Each stage commits, so a document's status shows how far it got. Any
failure marks the document failed with the error; ingesting the same bytes
again retries it. A document that is already ready is left untouched unless
force=True (e.g. after a prompt change), which rebuilds it in place.
"""

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.numbers import NumberFormatError, parse_number
from app.ingestion.catalogue import CatalogueEntry, discover_charges
from app.ingestion.chunker import ChunkDraft, build_chunks
from app.ingestion.cleaner import clean
from app.ingestion.parser import ParsedDocument, PdfParser, PyMuPdfParser
from app.ingestion.profile import DocumentProfile, read_profile
from app.ingestion.structure import Section, build_sections, table_count
from app.llm.client import LLMClient
from app.llm.embeddings import Embedder
from app.models import (
    ChargeCatalogueEntry,
    Chunk,
    CompiledRule,
    DocumentSection,
    DocumentStatus,
    TariffDocument,
)

logger = structlog.get_logger("app.ingestion")

EMBEDDING_BATCH_SIZE = 64
_MAX_ERROR_CHARS = 2000


@dataclass(frozen=True)
class IngestResult:
    document_id: uuid.UUID
    status: DocumentStatus
    created: bool  # False when these exact bytes were already ingested
    sections: int = 0
    chunks: int = 0
    tables: int = 0
    charges: int = 0
    error: str | None = None


class IngestionPipeline:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        llm: LLMClient,
        embedder: Embedder,
        parser: PdfParser | None = None,
    ) -> None:
        self._sessions = session_factory
        self._llm = llm
        self._embedder = embedder
        self._parser = parser or PyMuPdfParser()

    async def ingest(self, content: bytes, filename: str, *, force: bool = False) -> IngestResult:
        checksum = hashlib.sha256(content).hexdigest()
        document_id, already_ready = await self._register(content, checksum, filename, force)
        log = logger.bind(document_id=str(document_id), filename=filename)
        if already_ready:
            log.info("ingestion_skipped", reason="already ingested")
            return IngestResult(document_id, DocumentStatus.READY, created=False)

        started = time.monotonic()
        try:
            parsed = await asyncio.to_thread(self._parse, content)
            sections = build_sections(parsed)
            chunks = build_chunks(sections)
            await self._store_structure(document_id, parsed, sections, chunks)
            log.info("ingestion_parsed", sections=len(sections), chunks=len(chunks))

            await self._embed_chunks(document_id)
            await self._set_status(document_id, DocumentStatus.CATALOGUING)

            (profile, _), (charges, _) = await asyncio.gather(
                read_profile(self._llm, parsed, sections),
                discover_charges(self._llm, sections),
            )
            await self._store_results(document_id, profile, charges)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS]
            log.exception("ingestion_failed")
            await self._set_status(document_id, DocumentStatus.FAILED, error=error)
            return IngestResult(document_id, DocumentStatus.FAILED, created=True, error=error)

        log.info("ingestion_ready", duration_ms=int((time.monotonic() - started) * 1000))
        return IngestResult(
            document_id,
            DocumentStatus.READY,
            created=True,
            sections=len(sections),
            chunks=len(chunks),
            tables=table_count(sections),
            charges=len(charges),
        )

    def _parse(self, content: bytes) -> ParsedDocument:
        # CPU-bound; runs in a worker thread so the event loop stays free.
        return clean(self._parser.parse(content))

    async def _register(
        self, content: bytes, checksum: str, filename: str, force: bool
    ) -> tuple[uuid.UUID, bool]:
        async with self._sessions() as session:
            document = await session.scalar(
                select(TariffDocument).where(TariffDocument.checksum == checksum)
            )
            if document is not None and document.status == DocumentStatus.READY and not force:
                return document.id, True
            if document is None:
                document = TariffDocument(
                    source_filename=filename, content=content, checksum=checksum
                )
                session.add(document)
                await session.flush()
            else:
                # A previous attempt failed or was interrupted, or a rebuild
                # was forced: start over.
                await _delete_derived_rows(session, document.id)
            document.status = DocumentStatus.PARSING
            document.error = None
            await session.commit()
            return document.id, False

    async def _store_structure(
        self,
        document_id: uuid.UUID,
        parsed: ParsedDocument,
        sections: list[Section],
        chunks: list[ChunkDraft],
    ) -> None:
        async with self._sessions() as session:
            section_ids: dict[int, int] = {}
            for section in sections:
                row = DocumentSection(
                    document_id=document_id,
                    parent_id=section_ids[section.parent.ordinal] if section.parent else None,
                    ref=section.ref,
                    title=section.title,
                    path=section.path,
                    level=section.level,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    ordinal=section.ordinal,
                    kind=section.kind,
                )
                session.add(row)
                await session.flush()
                section_ids[section.ordinal] = row.id
            session.add_all(
                Chunk(
                    document_id=document_id,
                    section_id=section_ids[draft.section_ordinal],
                    kind=draft.kind,
                    content=draft.content,
                    page=draft.page,
                    printed_page=draft.printed_page,
                    ordinal=ordinal,
                    token_count=draft.token_count,
                )
                for ordinal, draft in enumerate(chunks, start=1)
            )
            document = await session.get(TariffDocument, document_id)
            assert document is not None
            document.page_count = parsed.page_count
            document.status = DocumentStatus.INDEXING
            await session.commit()

    async def _embed_chunks(self, document_id: uuid.UUID) -> None:
        async with self._sessions() as session:
            chunks = list(
                await session.scalars(
                    select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.ordinal)
                )
            )
            for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
                batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
                vectors = await self._embedder.embed(
                    [chunk.content for chunk in batch], name="embed_chunks"
                )
                for chunk, vector in zip(batch, vectors, strict=True):
                    chunk.embedding = vector
            await session.commit()

    async def _store_results(
        self,
        document_id: uuid.UUID,
        profile: DocumentProfile,
        charges: list[CatalogueEntry],
    ) -> None:
        async with self._sessions() as session:
            session.add_all(
                ChargeCatalogueEntry(
                    document_id=document_id,
                    charge_id=charge.charge_id,
                    name=charge.name,
                    section_refs=charge.section_refs,
                    payer=charge.payer,
                    trigger=charge.trigger,
                    description=charge.description,
                )
                for charge in charges
            )
            document = await session.get(TariffDocument, document_id)
            assert document is not None
            document.title = profile.title
            document.authority = profile.authority
            document.currency = (profile.currency or "").upper()[:3] or None
            document.vat_rate = _vat_rate(profile.vat_percent)
            document.effective_from = profile.effective_from
            document.effective_to = profile.effective_to
            document.ports = [port.model_dump() for port in profile.ports]
            document.status = DocumentStatus.READY
            await session.commit()

    async def _set_status(
        self, document_id: uuid.UUID, status: DocumentStatus, error: str | None = None
    ) -> None:
        async with self._sessions() as session:
            document = await session.get(TariffDocument, document_id)
            assert document is not None
            document.status = status
            document.error = error
            await session.commit()


async def _delete_derived_rows(session: AsyncSession, document_id: uuid.UUID) -> None:
    # Compiled rules cite chunk ids, which a rebuild replaces.
    await session.execute(delete(CompiledRule).where(CompiledRule.document_id == document_id))
    await session.execute(
        delete(ChargeCatalogueEntry).where(ChargeCatalogueEntry.document_id == document_id)
    )
    await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
    await session.execute(delete(DocumentSection).where(DocumentSection.document_id == document_id))


def _vat_rate(vat_percent: str | None) -> Decimal | None:
    if not vat_percent:
        return None
    try:
        return parse_number(vat_percent) / 100
    except NumberFormatError:
        return None
