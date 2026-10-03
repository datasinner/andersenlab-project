"""Read access to ingested tariff documents."""

import uuid
from dataclasses import dataclass
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.ports import match_port
from app.errors import AppError
from app.models import (
    ChargeCatalogueEntry,
    Chunk,
    DocumentSection,
    DocumentStatus,
    TariffDocument,
)
from app.retrieval.context import SectionText, read_section


class DocumentNotFoundError(AppError):
    status_code = 404
    code = "DOCUMENT_NOT_FOUND"


class SectionNotFoundError(AppError):
    status_code = 404
    code = "SECTION_NOT_FOUND"


class DocumentNotReadyError(AppError):
    status_code = 409
    code = "DOCUMENT_NOT_READY"


class PortNotCoveredError(AppError):
    status_code = 422
    code = "PORT_NOT_COVERED"


class UploadTooLargeError(AppError):
    status_code = 413
    code = "UPLOAD_TOO_LARGE"


class NotAPdfError(AppError):
    status_code = 415
    code = "NOT_A_PDF"


@dataclass(frozen=True)
class DocumentCounts:
    sections: int
    chunks: int
    charges: int


async def list_documents(session: AsyncSession) -> list[TariffDocument]:
    result = await session.scalars(
        select(TariffDocument).order_by(TariffDocument.created_at.desc())
    )
    return list(result)


async def get_document(session: AsyncSession, document_id: uuid.UUID) -> TariffDocument:
    document = await session.get(TariffDocument, document_id)
    if document is None:
        raise DocumentNotFoundError(f"No tariff document with id {document_id}")
    return document


async def count_contents(session: AsyncSession, document_id: uuid.UUID) -> DocumentCounts:
    async def count(model: type) -> int:
        statement = select(func.count()).select_from(model).where(model.document_id == document_id)
        return await session.scalar(statement) or 0

    return DocumentCounts(
        sections=await count(DocumentSection),
        chunks=await count(Chunk),
        charges=await count(ChargeCatalogueEntry),
    )


async def list_charges(session: AsyncSession, document_id: uuid.UUID) -> list[ChargeCatalogueEntry]:
    await get_document(session, document_id)
    result = await session.scalars(
        select(ChargeCatalogueEntry)
        .where(ChargeCatalogueEntry.document_id == document_id)
        .order_by(ChargeCatalogueEntry.id)
    )
    return list(result)


async def get_section(session: AsyncSession, document_id: uuid.UUID, ref: str) -> SectionText:
    await get_document(session, document_id)
    section = await read_section(session, document_id, ref)
    if section is None:
        raise SectionNotFoundError(f"Document {document_id} has no section '{ref}'")
    return section


async def resolve_document_for_port(
    session: AsyncSession,
    port: str,
    *,
    document_id: uuid.UUID | None = None,
    on_date: date | None = None,
) -> tuple[TariffDocument, str]:
    """The ready document that covers `port` and the port's canonical name in
    it. With document_id, that document must cover the port; otherwise the
    newest ready document covering it (and in force on `on_date`, if given)."""
    if document_id is not None:
        document = await get_document(session, document_id)
        if document.status != DocumentStatus.READY:
            raise DocumentNotReadyError(f"Document {document_id} is {document.status}, not ready")
        candidates = [document]
    else:
        candidates = list(
            await session.scalars(
                select(TariffDocument)
                .where(TariffDocument.status == DocumentStatus.READY)
                .order_by(TariffDocument.created_at.desc())
            )
        )

    for document in candidates:
        port_key = match_port(port, document.ports)
        if port_key is None or not _in_force(document, on_date):
            continue
        return document, port_key
    scope = f"document {document_id}" if document_id else "any ingested tariff document"
    raise PortNotCoveredError(f"Port {port!r} is not covered by {scope}")


def _in_force(document: TariffDocument, on_date: date | None) -> bool:
    if on_date is None:
        return True
    if document.effective_from and on_date < document.effective_from:
        return False
    return not (document.effective_to and on_date > document.effective_to)
