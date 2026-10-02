"""Read access to ingested tariff documents."""

import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.errors import AppError
from app.models import ChargeCatalogueEntry, Chunk, DocumentSection, TariffDocument
from app.retrieval.context import SectionText, read_section


class DocumentNotFoundError(AppError):
    status_code = 404
    code = "DOCUMENT_NOT_FOUND"


class SectionNotFoundError(AppError):
    status_code = 404
    code = "SECTION_NOT_FOUND"


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
