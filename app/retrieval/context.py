"""Read-only views of a document for the agent: a whole section, the
outline, and definitions of a term."""

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, ChunkKind, DocumentSection, SectionKind


@dataclass(frozen=True)
class SectionText:
    ref: str
    title: str
    path: str
    page_start: int
    page_end: int
    text: str  # the section's own chunks, in order, without repeated breadcrumbs
    chunk_ids: list[int]
    children: list[tuple[str, str]]  # (ref, title) of direct subsections


async def read_section(
    session: AsyncSession, document_id: uuid.UUID, ref: str
) -> SectionText | None:
    section = await session.scalar(
        select(DocumentSection).where(
            DocumentSection.document_id == document_id, DocumentSection.ref == ref
        )
    )
    if section is None:
        return None
    chunks = list(
        await session.scalars(
            select(Chunk).where(Chunk.section_id == section.id).order_by(Chunk.ordinal)
        )
    )
    children = list(
        await session.execute(
            select(DocumentSection.ref, DocumentSection.title)
            .where(DocumentSection.parent_id == section.id)
            .order_by(DocumentSection.ordinal)
        )
    )
    return SectionText(
        ref=section.ref,
        title=section.title,
        path=section.path,
        page_start=section.page_start,
        page_end=section.page_end,
        text="\n\n".join(chunk_body(chunk.content) for chunk in chunks),
        chunk_ids=[chunk.id for chunk in chunks],
        children=[(child_ref, title) for child_ref, title in children],
    )


async def outline(
    session: AsyncSession, document_id: uuid.UUID, prefix: str | None = None
) -> list[DocumentSection]:
    """Sections in document order, optionally only those whose ref starts
    with `prefix`. Contents pages are left out."""
    query = (
        select(DocumentSection)
        .where(
            DocumentSection.document_id == document_id,
            DocumentSection.kind != SectionKind.CONTENTS,
        )
        .order_by(DocumentSection.ordinal)
    )
    if prefix:
        query = query.where(DocumentSection.ref.startswith(prefix, autoescape=True))
    return list(await session.scalars(query))


async def find_definitions(
    session: AsyncSession, document_id: uuid.UUID, term: str, limit: int = 5
) -> list[Chunk]:
    """Definition chunks that mention `term` (case-insensitive)."""
    result = await session.scalars(
        select(Chunk)
        .where(
            Chunk.document_id == document_id,
            Chunk.kind == ChunkKind.DEFINITION,
            Chunk.content.icontains(term, autoescape=True),
        )
        .order_by(Chunk.ordinal)
        .limit(limit)
    )
    return list(result)


def chunk_body(content: str) -> str:
    """A chunk's text without the breadcrumb line every chunk starts with."""
    _, separator, body = content.partition("\n\n")
    return body if separator else content
