import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import TariffDocument
from app.schemas import (
    ChargeOut,
    DocumentDetail,
    DocumentSummary,
    PortOut,
    SectionChild,
    SectionOut,
)
from app.services import documents

router = APIRouter(prefix="/v1/documents", tags=["documents"])


def _port_names(document: TariffDocument) -> list[str]:
    return [port["name"] for port in document.ports]


@router.get("", response_model=list[DocumentSummary])
async def list_documents(session: AsyncSession = Depends(get_session)) -> list[DocumentSummary]:
    return [
        DocumentSummary(
            id=document.id,
            title=document.title,
            authority=document.authority,
            source_filename=document.source_filename,
            status=document.status,
            currency=document.currency,
            ports=_port_names(document),
            created_at=document.created_at,
        )
        for document in await documents.list_documents(session)
    ]


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> DocumentDetail:
    document = await documents.get_document(session, document_id)
    counts = await documents.count_contents(session, document_id)
    return DocumentDetail(
        id=document.id,
        title=document.title,
        authority=document.authority,
        source_filename=document.source_filename,
        checksum=document.checksum,
        status=document.status,
        error=document.error,
        page_count=document.page_count,
        currency=document.currency,
        vat_rate=document.vat_rate,
        effective_from=document.effective_from,
        effective_to=document.effective_to,
        ports=[PortOut(**port) for port in document.ports],
        sections=counts.sections,
        chunks=counts.chunks,
        charges=counts.charges,
        created_at=document.created_at,
    )


@router.get("/{document_id}/charges", response_model=list[ChargeOut])
async def list_charges(
    document_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> list[ChargeOut]:
    return [
        ChargeOut(
            charge_id=charge.charge_id,
            name=charge.name,
            section_refs=charge.section_refs,
            payer=charge.payer,
            trigger=charge.trigger,
            description=charge.description,
        )
        for charge in await documents.list_charges(session, document_id)
    ]


@router.get("/{document_id}/sections/{ref}", response_model=SectionOut)
async def get_section(
    document_id: uuid.UUID, ref: str, session: AsyncSession = Depends(get_session)
) -> SectionOut:
    section = await documents.get_section(session, document_id, ref)
    return SectionOut(
        ref=section.ref,
        title=section.title,
        path=section.path,
        page_start=section.page_start,
        page_end=section.page_end,
        text=section.text,
        children=[
            SectionChild(ref=child_ref, title=title) for child_ref, title in section.children
        ],
    )
