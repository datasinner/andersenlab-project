import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Response, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_session
from app.dependencies import get_ingestion
from app.ingestion.pipeline import IngestionPipeline
from app.models import DocumentStatus, TariffDocument
from app.schemas import (
    ChargeOut,
    DocumentDetail,
    DocumentSummary,
    PortOut,
    SectionChild,
    SectionOut,
    UploadOut,
)
from app.services import documents

router = APIRouter(prefix="/v1/documents", tags=["documents"])


def _port_names(document: TariffDocument) -> list[str]:
    return [port["name"] for port in document.ports]


@router.post(
    "",
    response_model=UploadOut,
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"model": UploadOut, "description": "Already ingested"}},
)
async def upload_document(
    response: Response,
    background: BackgroundTasks,
    file: UploadFile = File(description="A tariff document (PDF)."),
    force: bool = False,
    pipeline: IngestionPipeline = Depends(get_ingestion),
) -> UploadOut:
    """Upload a tariff PDF. It is ingested in the background (parsing,
    indexing, profile and charge catalogue); poll GET /v1/documents/{id}
    until its status is `ready`. Uploading the same file again returns the
    existing document (200), unless `force` rebuilds it."""
    limit = settings.max_upload_mb * 1024 * 1024
    content = await file.read(limit + 1)
    if len(content) > limit:
        raise documents.UploadTooLargeError(f"The file is larger than {settings.max_upload_mb} MB")
    if not content.startswith(b"%PDF"):
        raise documents.NotAPdfError("The file is not a PDF")
    filename = Path(file.filename or "tariff.pdf").name[:255]

    registration = await pipeline.register(content, filename, force=force)
    if not registration.needs_processing:
        response.status_code = status.HTTP_200_OK
        return UploadOut(
            document_id=registration.document_id, status=DocumentStatus.READY, created=False
        )
    background.add_task(pipeline.process, registration.document_id, content, filename)
    return UploadOut(
        document_id=registration.document_id, status=DocumentStatus.PARSING, created=True
    )


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
