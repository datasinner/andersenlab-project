from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    documents_ready: int | None = None


class PortOut(BaseModel):
    name: str
    aliases: list[str] = []


class DocumentSummary(BaseModel):
    id: UUID
    title: str | None
    authority: str | None
    source_filename: str
    status: str
    currency: str | None
    ports: list[str]
    created_at: datetime


class DocumentDetail(BaseModel):
    id: UUID
    title: str | None
    authority: str | None
    source_filename: str
    checksum: str
    status: str
    error: str | None
    page_count: int | None
    currency: str | None
    vat_rate: Decimal | None
    effective_from: date | None
    effective_to: date | None
    ports: list[PortOut]
    sections: int
    chunks: int
    charges: int
    created_at: datetime


class ChargeOut(BaseModel):
    charge_id: str
    name: str
    section_refs: list[str]
    payer: str
    trigger: str
    description: str


class SectionChild(BaseModel):
    ref: str
    title: str


class SectionOut(BaseModel):
    ref: str
    title: str
    path: str
    page_start: int
    page_end: int
    text: str
    children: list[SectionChild]
