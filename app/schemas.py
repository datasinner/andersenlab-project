from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


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


class CompileRequest(BaseModel):
    port: str = Field(min_length=1, max_length=100)
    document_id: UUID | None = None
    charge_ids: list[str] | None = Field(
        default=None, description="Compile only these charges (ids from the catalogue)."
    )
    include_all: bool = Field(
        default=False, description="Compile every charge, not only those a vessel routinely pays."
    )
    refresh: bool = Field(default=False, description="Recompile rules that are already cached.")


class CompiledRuleOut(BaseModel):
    charge_id: str
    name: str
    status: str
    from_cache: bool
    revisions: int
    issues: list[str]
    review_notes: list[str]
    sections_read: list[str]
    research_notes: str
    model: str | None
    compiled_at: datetime | None
    error: str | None
    rule: dict[str, Any] | None


class RulebookOut(BaseModel):
    document_id: UUID
    port: str
    prompt_version: str
    prompt_tokens: int
    completion_tokens: int
    rules: list[CompiledRuleOut]
