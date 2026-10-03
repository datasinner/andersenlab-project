from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


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


SUDESTADA_EXAMPLE = {
    "port": "Durban",
    "vessel": {
        "vessel_metadata": {"name": "SUDESTADA", "built_year": 2010, "flag": "MLT - Malta"},
        "technical_specs": {
            "type": "Bulk Carrier",
            "dwt": 93274,
            "gross_tonnage": 51300,
            "net_tonnage": 31192,
            "loa_meters": 229.2,
            "beam_meters": 38.0,
            "draft_sw_s_w_t": [14.9, 0.0, 0.0],
        },
        "operational_data": {
            "cargo_quantity_mt": 40000,
            "days_alongside": 3.39,
            "arrival_time": "2024-11-15T10:12:00",
            "departure_time": "2024-11-22T13:00:00",
            "activity": "Exporting Iron Ore",
        },
    },
}


class CalculationOverrides(BaseModel):
    num_services: int | None = Field(
        default=None, ge=0, description="Marine-service movements (default 2: entering + leaving)."
    )
    facts: dict[str, bool | int | float | str] = Field(
        default_factory=dict,
        description="Fact values by name; they win over what the agent decides.",
    )


class CalculationRequest(BaseModel):
    model_config = {"json_schema_extra": {"examples": [SUDESTADA_EXAMPLE]}}

    port: str | None = Field(default=None, max_length=100)
    vessel: dict[str, Any] | None = Field(
        default=None,
        description="Vessel profile: vessel_metadata / technical_specs / operational_data, "
        "or {vessel, call}.",
    )
    query: str | None = Field(
        default=None, max_length=4000, description="The call described in plain language."
    )
    document_id: UUID | None = None
    overrides: CalculationOverrides = Field(default_factory=CalculationOverrides)
    charge_ids: list[str] | None = Field(default=None, description="Price only these charges.")
    requested_charges: list[str] = Field(
        default_factory=list, description="On-request charges (e.g. fresh water) to include."
    )
    refresh_rules: bool = Field(default=False, description="Recompile cached rules first.")

    @model_validator(mode="after")
    def _vessel_or_query(self) -> "CalculationRequest":
        if not self.vessel and not (self.query and self.query.strip()):
            raise ValueError("give a vessel profile, a query, or both")
        return self


class CitationOut(BaseModel):
    chunk_id: int
    section_ref: str
    page: int
    quote: str


class AdjustmentOut(BaseModel):
    id: str
    kind: str
    description: str
    percent: Decimal
    amount: Decimal


class FactOut(BaseModel):
    name: str
    value: str
    source: str
    reason: str


class LineItemOut(BaseModel):
    charge_id: str
    name: str
    section_refs: list[str]
    amount: Decimal
    currency: str
    confidence: str  # high | low
    formula: list[str]
    adjustments: list[AdjustmentOut]
    assumptions: list[str]
    facts: list[FactOut]
    citations: list[CitationOut]
    notes: list[str]
    review_notes: list[str]
    open_issues: list[str]
    rule_source: str  # cache | compiled


class OtherChargeOut(BaseModel):
    charge_id: str
    name: str
    section_refs: list[str]
    reason: str
    facts: list["FactOut"] = []


class FailedChargeOut(BaseModel):
    charge_id: str
    name: str
    error: str


class DocumentRef(BaseModel):
    id: UUID
    title: str | None
    authority: str | None


class VatOut(BaseModel):
    rate: Decimal | None
    included: bool


class CalculationOut(BaseModel):
    calculation_id: UUID
    status: str  # success | partial | failed
    port: str
    document: DocumentRef
    vessel: dict[str, Any]
    currency: str | None
    vat: VatOut
    line_items: list[LineItemOut]
    total: Decimal
    not_applicable: list[OtherChargeOut]
    not_priced: list[OtherChargeOut]
    on_request: list[OtherChargeOut]
    excluded: list[OtherChargeOut]
    failed: list[FailedChargeOut]
    warnings: list[str]
    model: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int


class AgentStepOut(BaseModel):
    charge_id: str | None
    node: str
    tool: str | None
    input_summary: str | None
    output: dict[str, Any] | None
    latency_ms: int


class CalculationRecordOut(BaseModel):
    calculation_id: UUID
    created_at: datetime
    status: str
    error_code: str | None
    port: str | None
    query: str | None
    latency_ms: int
    result: dict[str, Any] | None
    steps: list[AgentStepOut]


class CalculationSummaryOut(BaseModel):
    calculation_id: UUID
    created_at: datetime
    status: str
    port: str | None
    vessel_name: str | None
    total: Decimal | None
    latency_ms: int
    error_code: str | None


class UploadOut(BaseModel):
    document_id: UUID
    status: str
    created: bool  # False when these exact bytes were already ingested
