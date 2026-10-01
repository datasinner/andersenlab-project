import uuid
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Computed,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, deferred, mapped_column
from sqlalchemy.sql.naming import conv

from app.config import settings

# Deterministic constraint names, so Alembic migrations (and their
# downgrades) can refer to constraints by name.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PARSING = "parsing"
    INDEXING = "indexing"
    CATALOGUING = "cataloguing"
    READY = "ready"
    FAILED = "failed"


class SectionKind(StrEnum):
    BODY = "body"
    DEFINITIONS = "definitions"
    CONTENTS = "contents"


class ChunkKind(StrEnum):
    TEXT = "text"
    TABLE = "table"
    DEFINITION = "definition"


class ChargePayer(StrEnum):
    VESSEL = "vessel"
    CARGO = "cargo"
    OTHER = "other"


class ChargeTrigger(StrEnum):
    PER_CALL = "per_call"
    PER_SERVICE = "per_service"
    PER_PERIOD = "per_period"
    ON_REQUEST = "on_request"
    LICENCE_OR_PERMIT = "licence_or_permit"


class CalculationStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


def _one_of(column: str, enum: type[StrEnum]) -> CheckConstraint:
    allowed = ", ".join(f"'{member.value}'" for member in enum)
    return CheckConstraint(f"{column} IN ({allowed})", name=column)


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class TariffDocument(Base):
    """One uploaded tariff PDF. A new edition is a new row; rows are never
    updated in place once ready, so compiled rules and calculations always
    point at the exact document text that produced them."""

    __tablename__ = "tariff_document"
    __table_args__ = (_one_of("status", DocumentStatus),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str | None] = mapped_column(String(300))
    authority: Mapped[str | None] = mapped_column(String(300))
    source_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    # Deferred so listing documents never loads the PDF bytes.
    content: Mapped[bytes] = deferred(mapped_column(LargeBinary, nullable=False))
    checksum: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    page_count: Mapped[int | None] = mapped_column(Integer)
    currency: Mapped[str | None] = mapped_column(String(3))
    vat_rate: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    effective_from: Mapped[date | None] = mapped_column(Date)
    effective_to: Mapped[date | None] = mapped_column(Date)
    # Ports the document covers: [{"name": ..., "aliases": [...]}].
    ports: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=DocumentStatus.PENDING.value
    )
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class DocumentSection(Base):
    __tablename__ = "document_section"
    __table_args__ = (
        UniqueConstraint("document_id", "ordinal"),
        _one_of("kind", SectionKind),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tariff_document.id", ondelete="CASCADE"), nullable=False
    )
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_section.id", ondelete="CASCADE")
    )
    # The document's own numbering (e.g. "3.6"), or a generated "s-17" when
    # the document doesn't number its headings.
    ref: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=SectionKind.BODY.value
    )


class Chunk(Base):
    __tablename__ = "chunk"
    __table_args__ = (_one_of("kind", ChunkKind),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tariff_document.id", ondelete="CASCADE"), nullable=False
    )
    section_id: Mapped[int] = mapped_column(
        ForeignKey("document_section.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    # Markdown, prefixed with the section breadcrumb.
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    printed_page: Mapped[str | None] = mapped_column(String(20))
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Null until the indexing step has embedded the chunk.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(settings.embedding_dimensions))
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True)
    )


class ChargeCatalogueEntry(Base):
    """A charge the document defines, as discovered by the catalogue step."""

    __tablename__ = "charge_catalogue_entry"
    __table_args__ = (
        UniqueConstraint("document_id", "charge_id"),
        _one_of("payer", ChargePayer),
        _one_of("trigger", ChargeTrigger),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tariff_document.id", ondelete="CASCADE"), nullable=False
    )
    charge_id: Mapped[str] = mapped_column(String(100), nullable=False)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    section_refs: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    payer: Mapped[str] = mapped_column(String(20), nullable=False)
    trigger: Mapped[str] = mapped_column(String(30), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _created_at()


class CompiledRule(Base):
    """A ChargeRule the agent extracted for one port, written only after it
    passed validation and the critic. Bumping the rule schema or a prompt
    version makes old rows unreachable without a migration."""

    __tablename__ = "compiled_rule"
    __table_args__ = (
        # Named explicitly: the naming convention would produce an 82-character
        # name, past Postgres' 63-character identifier limit.
        UniqueConstraint(
            "document_id",
            "port_key",
            "charge_id",
            "rule_schema_version",
            "prompt_version",
            name=conv("uq_compiled_rule_cache_key"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tariff_document.id", ondelete="CASCADE"), nullable=False
    )
    port_key: Mapped[str] = mapped_column(String(100), nullable=False)
    charge_id: Mapped[str] = mapped_column(String(100), nullable=False)
    rule: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    rule_schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    critic_verdict: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _created_at()


class Calculation(Base):
    """One pricing request. Written for failures too: silently losing a
    failed calculation is a defect."""

    __tablename__ = "calculation"
    __table_args__ = (_one_of("status", CalculationStatus),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # Null when the request failed before a document was resolved.
    document_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("tariff_document.id"))
    port_key: Mapped[str | None] = mapped_column(String(100))
    vessel_call: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    query_text: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    total: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    error_code: Mapped[str | None] = mapped_column(String(80))
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = _created_at()


class AgentStep(Base):
    """Audit trail of what the agent did for a calculation: which tools it
    called, what it extracted, and what the critic said."""

    __tablename__ = "agent_step"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    calculation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("calculation.id", ondelete="CASCADE"), nullable=False
    )
    charge_id: Mapped[str | None] = mapped_column(String(100))
    node: Mapped[str] = mapped_column(String(50), nullable=False)
    tool: Mapped[str | None] = mapped_column(String(50))
    input_summary: Mapped[str | None] = mapped_column(Text)
    output: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = _created_at()


Index("ix_document_section_document_ref", DocumentSection.document_id, DocumentSection.ref)
Index("ix_document_section_parent", DocumentSection.parent_id)
Index("ix_chunk_document_section", Chunk.document_id, Chunk.section_id)
Index(
    "ix_chunk_embedding_hnsw",
    Chunk.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)
Index("ix_chunk_tsv", Chunk.tsv, postgresql_using="gin")
Index("ix_compiled_rule_lookup", CompiledRule.document_id, CompiledRule.port_key)
Index("ix_calculation_created", Calculation.created_at.desc())
Index("ix_calculation_port_created", Calculation.port_key, Calculation.created_at.desc())
Index("ix_calculation_status", Calculation.status)
Index("ix_agent_step_calculation", AgentStep.calculation_id, AgentStep.id)
