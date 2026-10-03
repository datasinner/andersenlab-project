"""Rulebook files: a document's compiled rules, portable between databases.

Compiling a port's rulebook takes minutes and about a million tokens, and the
result is reviewed before use, so a fresh database should load it rather
than rebuild it. export_rulebook captures, for one ingested document, what
its rules were compiled against and the rules themselves:

- the document profile (ports, currency, VAT, effective dates) and the charge
  catalogue, which come from LLM calls at ingestion and could come out
  differently in another database;
- the compiled rules for the current rule schema and compile prompts, per
  port, with the critic's verdict;
- the position in the document of every chunk a rule cites (chunk ids are
  database ids).

import_rulebook loads a file into a database where the same PDF (same
checksum) is ingested: it adopts the profile and catalogue, points each
citation at the chunk that holds its quote here, and caches the rules. A
file is skipped when its document isn't ingested, when it was compiled with
other prompts or another rule schema (its rules would never be used), or
when the document already has compiled rules here. A rule whose quotes can't
be found is left out and compiles on demand.

The file is data produced from the document, like the cache it fills; it
adds no tariff knowledge to the code.
"""

import copy
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import ValidationError
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import (
    ChargeCatalogueEntry,
    Chunk,
    CompiledRule,
    DocumentSection,
    DocumentStatus,
    TariffDocument,
)
from app.rules.dsl import RULE_SCHEMA_VERSION, ChargeRule
from app.rules.grounding import quote_in
from app.services.rulebook import prompt_version

FORMAT_VERSION = 1


@dataclass(frozen=True)
class ImportResult:
    imported: bool
    message: str
    document_id: uuid.UUID | None = None
    rules: int = 0
    left_out: list[str] = field(default_factory=list)  # charge ids whose quotes weren't found


async def export_rulebook(session: AsyncSession, document: TariffDocument) -> dict[str, Any]:
    """The document's profile, catalogue and current compiled rules (all ports)."""
    rows = list(
        await session.scalars(
            select(CompiledRule)
            .where(
                CompiledRule.document_id == document.id,
                CompiledRule.rule_schema_version == RULE_SCHEMA_VERSION,
                CompiledRule.prompt_version == prompt_version(),
            )
            .order_by(CompiledRule.port_key, CompiledRule.charge_id)
        )
    )
    catalogue = await session.scalars(
        select(ChargeCatalogueEntry)
        .where(ChargeCatalogueEntry.document_id == document.id)
        .order_by(ChargeCatalogueEntry.id)
    )
    cited = {citation["chunk_id"] for row in rows if row.rule for citation in _citations(row.rule)}
    ordinals = dict(
        (await session.execute(select(Chunk.id, Chunk.ordinal).where(Chunk.id.in_(cited)))).all()
    )

    ports: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ports.setdefault(row.port_key, []).append(
            {
                "charge_id": row.charge_id,
                "model": row.model,
                "critic_verdict": row.critic_verdict,
                "rule": row.rule,
            }
        )
    return {
        "format_version": FORMAT_VERSION,
        "rule_schema_version": RULE_SCHEMA_VERSION,
        "prompt_version": prompt_version(),
        "document": {
            "checksum": document.checksum,
            "source_filename": document.source_filename,
            "title": document.title,
            "authority": document.authority,
            "currency": document.currency,
            "vat_rate": str(document.vat_rate) if document.vat_rate is not None else None,
            "effective_from": _iso(document.effective_from),
            "effective_to": _iso(document.effective_to),
            "ports": document.ports,
        },
        "catalogue": [
            {
                "charge_id": entry.charge_id,
                "name": entry.name,
                "section_refs": entry.section_refs,
                "payer": entry.payer,
                "trigger": entry.trigger,
                "description": entry.description,
            }
            for entry in catalogue
        ],
        "chunk_ordinals": {str(chunk_id): ordinal for chunk_id, ordinal in ordinals.items()},
        "ports": ports,
    }


async def import_rulebook(
    session_factory: async_sessionmaker[AsyncSession], data: dict[str, Any]
) -> ImportResult:
    if data.get("format_version") != FORMAT_VERSION:
        return ImportResult(False, f"unknown format {data.get('format_version')!r}")
    if (
        data.get("rule_schema_version") != RULE_SCHEMA_VERSION
        or data.get("prompt_version") != prompt_version()
    ):
        return ImportResult(
            False,
            "compiled with other prompts or rule schema "
            f"({data.get('prompt_version')}, schema {data.get('rule_schema_version')}; this "
            f"version uses {prompt_version()}, schema {RULE_SCHEMA_VERSION}); recompile and "
            "export it again",
        )

    async with session_factory() as session:
        source = data["document"]
        document = await session.scalar(
            select(TariffDocument).where(TariffDocument.checksum == source["checksum"])
        )
        if document is None or document.status != DocumentStatus.READY:
            return ImportResult(False, f"{source['source_filename']} is not ingested here")
        existing = await session.scalar(
            select(CompiledRule.id)
            .where(
                CompiledRule.document_id == document.id,
                CompiledRule.rule_schema_version == RULE_SCHEMA_VERSION,
                CompiledRule.prompt_version == prompt_version(),
            )
            .limit(1)
        )
        if existing is not None:
            return ImportResult(False, "the document already has compiled rules here", document.id)

        chunks = await _document_chunks(session, document.id)
        ordinals = {int(old): ordinal for old, ordinal in data["chunk_ordinals"].items()}
        rows: list[CompiledRule] = []
        left_out: list[str] = []
        for port_key, entries in data["ports"].items():
            for entry in entries:
                rule = _retarget(entry["rule"], ordinals, chunks) if entry["rule"] else None
                if entry["rule"] and rule is None:
                    left_out.append(entry["charge_id"])
                    continue
                rows.append(
                    CompiledRule(
                        document_id=document.id,
                        port_key=port_key,
                        charge_id=entry["charge_id"],
                        rule=rule,
                        rule_schema_version=RULE_SCHEMA_VERSION,
                        prompt_version=prompt_version(),
                        model=entry["model"],
                        critic_verdict=entry["critic_verdict"],
                    )
                )

        await session.execute(
            delete(ChargeCatalogueEntry).where(ChargeCatalogueEntry.document_id == document.id)
        )
        session.add_all(
            ChargeCatalogueEntry(document_id=document.id, **entry) for entry in data["catalogue"]
        )
        document.title = source["title"]
        document.authority = source["authority"]
        document.currency = source["currency"]
        document.vat_rate = Decimal(source["vat_rate"]) if source["vat_rate"] else None
        document.effective_from = _date(source["effective_from"])
        document.effective_to = _date(source["effective_to"])
        document.ports = source["ports"]
        session.add_all(rows)
        await session.commit()
        return ImportResult(
            True, f"{len(rows)} rule(s) loaded", document.id, rules=len(rows), left_out=left_out
        )


@dataclass(frozen=True)
class _ChunkText:
    id: int
    section_ref: str
    content: str


async def _document_chunks(session: AsyncSession, document_id: uuid.UUID) -> dict[int, _ChunkText]:
    rows = await session.execute(
        select(Chunk.ordinal, Chunk.id, DocumentSection.ref, Chunk.content)
        .join(DocumentSection, Chunk.section_id == DocumentSection.id)
        .where(Chunk.document_id == document_id)
    )
    return {ordinal: _ChunkText(id, ref, content) for ordinal, id, ref, content in rows.all()}


def _retarget(
    rule: dict[str, Any], ordinals: dict[int, int], chunks: dict[int, _ChunkText]
) -> dict[str, Any] | None:
    """The rule with each citation pointing at this database's chunk for its
    quote: the chunk at the same position if it holds the quote, else any
    chunk that does (one in the cited section first). None if a quote is
    nowhere, or the result is not a valid rule."""
    rule = copy.deepcopy(rule)
    for citation in _citations(rule):
        same_place = chunks.get(ordinals.get(citation["chunk_id"], -1))
        candidates = [same_place] if same_place else []
        candidates += sorted(
            chunks.values(), key=lambda chunk: chunk.section_ref != citation["section_ref"]
        )
        found = next((c for c in candidates if quote_in(citation["quote"], c.content)), None)
        if found is None:
            return None
        citation["chunk_id"] = found.id
    try:
        ChargeRule.model_validate(rule)
    except ValidationError:
        return None
    return rule


def _citations(rule: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield from rule.get("citations", [])
    for holder in (*rule.get("exemptions", []), *rule.get("adjustments", [])):
        if holder.get("citation"):
            yield holder["citation"]


def _iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None
