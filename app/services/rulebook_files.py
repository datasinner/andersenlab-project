"""Rulebook files: a document's compiled rules, portable between databases.

Compiling a port's rulebook takes minutes and about a million tokens, and the
result is reviewed before use, so a fresh database should load it rather
than rebuild it. export_rulebook captures, for one ingested document, what
its rules were compiled against and the rules themselves:

- the document profile (ports, currency, VAT, effective dates) and the charge
  catalogue, which come from LLM calls at ingestion and could come out
  differently in another database;
- the compiled rules in use for the current rule schema, per port, each with
  the prompt version it was compiled with and the critic's verdict;
- the position in the document of every chunk a rule cites (chunk ids are
  database ids).

import_rulebook loads a file into a database where the same PDF (same
checksum) is ingested: it adopts the profile and catalogue, points each
citation at the chunk that holds its quote here, and caches the rules. A
file is skipped when its document isn't ingested, when it was compiled for
another rule schema, or with older prompts while RULES_REUSE_OLDER_PROMPTS is
off (its rules would never be used), or when the document already has
compiled rules here. A rule whose quotes can't be found is left out and
compiles on demand. Ingestion also uses a file's profile and catalogue
instead of asking the model again (find_rulebook_file).

The file is data produced from the document, like the cache it fills; it
adds no tariff knowledge to the code.
"""

import copy
import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import settings
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
from app.services.rulebook import effective_rules, prompt_version

FORMAT_VERSION = 1


@dataclass(frozen=True)
class ImportResult:
    imported: bool
    message: str
    document_id: uuid.UUID | None = None
    rules: int = 0
    left_out: list[str] = field(default_factory=list)  # charge ids whose quotes weren't found


async def export_rulebook(session: AsyncSession, document: TariffDocument) -> dict[str, Any]:
    """The document's profile, catalogue and the compiled rules in use (all ports)."""
    cached = list(
        await session.scalars(
            select(CompiledRule).where(
                CompiledRule.document_id == document.id,
                CompiledRule.rule_schema_version == RULE_SCHEMA_VERSION,
            )
        )
    )
    rows = sorted(
        (
            row
            for port_key in {row.port_key for row in cached}
            for row in effective_rules(row for row in cached if row.port_key == port_key)
        ),
        key=lambda row: (row.port_key, row.charge_id),
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
                "prompt_version": row.prompt_version,
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
    if data.get("rule_schema_version") != RULE_SCHEMA_VERSION:
        return ImportResult(
            False,
            f"compiled for rule schema {data.get('rule_schema_version')}; this version uses "
            f"{RULE_SCHEMA_VERSION}. Recompile and export it again",
        )
    versions = {
        _entry_prompts(data, entry) for entries in data["ports"].values() for entry in entries
    }
    if not settings.rules_reuse_older_prompts and versions - {prompt_version()}:
        return ImportResult(
            False,
            f"compiled with other prompts ({', '.join(sorted(versions))}; this version uses "
            f"{prompt_version()}) and RULES_REUSE_OLDER_PROMPTS is off. Recompile and export "
            "it again",
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
                        prompt_version=_entry_prompts(data, entry),
                        model=entry["model"],
                        critic_verdict=entry["critic_verdict"],
                    )
                )

        await apply_profile_and_catalogue(session, document, data)
        session.add_all(rows)
        await session.commit()
        return ImportResult(
            True, f"{len(rows)} rule(s) loaded", document.id, rules=len(rows), left_out=left_out
        )


def find_rulebook_file(directory: str | Path | None, checksum: str) -> dict[str, Any] | None:
    """The rulebook file in `directory` exported from the PDF with this checksum."""
    if not directory:
        return None
    for path in sorted(Path(directory).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("document", {}).get("checksum") == checksum:
            return data
    return None


async def apply_profile_and_catalogue(
    session: AsyncSession, document: TariffDocument, data: dict[str, Any]
) -> None:
    """Give the document the profile and charge catalogue from a rulebook
    file (not committed)."""
    source = data["document"]
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


def _entry_prompts(data: dict[str, Any], entry: dict[str, Any]) -> str:
    return entry.get("prompt_version") or data["prompt_version"]


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
