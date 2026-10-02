"""The charge catalogue: every charge the document defines, discovered by
one structured LLM call over the outline. This is how the system knows
which charges exist without a hard-coded list of charge names."""

from typing import Literal

import structlog
from pydantic import BaseModel, Field

from app.ingestion.chunker import join_lines
from app.ingestion.parser import TableBlock, TextLine
from app.ingestion.structure import Section
from app.llm.client import LLMClient, Usage
from app.llm.prompts import PROMPTS
from app.models import SectionKind
from app.rules.dsl import Identifier

logger = structlog.get_logger("app.ingestion")

_OPENING_CHARS = 300


class CatalogueEntry(BaseModel):
    charge_id: Identifier
    name: str = Field(min_length=1)
    section_refs: list[str] = Field(min_length=1)
    payer: Literal["vessel", "cargo", "other"]
    trigger: Literal["per_call", "per_service", "per_period", "on_request", "licence_or_permit"]
    description: str = Field(min_length=1)


class ChargeCatalogue(BaseModel):
    charges: list[CatalogueEntry]


async def discover_charges(
    llm: LLMClient, sections: list[Section]
) -> tuple[list[CatalogueEntry], Usage]:
    messages = PROMPTS["charge_catalogue"].render(outline=outline_text(sections))
    result = await llm.generate_structured(ChargeCatalogue, messages, name="charge_catalogue")
    return _clean_entries(result.value.charges, {section.ref for section in sections}), result.usage


def outline_text(sections: list[Section]) -> str:
    lines = []
    for section in sections:
        if section.kind == SectionKind.CONTENTS:
            continue
        indent = "  " * (section.level - 1)
        opening = _opening(section)
        lines.append(
            f"{indent}{section.ref} | {section.title}" + (f" | {opening}" if opening else "")
        )
    return "\n".join(lines)


def _opening(section: Section) -> str:
    texts: list[str] = []
    for placed in section.blocks:
        element = placed.element
        if isinstance(element, TextLine):
            texts.append(element.text)
        elif isinstance(element, TableBlock):
            texts.append(element.markdown.splitlines()[0].replace("<br>", " "))
        if sum(len(text) for text in texts) >= _OPENING_CHARS:
            break
    return " ".join(join_lines(texts).split())[:_OPENING_CHARS]


def _clean_entries(entries: list[CatalogueEntry], known_refs: set[str]) -> list[CatalogueEntry]:
    """Resolve section refs to sections that exist (a numbered item such as
    "4.11.1" that isn't a section of its own maps to its nearest ancestor,
    "4.11"), drop entries left with none, and make charge ids unique."""
    cleaned: list[CatalogueEntry] = []
    seen: set[str] = set()
    for entry in entries:
        resolved = [_resolve_ref(ref, known_refs) for ref in entry.section_refs]
        unknown = [
            ref for ref, match in zip(entry.section_refs, resolved, strict=True) if not match
        ]
        if unknown:
            logger.warning("catalogue_unknown_refs", charge_id=entry.charge_id, unknown=unknown)
        refs = list(dict.fromkeys(ref for ref in resolved if ref))
        if not refs:
            continue
        charge_id = entry.charge_id
        suffix = 2
        while charge_id in seen:
            charge_id = f"{entry.charge_id}_{suffix}"
            suffix += 1
        seen.add(charge_id)
        cleaned.append(entry.model_copy(update={"charge_id": charge_id, "section_refs": refs}))
    return cleaned


def _resolve_ref(ref: str, known_refs: set[str]) -> str | None:
    candidate = ref.strip().rstrip(".")
    while candidate:
        if candidate in known_refs:
            return candidate
        candidate, _, _ = candidate.rpartition(".")
    return None
