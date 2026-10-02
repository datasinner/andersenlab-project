"""The document profile: issuer, ports, currency, VAT and validity period,
read by one structured LLM call from the opening pages, the outline and the
table headers (port names usually appear as table columns)."""

from datetime import date

from pydantic import BaseModel, Field

from app.ingestion.parser import ParsedDocument, TableBlock, TextLine
from app.ingestion.structure import Section
from app.llm.client import LLMClient, Usage
from app.llm.prompts import PROMPTS
from app.models import SectionKind

_OPENING_CHARS = 6000
_MAX_TABLE_HEADERS = 40


class PortEntry(BaseModel):
    name: str = Field(min_length=1)
    aliases: list[str]


class DocumentProfile(BaseModel):
    title: str | None
    authority: str | None
    currency: str | None = Field(description="ISO 4217 code, e.g. EUR")
    vat_percent: str | None = Field(description="Percentage as printed, e.g. '15'")
    effective_from: date | None
    effective_to: date | None
    ports: list[PortEntry]


async def read_profile(
    llm: LLMClient, document: ParsedDocument, sections: list[Section]
) -> tuple[DocumentProfile, Usage]:
    messages = PROMPTS["document_profile"].render(
        running_text="\n".join(document.running_text) or "(none)",
        opening_text=_opening_text(sections),
        outline="\n".join(f"{'  ' * (s.level - 1)}{s.title}" for s in sections) or "(none)",
        table_headers=_table_headers(sections) or "(none)",
    )
    result = await llm.generate_structured(DocumentProfile, messages, name="document_profile")
    return result.value, result.usage


def _opening_text(sections: list[Section]) -> str:
    lines: list[str] = []
    length = 0
    for section in sections:
        if section.kind == SectionKind.CONTENTS:
            continue
        for placed in section.blocks:
            if isinstance(placed.element, TextLine):
                lines.append(placed.element.text)
                length += len(placed.element.text)
            if length >= _OPENING_CHARS:
                return "\n".join(lines)
    return "\n".join(lines)


def _table_headers(sections: list[Section]) -> str:
    headers: list[str] = []
    for section in sections:
        for placed in section.blocks:
            if isinstance(placed.element, TableBlock):
                header = placed.element.markdown.splitlines()[0].replace("<br>", " ")
                if header not in headers:
                    headers.append(header)
    return "\n".join(headers[:_MAX_TABLE_HEADERS])
