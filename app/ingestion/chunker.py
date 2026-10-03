"""Sections → retrieval chunks.

- One chunk per section, split at line boundaries once it passes
  MAX_CHUNK_CHARS (about 1,000 tokens).
- Tables are never split. A table chunk carries the lines that introduce it,
  because a table without its caption ("Charges per call, by length
  overall:") is hard to interpret.
- Definition sections are split per defined term ("“Act” means ...").
- Every chunk starts with its section breadcrumb, so a chunk retrieved on
  its own still says where it comes from.
- Contents pages are not chunked: the section tree already holds them.
"""

import math
import re
from collections import deque
from dataclasses import dataclass

from app.ingestion.parser import TableBlock, TextLine
from app.ingestion.structure import PlacedElement, Section
from app.models import ChunkKind, SectionKind

MAX_CHUNK_CHARS = 4000
_TABLE_INTRO_LINES = 2
_DEFINED_TERM = re.compile(r"^[“\"‘'][^”\"’']{1,80}[”\"’']\s")
_HYPHENATED = re.compile(r"[^\W\d_]-$")


@dataclass(frozen=True)
class ChunkDraft:
    section_ordinal: int
    kind: ChunkKind
    content: str
    page: int
    printed_page: str | None

    @property
    def token_count(self) -> int:
        # Rough estimate (≈4 characters per token); only used for budgeting.
        return math.ceil(len(self.content) / 4)


def build_chunks(sections: list[Section]) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    for section in sections:
        if section.kind != SectionKind.CONTENTS:
            chunks.extend(_section_chunks(section))
    return chunks


def _section_chunks(section: Section) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    text_kind = ChunkKind.DEFINITION if section.kind == SectionKind.DEFINITIONS else ChunkKind.TEXT
    buffer: list[PlacedElement] = []
    recent: deque[str] = deque(maxlen=_TABLE_INTRO_LINES)

    def flush() -> None:
        if buffer:
            lines = [
                placed.element.text for placed in buffer if isinstance(placed.element, TextLine)
            ]
            body = join_lines(lines)
            chunks.append(_draft(section, text_kind, body, buffer[0]))
            buffer.clear()

    for placed in section.blocks:
        element = placed.element
        if isinstance(element, TableBlock):
            flush()
            intro = "\n".join(recent)
            body = f"{intro}\n\n{element.markdown}" if intro else element.markdown
            chunks.append(_draft(section, ChunkKind.TABLE, body, placed))
            continue
        assert isinstance(element, TextLine)
        starts_term = text_kind == ChunkKind.DEFINITION and _DEFINED_TERM.match(element.text)
        if starts_term or _buffer_chars(buffer) + len(element.text) > MAX_CHUNK_CHARS:
            flush()
        buffer.append(placed)
        recent.append(element.text)
    flush()
    return chunks


def _draft(section: Section, kind: ChunkKind, body: str, first: PlacedElement) -> ChunkDraft:
    return ChunkDraft(
        section_ordinal=section.ordinal,
        kind=kind,
        content=f"{section.path}\n\n{body.strip()}",
        page=first.pdf_page,
        printed_page=first.label,
    )


def _buffer_chars(buffer: list[PlacedElement]) -> int:
    return sum(len(p.element.text) for p in buffer if isinstance(p.element, TextLine))


def join_lines(lines: list[str]) -> str:
    """Join lines, mending words hyphenated across a line break ("de-" +
    "fined" → "defined")."""
    text = ""
    for line in lines:
        line = line.strip()
        if not text:
            text = line
        elif _HYPHENATED.search(text) and line[:1].islower():
            text = text[:-1] + line
        else:
            text += "\n" + line
    return text
