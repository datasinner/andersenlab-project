"""The research agent's tools. All are read-only and scoped to one document.

Each tool is a Pydantic model: its docstring and fields are the tool
description the model sees, and the model's arguments are validated against
it. Every excerpt a tool shows is remembered by chunk id, so the evidence the
agent submits can be resolved back to exact text for extraction and
grounding. An excerpt already shown in the conversation is shown again only
as a reference to it: the whole conversation is resent on every turn, so a
repeated table would be paid for on every later turn.
"""

import uuid
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.state import Excerpt
from app.models import Chunk, DocumentSection
from app.retrieval.context import find_definitions, outline
from app.retrieval.search import TariffSearch

_MAX_OUTLINE_LINES = 200


class SearchTariff(BaseModel):
    """Search the tariff document. Returns the best-matching excerpts with their chunk ids,
    sections and pages."""

    query: str = Field(description="What to look for, in the document's own words where known.")


class ReadSection(BaseModel):
    """Read one whole section of the tariff by its ref (e.g. "7.2"), including its tables,
    and list its subsections."""

    ref: str


class LookupDefinition(BaseModel):
    """Look up how the tariff's definitions section defines a term (e.g. "tonnage")."""

    term: str


class ListSections(BaseModel):
    """List the document's sections (ref and title), optionally only those whose ref
    starts with a prefix (e.g. "3.")."""

    prefix: str | None = None


class SubmitEvidence(BaseModel):
    """Finish research. List the chunk ids of every excerpt needed to price the charge,
    and notes on what you found (e.g. which rate column applies at this port and why)."""

    chunk_ids: list[int]
    notes: str


RESEARCH_TOOLS: list[type[BaseModel]] = [
    SearchTariff,
    ReadSection,
    LookupDefinition,
    ListSections,
    SubmitEvidence,
]


class ToolExecutor:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        search: TariffSearch,
        document_id: uuid.UUID,
    ) -> None:
        self._sessions = session_factory
        self._search = search
        self.document_id = document_id
        self.seen: dict[int, Excerpt] = {}
        self.sections_read: list[str] = []
        self._shown: set[int] = set()

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool call and return its result as text for the model.
        Bad arguments and unknown tools come back as text too, so the model
        can correct itself."""
        try:
            if name == "SearchTariff":
                return await self.search(SearchTariff.model_validate(arguments).query)
            if name == "ReadSection":
                return await self.read_section(ReadSection.model_validate(arguments).ref)
            if name == "LookupDefinition":
                return await self.lookup_definition(LookupDefinition.model_validate(arguments).term)
            if name == "ListSections":
                return await self.list_sections(ListSections.model_validate(arguments).prefix)
        except ValidationError as exc:
            return f"Invalid arguments for {name}: {exc.errors()[0]['msg']}"
        return f"Unknown tool {name}."

    async def search(self, query: str) -> str:
        hits = await self._search.search(self.document_id, query)
        excerpts = [
            self._remember(hit.chunk_id, hit.section_ref, hit.page, hit.printed_page, hit.content)
            for hit in hits
        ]
        return f"Results for {query!r}:\n\n{self._show(excerpts)}"

    async def read_section(self, ref: str) -> str:
        async with self._sessions() as session:
            section = await session.scalar(
                select(DocumentSection).where(
                    DocumentSection.document_id == self.document_id, DocumentSection.ref == ref
                )
            )
            if section is None:
                return f"No section with ref {ref!r}. Use ListSections to see the refs."
            chunks = list(
                await session.scalars(
                    select(Chunk).where(Chunk.section_id == section.id).order_by(Chunk.ordinal)
                )
            )
            children = list(
                await session.scalars(
                    select(DocumentSection)
                    .where(DocumentSection.parent_id == section.id)
                    .order_by(DocumentSection.ordinal)
                )
            )
        if ref not in self.sections_read:
            self.sections_read.append(ref)
        excerpts = [
            self._remember(chunk.id, section.ref, chunk.page, chunk.printed_page, chunk.content)
            for chunk in chunks
        ]
        pages = f"pages {section.page_start}-{section.page_end}"
        lines = [f"Section {section.ref}: {section.title} ({pages})"]
        lines.append(self._show(excerpts) if excerpts else "(no text of its own)")
        if children:
            lines.append("Subsections: " + "; ".join(f"{c.ref} {c.title}" for c in children))
        return "\n\n".join(lines)

    async def lookup_definition(self, term: str) -> str:
        async with self._sessions() as session:
            chunks = await find_definitions(session, self.document_id, term)
            refs = await _section_refs(session, {chunk.section_id for chunk in chunks})
        if not chunks:
            return f"No definition mentioning {term!r} was found."
        excerpts = [
            self._remember(
                chunk.id, refs[chunk.section_id], chunk.page, chunk.printed_page, chunk.content
            )
            for chunk in chunks
        ]
        return self._show(excerpts)

    async def list_sections(self, prefix: str | None) -> str:
        async with self._sessions() as session:
            sections = await outline(session, self.document_id, prefix)
        lines = [f"{s.ref} | {s.title} (p{s.page_start})" for s in sections[:_MAX_OUTLINE_LINES]]
        if len(sections) > _MAX_OUTLINE_LINES:
            lines.append(f"... {len(sections) - _MAX_OUTLINE_LINES} more; narrow with a prefix")
        return "\n".join(lines) or "No sections match."

    def _show(self, excerpts: list[Excerpt]) -> str:
        rendered = []
        for excerpt in excerpts:
            if excerpt.chunk_id in self._shown:
                rendered.append(excerpt.render_reference())
            else:
                rendered.append(excerpt.render())
                self._shown.add(excerpt.chunk_id)
        return "\n\n".join(rendered) or "(none)"

    def _remember(
        self, chunk_id: int, section_ref: str, page: int, printed_page: str | None, text: str
    ) -> Excerpt:
        excerpt = Excerpt(chunk_id, section_ref, page, printed_page, text)
        self.seen[chunk_id] = excerpt
        return excerpt


async def _section_refs(session: AsyncSession, section_ids: set[int]) -> dict[int, str]:
    if not section_ids:
        return {}
    rows = await session.execute(
        select(DocumentSection.id, DocumentSection.ref).where(DocumentSection.id.in_(section_ids))
    )
    return {section_id: ref for section_id, ref in rows}
