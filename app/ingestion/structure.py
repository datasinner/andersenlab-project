"""Cleaned pages → a tree of sections, each holding its content in order.

Headings are recognised by numbering and typography, never by wording:
- numbered: "2.4 MOORING SERVICES", "5.1.2 HARBOUR DUES" in bold or in a
  font larger than the body;
- keyword: "SECTION 3" / "PART 2" / "CHAPTER 1", with the bold lines that
  follow as its title;
- unnumbered: a short bold line clearly larger than the body.
A heading must be at least body size, so contents-page entries (usually
smaller) and plain numbered lists ("1. Vessel does not ...") are not headings,
and a date ("1 April 2024") never is.

Numbered headings are placed under the nearest open heading whose number is
a prefix of theirs. A number that doesn't fit the current context (a "2.1"
inside section 6.2) is nested under the current section and gets a
disambiguated ref ("6.2.1.3"); its siblings ("1.4") and children ("1.3.1")
follow it there, so refs stay unique within a document.
"""

import re
from dataclasses import dataclass, field

from app.ingestion.parser import Element, LogicalPage, ParsedDocument, TableBlock, TextLine
from app.models import SectionKind

_NUMBERED = re.compile(r"^(\d{1,3}(?:\.\d{1,3})*)\.?\s+(\S.*)$")
_KEYWORD = re.compile(r"^(SECTION|PART|CHAPTER)\s+(\d{1,3})\.?(?:\s+(\S.*))?$", re.IGNORECASE)
_HAS_LETTER = re.compile(r"[^\W\d_]")
_DATE = re.compile(r"^\d{1,2}(?:st|nd|rd|th)?\s+[^\W\d_]{3,9}\.?,?\s+\d{4}$", re.IGNORECASE)
_IMPLICIT_TITLE_CHARS = 80
_PAGE_REFERENCE = re.compile(r"(?<![\d.,])\d{1,3}(?:-\d{1,3})?$")
_DEFINITIONS_TITLE = re.compile(r"\b(definitions?|glossary|interpretation)\b", re.IGNORECASE)
_CONTENTS_TITLE = re.compile(r"\b(contents|index)\b", re.IGNORECASE)

_MAX_HEADING_CHARS = 200
_MAX_UNNUMBERED_HEADING_CHARS = 120
_MAX_CONTINUATION_LINES = 3
_CONTENTS_MIN_ENTRIES = 6
_CONTENTS_MIN_SHARE = 0.3


@dataclass(frozen=True)
class PlacedElement:
    element: Element
    pdf_page: int
    label: str | None


@dataclass
class Section:
    ordinal: int
    ref: str
    title: str  # the heading as printed, e.g. "2.4 MOORING SERVICES"
    level: int
    parent: "Section | None"
    kind: SectionKind
    page_start: int
    number: tuple[int, ...] | None = None
    # For a heading re-based under another section: its number as printed.
    local_number: tuple[int, ...] | None = None
    blocks: list[PlacedElement] = field(default_factory=list)

    @property
    def page_end(self) -> int:
        return self.blocks[-1].pdf_page if self.blocks else self.page_start

    @property
    def path(self) -> str:
        titles = []
        node: Section | None = self
        while node is not None:
            titles.append(node.title)
            node = node.parent
        return " > ".join(reversed(titles))


def build_sections(document: ParsedDocument) -> list[Section]:
    return _Builder(document).build()


class _Builder:
    def __init__(self, document: ParsedDocument) -> None:
        self.body_size = document.body_size
        self.sections: list[Section] = []
        self.stack: list[Section] = []
        self.refs: set[str] = set()
        self.items: list[tuple[PlacedElement, bool]] = []
        for page in document.pages:
            is_contents = _is_contents_page(page)
            for element in page.elements:
                self.items.append((PlacedElement(element, page.pdf_page, page.label), is_contents))
        self.unnumbered_ranks = self._unnumbered_size_ranks()

    def build(self) -> list[Section]:
        index = 0
        while index < len(self.items):
            placed, is_contents = self.items[index]
            element = placed.element
            if is_contents:
                self._contents_section(placed).blocks.append(placed)
                index += 1
                continue
            kind = self._heading_kind(element) if isinstance(element, TextLine) else None
            if kind is None:
                self._current(placed).blocks.append(placed)
                index += 1
                continue
            assert isinstance(element, TextLine)
            title, consumed = self._heading_title(index, kind)
            self._open_heading(kind, element, title, placed.pdf_page)
            index += consumed
        return self.sections

    # -- heading recognition ---------------------------------------------------

    def _heading_kind(self, line: TextLine) -> str | None:
        text = line.text.strip()
        if "…" in text or len(text) > _MAX_HEADING_CHARS or not _HAS_LETTER.search(text):
            return None
        if _DATE.match(text):
            return None
        at_least_body = line.size >= self.body_size - 0.1
        emphasised = line.bold or line.size >= self.body_size + 0.5
        if not (at_least_body and emphasised):
            return None
        if _KEYWORD.match(text):
            return "keyword"
        match = _NUMBERED.match(text)
        if match and _HAS_LETTER.search(match.group(2)):
            return "numbered"
        if (
            line.bold
            and line.size >= self.body_size + 1.5
            and len(text) <= _MAX_UNNUMBERED_HEADING_CHARS
            and not text.endswith(":")
        ):
            return "unnumbered"
        return None

    def _heading_title(self, index: int, kind: str) -> tuple[str, int]:
        """The heading text plus any continuation lines; returns the title
        and how many items it used."""
        first = self.items[index][0]
        heading = first.element
        assert isinstance(heading, TextLine)
        parts = [heading.text.strip()]
        previous = heading
        consumed = 1
        while consumed <= _MAX_CONTINUATION_LINES and index + consumed < len(self.items):
            candidate, is_contents = self.items[index + consumed]
            line = candidate.element
            if (
                is_contents
                or candidate.pdf_page != first.pdf_page
                or not isinstance(line, TextLine)
                or self._heading_kind(line) in ("numbered", "keyword")
                or line.y - previous.y > 2.5 * max(line.size, previous.size)
            ):
                break
            if kind == "keyword":
                continues = line.bold and line.size >= self.body_size - 0.1
            else:
                continues = line.bold == heading.bold and abs(line.size - heading.size) <= 0.3
            if not continues:
                break
            parts.append(line.text.strip())
            previous = line
            consumed += 1
        return " ".join(parts), consumed

    def _unnumbered_size_ranks(self) -> dict[float, int]:
        sizes = {
            placed.element.size
            for placed, is_contents in self.items
            if not is_contents
            and isinstance(placed.element, TextLine)
            and self._heading_kind(placed.element) == "unnumbered"
        }
        return {size: rank for rank, size in enumerate(sorted(sizes, reverse=True), start=1)}

    # -- tree placement --------------------------------------------------------

    def _open_heading(self, kind: str, line: TextLine, title: str, page: int) -> None:
        text = line.text.strip()
        number: tuple[int, ...] | None = None
        local_number: tuple[int, ...] | None = None
        parent: Section | None
        if kind == "keyword":
            match = _KEYWORD.match(text)
            assert match is not None
            number = (int(match.group(2)),)
            ref = match.group(2)
            parent = None
        elif kind == "numbered":
            match = _NUMBERED.match(text)
            assert match is not None
            ref = match.group(1)
            number = tuple(int(part) for part in ref.split("."))
            parent, found = self._numbered_parent(number)
            if not found and len(number) > 1 and self.stack:
                parent = self._rebase_parent()
            if parent is not None and not (parent.number and _extends(parent.number, number)):
                # The number doesn't continue its parent's numbering (a "2.1"
                # inside 6.2, or a child of such a heading): re-base the ref.
                local_number, number = number, None
                if parent.local_number and _extends(parent.local_number, local_number):
                    printed_prefix = ".".join(str(part) for part in parent.local_number)
                    ref = parent.ref + ref[len(printed_prefix) :]
                else:
                    ref = f"{parent.ref}.{ref}"
        else:
            ref = f"s-{len(self.sections) + 1}"
            parent = self._unnumbered_parent(line.size)

        if ref in self.refs:
            ref = f"{ref}~{len(self.sections) + 1}"
        self.refs.add(ref)
        section = Section(
            ordinal=len(self.sections) + 1,
            ref=ref,
            title=title,
            level=parent.level + 1 if parent else 1,
            parent=parent,
            kind=SectionKind.DEFINITIONS if _DEFINITIONS_TITLE.search(title) else SectionKind.BODY,
            page_start=page,
            number=number,
            local_number=local_number,
        )
        if _CONTENTS_TITLE.search(title) and len(title) < 40:
            section.kind = SectionKind.CONTENTS
        self._push(section)

    def _numbered_parent(self, number: tuple[int, ...]) -> tuple[Section | None, bool]:
        """The parent for a numbered heading and whether its numbering placed
        it: under the nearest open heading whose number it extends, or next
        to an open heading it is a sibling of ("4.2" after "4.1")."""
        for open_section in reversed(self.stack):
            printed = open_section.number or open_section.local_number
            if not printed:
                continue
            if _extends(printed, number):
                return open_section, True
            if len(printed) == len(number) and printed[:-1] == number[:-1]:
                return open_section.parent, True
        return None, False

    def _rebase_parent(self) -> Section:
        for open_section in reversed(self.stack):
            if open_section.number:
                return open_section
        return self.stack[-1]

    def _unnumbered_parent(self, size: float) -> Section | None:
        # Inside numbered sections an unnumbered heading is a sub-heading of
        # the current section; otherwise its font size decides its level.
        numbered = [section for section in self.stack if section.number]
        if numbered:
            return self.stack[-1]
        level = self.unnumbered_ranks.get(size, 1)
        for open_section in reversed(self.stack):
            if open_section.level < level:
                return open_section
        return None

    def _push(self, section: Section) -> None:
        if section.parent is None:
            self.stack = []
        else:
            self.stack = self.stack[: self.stack.index(section.parent) + 1]
        self.stack.append(section)
        self.sections.append(section)

    def _current(self, placed: PlacedElement) -> Section:
        if self.stack and self.stack[-1].kind != SectionKind.CONTENTS:
            return self.stack[-1]
        # Content with no heading of its own is titled by its first line.
        first_line = placed.element.text.strip() if isinstance(placed.element, TextLine) else ""
        front = Section(
            ordinal=len(self.sections) + 1,
            ref=f"s-{len(self.sections) + 1}",
            title=first_line[:_IMPLICIT_TITLE_CHARS] or "Front matter",
            level=1,
            parent=None,
            kind=SectionKind.BODY,
            page_start=placed.pdf_page,
        )
        self.refs.add(front.ref)
        self._push(front)
        return front

    def _contents_section(self, placed: PlacedElement) -> Section:
        if self.stack and self.stack[-1].kind == SectionKind.CONTENTS:
            return self.stack[-1]
        contents = Section(
            ordinal=len(self.sections) + 1,
            ref=f"s-{len(self.sections) + 1}",
            title="Contents",
            level=1,
            parent=None,
            kind=SectionKind.CONTENTS,
            page_start=placed.pdf_page,
        )
        self.refs.add(contents.ref)
        self._push(contents)
        return contents


def _extends(prefix: tuple[int, ...], number: tuple[int, ...]) -> bool:
    return len(prefix) < len(number) and number[: len(prefix)] == prefix


def _is_contents_page(page: LogicalPage) -> bool:
    lines = [element.text.strip() for element in page.elements if isinstance(element, TextLine)]
    entries = sum(1 for text in lines if _PAGE_REFERENCE.search(text))
    return entries >= _CONTENTS_MIN_ENTRIES and entries >= len(lines) * _CONTENTS_MIN_SHARE


def table_count(sections: list[Section]) -> int:
    return sum(
        1
        for section in sections
        for placed in section.blocks
        if isinstance(placed.element, TableBlock)
    )
