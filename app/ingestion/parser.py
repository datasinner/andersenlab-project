"""PDF → logical pages of text lines and tables (PyMuPDF).

What this handles, generically:
- "2-up" sheets: a landscape PDF page carrying two printed pages side by side
  is split at its vertical gutter into two logical pages, read left then right;
- reading order: lines are sorted by position, because PDFs often store
  text out of visual order;
- fragments on one baseline (a list number and its text, a contents entry
  and its page number) are joined into one line;
- tables are extracted as markdown, and text inside a table's area is
  dropped from the prose so it isn't read twice.

PyMuPDF is AGPL-licensed; everything else depends on the PdfParser protocol,
so another library can replace it.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import pymupdf

_BOLD_FLAG = 16
_SAME_LINE_TOLERANCE = 2.0  # points between baselines that count as one line
_TWO_UP_MIN_ASPECT = 1.2
_TWO_UP_MAX_CROSSING_SHARE = 0.02
_BULLETS = {"•", "·", "●", "◦", "▪", "‣", "-", "–", "o"}


@dataclass(frozen=True)
class TextLine:
    text: str
    size: float
    bold: bool
    y: float


@dataclass(frozen=True)
class TableBlock:
    markdown: str
    y: float


Element = TextLine | TableBlock


@dataclass
class LogicalPage:
    pdf_page: int  # 1-based
    height: float
    elements: list[Element] = field(default_factory=list)
    label: str | None = None  # the printed page number, when one is found


@dataclass
class ParsedDocument:
    page_count: int
    pages: list[LogicalPage]
    body_size: float  # the most common font size, by characters
    running_text: list[str] = field(default_factory=list)  # removed headers/footers


class PdfParser(Protocol):
    def parse(self, content: bytes) -> ParsedDocument: ...


@dataclass
class _RawLine:
    text: str
    size: float
    bold: bool
    x0: float
    x1: float
    y0: float
    y1: float


class PyMuPdfParser:
    def parse(self, content: bytes) -> ParsedDocument:
        size_chars: Counter[float] = Counter()
        pages: list[LogicalPage] = []
        with pymupdf.open(stream=content, filetype="pdf") as document:
            for index, page in enumerate(document):
                lines = _raw_lines(page)
                for line in lines:
                    size_chars[line.size] += len(line.text)
                tables = [
                    (table.bbox, (table.to_markdown() or "").strip())
                    for table in page.find_tables().tables
                ]
                for x0, x1 in _logical_columns(page.rect, lines):
                    pages.append(_logical_page(index + 1, page.rect.height, lines, tables, x0, x1))
            page_count = document.page_count
        body_size = size_chars.most_common(1)[0][0] if size_chars else 10.0
        return ParsedDocument(page_count=page_count, pages=pages, body_size=body_size)


def _raw_lines(page: pymupdf.Page) -> list[_RawLine]:
    lines: list[_RawLine] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [span for span in line["spans"] if span["text"].strip()]
            if not spans:
                continue
            text = "".join(span["text"] for span in line["spans"]).strip()
            # The line's style is that of its longest span.
            main = max(spans, key=lambda span: len(span["text"].strip()))
            bold = bool(main["flags"] & _BOLD_FLAG) or "bold" in main["font"].lower()
            x0, y0, x1, y1 = line["bbox"]
            lines.append(_RawLine(text, round(main["size"], 1), bold, x0, x1, y0, y1))
    return lines


def _logical_columns(rect: pymupdf.Rect, lines: list[_RawLine]) -> list[tuple[float, float]]:
    """[(x0, x1)] of the logical pages on this sheet: two halves for a 2-up
    landscape sheet whose middle no text crosses, otherwise the whole page."""
    whole = [(rect.x0, rect.x1)]
    if rect.width < rect.height * _TWO_UP_MIN_ASPECT or not lines:
        return whole
    middle = rect.x0 + rect.width / 2
    crossing = sum(1 for line in lines if line.x0 < middle - 2 < line.x1 and line.x1 > middle + 2)
    left = sum(1 for line in lines if line.x1 <= middle + 2)
    right = sum(1 for line in lines if line.x0 >= middle - 2)
    if crossing > len(lines) * _TWO_UP_MAX_CROSSING_SHARE or not left or not right:
        return whole
    return [(rect.x0, middle), (middle, rect.x1)]


def _logical_page(
    pdf_page: int,
    height: float,
    lines: list[_RawLine],
    tables: list[tuple[tuple[float, float, float, float], str]],
    x0: float,
    x1: float,
) -> LogicalPage:
    def in_column(left: float, right: float) -> bool:
        center = (left + right) / 2
        return x0 <= center < x1

    column_tables = [(bbox, md) for bbox, md in tables if md and in_column(bbox[0], bbox[2])]
    column_lines = [
        line
        for line in lines
        if in_column(line.x0, line.x1) and not any(_inside(line, bbox) for bbox, _ in column_tables)
    ]

    elements: list[Element] = []
    for group in _same_baseline_groups(column_lines):
        group.sort(key=lambda line: line.x0)
        main = max(group, key=lambda line: len(line.text))
        text = " ".join(line.text for line in group)
        elements.append(TextLine(text=text, size=main.size, bold=main.bold, y=group[0].y0))
    for bbox, markdown in column_tables:
        elements.append(TableBlock(markdown=markdown, y=bbox[1]))
    elements.sort(key=lambda element: element.y)
    return LogicalPage(pdf_page=pdf_page, height=height, elements=_attach_bullets(elements))


def _inside(line: _RawLine, bbox: tuple[float, float, float, float]) -> bool:
    center_x = (line.x0 + line.x1) / 2
    center_y = (line.y0 + line.y1) / 2
    return bbox[0] <= center_x <= bbox[2] and bbox[1] <= center_y <= bbox[3]


def _same_baseline_groups(lines: list[_RawLine]) -> list[list[_RawLine]]:
    groups: list[list[_RawLine]] = []
    for line in sorted(lines, key=lambda line: (line.y1, line.x0)):
        if groups and abs(groups[-1][0].y1 - line.y1) <= _SAME_LINE_TOLERANCE:
            groups[-1].append(line)
        else:
            groups.append([line])
    return groups


def _attach_bullets(elements: list[Element]) -> list[Element]:
    """A bullet glyph on its own line belongs to the text that follows it."""
    result: list[Element] = []
    pending_bullet: str | None = None
    for element in elements:
        if isinstance(element, TextLine) and element.text in _BULLETS:
            pending_bullet = element.text
            continue
        if pending_bullet and isinstance(element, TextLine):
            element = TextLine(
                f"{pending_bullet} {element.text}", element.size, element.bold, element.y
            )
        pending_bullet = None
        result.append(element)
    return result
