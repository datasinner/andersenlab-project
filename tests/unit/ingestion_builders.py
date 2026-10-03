"""Small hand-built parsed documents and generated PDFs for ingestion tests."""

import pymupdf

from app.ingestion.parser import LogicalPage, ParsedDocument, TableBlock, TextLine

BODY = 9.0


def line(text: str, y: float, size: float = BODY, bold: bool = False) -> TextLine:
    return TextLine(text=text, size=size, bold=bold, y=y)


def heading(text: str, y: float, size: float = 10.0) -> TextLine:
    return TextLine(text=text, size=size, bold=True, y=y)


def table(markdown: str, y: float) -> TableBlock:
    return TableBlock(markdown=markdown, y=y)


def page(*elements, number: int = 1, label: str | None = None) -> LogicalPage:
    return LogicalPage(pdf_page=number, height=600.0, elements=list(elements), label=label)


def document(*pages: LogicalPage) -> ParsedDocument:
    return ParsedDocument(page_count=len(pages), pages=list(pages), body_size=BODY)


def pdf_bytes(pages: list[list[tuple[float, float, str, float, bool]]], landscape: bool) -> bytes:
    """A PDF whose pages hold (x, y, text, size, bold) text runs."""
    width, height = (842, 595) if landscape else (595, 842)
    pdf = pymupdf.open()
    for runs in pages:
        sheet = pdf.new_page(width=width, height=height)
        for x, y, text, size, bold in runs:
            sheet.insert_text((x, y), text, fontsize=size, fontname="hebo" if bold else "helv")
    data = pdf.tobytes()
    pdf.close()
    return data
