"""Optional vision fallback for tables the text layer scrambles (PARSER_VISION_FALLBACK).

PyMuPDF's table finder reads most tariff tables well, but a table without
ruling lines, or with merged cells, can come out with ragged rows, mostly
empty cells or a single column. When the fallback is on, each logical page
with such a table is rendered to an image and a vision call transcribes the
page's tables to markdown.

The text layer stays the source of truth for grounding: a transcription is
used only if it has as many tables as the parser found on the page and every
number in it also appears in the page's own text. Otherwise, or if the call
fails, the page is kept as parsed. This never invents a number a rule could
later cite.
"""

import asyncio
import base64

import pymupdf
import structlog
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.domain.numbers import extract_numbers
from app.ingestion.parser import LogicalPage, ParsedDocument, TableBlock, TextLine
from app.llm.client import LLMClient
from app.llm.prompts import PROMPTS
from app.llm.resilience import LLMError

logger = structlog.get_logger("app.ingestion")

RENDER_DPI = 150
_EMPTY_CELL_SHARE = 0.5


class TableTranscription(BaseModel):
    tables: list[str]


def suspect_table(markdown: str) -> bool:
    """Whether a table looks scrambled: rows of different widths, a single
    column, or more than half of its cells empty."""
    rows = [_cells(line) for line in markdown.splitlines() if line.strip().startswith("|")]
    rows = [row for row in rows if not all("-" in cell and set(cell) <= set("-: ") for cell in row)]
    if not rows:
        return False
    widths = {len(row) for row in rows}
    if len(widths) > 1 or max(widths) < 2:
        return True
    cells = [cell for row in rows for cell in row]
    return sum(1 for cell in cells if not cell) > len(cells) * _EMPTY_CELL_SHARE


def accept_transcription(page: LogicalPage, tables: list[str]) -> bool:
    """Whether a transcription may replace the page's tables: one table for
    each the parser found, and no number that isn't in the page's text."""
    found = [element for element in page.elements if isinstance(element, TableBlock)]
    if len(tables) != len(found):
        return False
    page_text = "\n".join(
        element.text if isinstance(element, TextLine) else element.markdown
        for element in page.elements
    )
    return extract_numbers("\n".join(tables)) <= extract_numbers(page_text)


async def transcribe_suspect_tables(
    llm: LLMClient, content: bytes, document: ParsedDocument
) -> int:
    """Replace the tables of pages with a suspect table by a vision
    transcription, where it passes accept_transcription. Returns the number
    of pages changed."""
    pages = [
        page
        for page in document.pages
        if any(
            isinstance(element, TableBlock) and suspect_table(element.markdown)
            for element in page.elements
        )
    ]
    if not pages:
        return 0
    images = await asyncio.to_thread(_render, content, pages)
    results = await asyncio.gather(
        *(_transcribe(llm, page, image) for page, image in zip(pages, images, strict=True))
    )
    return sum(results)


async def _transcribe(llm: LLMClient, page: LogicalPage, image: bytes) -> bool:
    found = [element for element in page.elements if isinstance(element, TableBlock)]
    system, user = PROMPTS["transcribe_tables"].render(
        page=page.label or str(page.pdf_page), table_count=str(len(found))
    )
    encoded = base64.b64encode(image).decode()
    message = HumanMessage(
        content=[
            {"type": "text", "text": user.content},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
        ]
    )
    log = logger.bind(pdf_page=page.pdf_page)
    try:
        result = await llm.generate_structured(
            TableTranscription, [system, message], name="transcribe_tables"
        )
    except LLMError as exc:
        log.warning("table_transcription_failed", error=str(exc))
        return False
    tables = [table.strip() for table in result.value.tables if table.strip()]
    if not accept_transcription(page, tables):
        log.info("table_transcription_rejected", tables=len(tables), parsed=len(found))
        return False
    replacements = iter(tables)
    page.elements = [
        TableBlock(markdown=next(replacements), y=element.y)
        if isinstance(element, TableBlock)
        else element
        for element in page.elements
    ]
    log.info("table_transcription_used", tables=len(tables))
    return True


def _render(content: bytes, pages: list[LogicalPage]) -> list[bytes]:
    """PNG images of the logical pages (the right half of a 2-up sheet, etc.)."""
    images = []
    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        for page in pages:
            sheet = pdf[page.pdf_page - 1]
            clip = sheet.rect
            if page.x0 is not None and page.x1 is not None:
                clip = pymupdf.Rect(page.x0, sheet.rect.y0, page.x1, sheet.rect.y1)
            images.append(sheet.get_pixmap(clip=clip, dpi=RENDER_DPI).tobytes("png"))
    return images


def _cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]
