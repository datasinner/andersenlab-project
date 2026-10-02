"""Remove page furniture from parsed pages.

- Running headers and footers: a line repeated on at least half of the
  logical pages (ignoring a page number at either end of it, as in
  "Annual Tariffs 2024    15") is removed everywhere; one copy of each is kept in
  ParsedDocument.running_text, because it often states facts the document
  profile needs ("Tariffs subject to VAT at 15%").
- Printed page numbers: a bare number in the top or bottom margin, or the
  number attached to a running header/footer, becomes the logical page's
  label instead of text.
- Dot leaders ("Minimum fee………235.52") are shortened to a single ellipsis.
"""

import re
from collections import Counter

from app.ingestion.parser import LogicalPage, ParsedDocument, TextLine

_MIN_RUNNING_PAGES = 3
_RUNNING_SHARE = 0.5
_MARGIN_SHARE = 0.12
_PAGE_LABEL = re.compile(r"^\d{1,4}$")
_TRAILING_NUMBER = re.compile(r"^(.*[^\W\d_].*?)\s+(\d{1,4})$")
_LEADING_NUMBER = re.compile(r"^(\d{1,4})\s+(.*[^\W\d_].*)$")
_DASHES = re.compile(r"\s*[—–-]\s*")
_SPACES = re.compile(r"\s+")
_LEADER = re.compile(r"\s*(?:[.…]\s*){3,}\s*")


def clean(document: ParsedDocument) -> ParsedDocument:
    running = _running_lines(document.pages)
    kept_running: dict[str, str] = {}
    for page in document.pages:
        cleaned = []
        for element in page.elements:
            if isinstance(element, TextLine):
                key, number = _running_key(element, page)
                if key in running:
                    kept_running.setdefault(key, _without_page_number(element.text))
                    if number and _in_margin(element, page):
                        page.label = page.label or number
                    continue
                if _is_page_label(element, page):
                    page.label = page.label or element.text.strip()
                    continue
                element = TextLine(
                    _LEADER.sub(" … ", element.text).strip(),
                    element.size,
                    element.bold,
                    element.y,
                )
            cleaned.append(element)
        page.elements = cleaned
    document.running_text = list(kept_running.values())
    return document


def _normalize(text: str) -> str:
    return _SPACES.sub(" ", _DASHES.sub("-", text)).strip().casefold()


def _running_key(line: TextLine, page: LogicalPage) -> tuple[str, str | None]:
    """How a line is compared across pages. In the margins a page number at
    either end is set aside (and returned); body lines compare as they are."""
    normalized = _normalize(line.text)
    if not _in_margin(line, page):
        return normalized, None
    trailing = _TRAILING_NUMBER.match(normalized)
    if trailing:
        return trailing.group(1), trailing.group(2)
    leading = _LEADING_NUMBER.match(normalized)
    if leading:
        return leading.group(2), leading.group(1)
    return normalized, None


def _without_page_number(text: str) -> str:
    text = text.strip()
    for pattern, keep in ((_TRAILING_NUMBER, 1), (_LEADING_NUMBER, 2)):
        match = pattern.match(text)
        if match:
            return match.group(keep)
    return text


def _running_lines(pages: list[LogicalPage]) -> set[str]:
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({_running_key(e, page)[0] for e in page.elements if isinstance(e, TextLine)})
    threshold = max(_MIN_RUNNING_PAGES, len(pages) * _RUNNING_SHARE)
    return {
        text for text, count in counts.items() if count >= threshold and not _PAGE_LABEL.match(text)
    }


def _in_margin(line: TextLine, page: LogicalPage) -> bool:
    return line.y < page.height * _MARGIN_SHARE or line.y > page.height * (1 - _MARGIN_SHARE)


def _is_page_label(line: TextLine, page: LogicalPage) -> bool:
    return _in_margin(line, page) and bool(_PAGE_LABEL.match(line.text.strip()))
