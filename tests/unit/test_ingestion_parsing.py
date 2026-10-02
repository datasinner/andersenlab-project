from app.ingestion.cleaner import clean
from app.ingestion.parser import PyMuPdfParser, TextLine
from tests.unit.ingestion_builders import document, line, page, pdf_bytes


def _texts(logical_page) -> list[str]:
    return [element.text for element in logical_page.elements if isinstance(element, TextLine)]


# -- parser --------------------------------------------------------------------


def test_two_up_landscape_sheet_is_read_as_two_pages_left_then_right():
    runs = [
        (450, 100, "Right page first line", 9, False),
        (50, 120, "Left page second line", 9, False),
        (50, 100, "Left page first line", 9, False),
        (450, 120, "Right page second line", 9, False),
    ]
    parsed = PyMuPdfParser().parse(pdf_bytes([runs], landscape=True))
    assert parsed.page_count == 1
    assert [_texts(p) for p in parsed.pages] == [
        ["Left page first line", "Left page second line"],
        ["Right page first line", "Right page second line"],
    ]


def test_landscape_page_with_text_across_the_middle_stays_whole():
    runs = [(50, 100, "A line long enough to run straight across the middle " * 2, 9, False)]
    parsed = PyMuPdfParser().parse(pdf_bytes([runs], landscape=True))
    assert len(parsed.pages) == 1


def test_fragments_on_one_baseline_are_joined_and_styles_kept():
    runs = [
        (50, 100, "3.6 TUGS AND TOWAGE", 10, True),
        (50, 130, "1.", 9, False),
        (80, 130, "Vessel does not leave the port", 9, False),
        (50, 160, "Body text", 9, False),
    ]
    parsed = PyMuPdfParser().parse(pdf_bytes([runs], landscape=False))
    (only,) = parsed.pages
    first = only.elements[0]
    assert isinstance(first, TextLine) and first.bold and first.size == 10
    assert _texts(only) == ["3.6 TUGS AND TOWAGE", "1. Vessel does not leave the port", "Body text"]
    assert parsed.body_size == 9


def test_lone_bullet_glyph_joins_the_following_line():
    runs = [(50, 100, "•", 12, False), (70, 104, "Bona fide coasters;", 9, False)]
    parsed = PyMuPdfParser().parse(pdf_bytes([runs], landscape=False))
    # The base-14 font renders the bullet as "·"; either glyph is a bullet.
    (text,) = _texts(parsed.pages[0])
    assert text[0] in "•·" and text[1:] == " Bona fide coasters;"


# -- cleaner -----------------------------------------------------------------------


def _page_with_furniture(number: int, footer_dash: str) -> object:
    return page(
        line("Fees exclude VAT at 10%", 20),
        line(f"Fee for service {number}", 300),
        line(f"Harbour Tariff 2025{footer_dash}2026 {number + 10}", 590),
        number=number,
    )


def test_running_headers_and_footers_are_removed_and_kept_once():
    pages = [_page_with_furniture(n, "—" if n % 2 else " - ") for n in range(1, 5)]
    cleaned = clean(document(*pages))

    assert [_texts(p) for p in cleaned.pages] == [[f"Fee for service {n}"] for n in range(1, 5)]
    assert cleaned.running_text == ["Fees exclude VAT at 10%", "Harbour Tariff 2025—2026"]


def test_page_numbers_become_labels():
    pages = [page(line("Body", 300), line(str(n + 4), 595), number=n) for n in range(1, 3)] + [
        _page_with_furniture(n, "-") for n in range(3, 7)
    ]
    cleaned = clean(document(*pages))
    assert [p.label for p in cleaned.pages] == ["5", "6", "13", "14", "15", "16"]
    assert _texts(cleaned.pages[0]) == ["Body"]


def test_bare_numbers_in_the_body_are_kept():
    cleaned = clean(document(page(line("Plus", 290), line("250", 300))))
    assert _texts(cleaned.pages[0]) == ["Plus", "250"]


def test_dot_leaders_are_shortened():
    cleaned = clean(
        document(page(line("Minimum fee…………………235.52", 300), line("Fee ...... 12", 310)))
    )
    assert _texts(cleaned.pages[0]) == ["Minimum fee … 235.52", "Fee … 12"]
