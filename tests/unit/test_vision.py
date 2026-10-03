import pytest

from app.ingestion.parser import TableBlock
from app.ingestion.vision import accept_transcription, suspect_table, transcribe_suspect_tables
from app.llm.client import FakeLLMClient
from tests.unit.ingestion_builders import document, line, page, pdf_bytes, table

CLEAN = "|Length|Charge|\n|---|---|\n|Up to 100 m|1,200.00|\n|Over 100 m|2,450.00|"
RAGGED = "|Length|Charge|\n|---|---|\n|Up to 100 m 1,200.00|\n|Over 100 m|2,450.00|"
PAGE_TEXT = "Pilotage per operation: Up to 100 m 1,200.00; Over 100 m 2,450.00"


@pytest.mark.parametrize(
    ("markdown", "suspect"),
    [
        (CLEAN, False),
        (RAGGED, True),
        ("|Rates|\n|---|\n|1,200.00|\n|2,450.00|", True),  # one column
        ("|A|B|C|\n|---|---|---|\n|x|||\n||||\n|||y|", True),  # mostly empty
    ],
)
def test_scrambled_tables_are_suspect(markdown, suspect):
    assert suspect_table(markdown) is suspect


def test_a_transcription_must_match_the_table_count_and_the_page_numbers():
    scrambled = page(line(PAGE_TEXT, 50), table(RAGGED, 80))
    assert accept_transcription(scrambled, [CLEAN])
    assert not accept_transcription(scrambled, [CLEAN, CLEAN])
    assert not accept_transcription(scrambled, [CLEAN.replace("2,450.00", "2,540.00")])


def _scrambled_document():
    return document(page(line(PAGE_TEXT, 50), table(RAGGED, 80)))


def _one_page_pdf() -> bytes:
    return pdf_bytes([[(50, 100, PAGE_TEXT, 9, False)]], landscape=False)


async def test_suspect_tables_are_replaced_by_an_accepted_transcription():
    parsed = _scrambled_document()
    llm = FakeLLMClient()
    llm.script("transcribe_tables", {"tables": [CLEAN]})

    changed = await transcribe_suspect_tables(llm, _one_page_pdf(), parsed)

    assert changed == 1
    tables = [e for e in parsed.pages[0].elements if isinstance(e, TableBlock)]
    assert [t.markdown for t in tables] == [CLEAN]
    content = llm.calls[0].messages[1].content
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


async def test_a_transcription_with_a_number_not_on_the_page_is_ignored():
    parsed = _scrambled_document()
    llm = FakeLLMClient()
    llm.script("transcribe_tables", {"tables": [CLEAN.replace("1,200.00", "1,250.00")]})

    assert await transcribe_suspect_tables(llm, _one_page_pdf(), parsed) == 0
    tables = [e for e in parsed.pages[0].elements if isinstance(e, TableBlock)]
    assert [t.markdown for t in tables] == [RAGGED]


async def test_pages_with_clean_tables_are_not_sent():
    parsed = document(page(line(PAGE_TEXT, 50), table(CLEAN, 80)))
    llm = FakeLLMClient()
    assert await transcribe_suspect_tables(llm, _one_page_pdf(), parsed) == 0
    assert llm.calls == []
