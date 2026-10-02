from decimal import Decimal

import pytest

from app.domain.numbers import (
    NumberFormatError,
    extract_numbers,
    format_money,
    format_number,
    parse_number,
    round_money,
)


@pytest.mark.parametrize(
    ("printed", "expected"),
    [
        ("117.08", "117.08"),
        ("30 960.46", "30960.46"),
        ("2  801.91", "2801.91"),  # doubled space from PDF extraction
        ("8 970.00", "8970.00"),  # no-break space
        ("1 654.56", "1654.56"),  # narrow no-break space
        ("1,654.56", "1654.56"),
        ("1,000,000", "1000000"),
        ("0,65", "0.65"),  # decimal comma
        ("1.234,56", "1234.56"),
        ("1.234.567", "1234567"),
        ("35%", "35"),
        (" 0.54 ", "0.54"),
        ("-12.5", "-12.5"),
    ],
)
def test_parse_number_reads_printed_formats(printed, expected):
    assert parse_number(printed) == Decimal(expected)


def test_parse_number_passes_through_numbers():
    assert parse_number(Decimal("1.5")) == Decimal("1.5")
    assert parse_number(7) == Decimal(7)


@pytest.mark.parametrize("bad", ["", "abc", "12 tons", "NaN", "Infinity", True])
def test_parse_number_rejects_non_numbers(bad):
    with pytest.raises(NumberFormatError):
        parse_number(bad)


def test_extract_numbers_reads_dotted_leaders_and_grouped_thousands():
    text = "Per 100 tons or part thereof…………117.08\nBasic Fee 30 960.46 and 2  801.91"
    numbers = extract_numbers(text)
    assert {Decimal("100"), Decimal("117.08"), Decimal("30960.46"), Decimal("2801.91")} <= numbers


def test_extract_numbers_keeps_every_reading_of_ambiguous_spacing():
    numbers = extract_numbers("Up to 2 000 0.50")
    assert {Decimal("2000"), Decimal("2"), Decimal("0.50")} <= numbers


def test_extract_numbers_reads_markdown_table_cells_and_percentages():
    numbers = extract_numbers("|50 001 to 100 000<br>Plus|73 118.07<br>32.24| a surcharge of 25%")
    assert {
        Decimal("50001"),
        Decimal("100000"),
        Decimal("73118.07"),
        Decimal("32.24"),
        Decimal("25"),
    } <= numbers


def test_extract_numbers_does_not_join_numbers_across_lines():
    numbers = extract_numbers("2 000\n500")
    assert Decimal("2000500") not in numbers
    assert {Decimal("2000"), Decimal("500")} <= numbers


def test_extract_numbers_skips_runs_that_are_not_numbers():
    numbers = extract_numbers("clause 1,2.3,4 and 1.5 000")
    assert Decimal("1.5") in numbers
    assert Decimal("1500") not in numbers  # "1.5" + "000" are not thousands groups


def test_round_money_rounds_half_up():
    assert round_money(Decimal("199371.3453")) == Decimal("199371.35")
    assert round_money(Decimal("0.125")) == Decimal("0.13")
    assert round_money(Decimal("2.675")) == Decimal("2.68")


def test_format_number_and_money():
    assert format_number(Decimal("51300")) == "51,300"
    assert format_number(Decimal("5.13E+4")) == "51,300"
    assert format_number(Decimal("3.390")) == "3.39"
    assert format_number(Decimal("100500.8553")) == "100,500.8553"
    assert format_money(Decimal("199371.3453")) == "199,371.35"
