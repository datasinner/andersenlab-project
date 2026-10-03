import pytest

from app.domain.ports import match_port, port_words

PORTS = [
    {"name": "Richards Bay", "aliases": ["Port of Richards Bay"]},
    {"name": "Durban", "aliases": ["Port of Durban"]},
    {"name": "Port Elizabeth / Ngqura", "aliases": ["Port of Port Elizabeth"]},
    {"name": "Saldanha", "aliases": []},
    {"name": "East London", "aliases": []},
    {"name": "London Gateway", "aliases": []},
]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Durban", "Durban"),
        ("port of durban", "Durban"),
        ("DURBAN harbour", "Durban"),
        ("Ngqura", "Port Elizabeth / Ngqura"),  # one word of a combined name
        ("Port Elizabeth", "Port Elizabeth / Ngqura"),
        ("Saldanha Bay", "Saldanha"),  # the request says more than the name
        ("Richards Bay", "Richards Bay"),
        ("East London", "East London"),  # exact beats the partial "London Gateway"
    ],
)
def test_matching_names(query, expected):
    assert match_port(query, PORTS) == expected


@pytest.mark.parametrize("query", ["Amsterdam", "", "Port of", "London"])
def test_unknown_generic_or_ambiguous_names_do_not_match(query):
    assert match_port(query, PORTS) is None


def test_port_words_ignore_generic_words_and_punctuation():
    assert port_words("Port of Port Elizabeth / Ngqura") == {"elizabeth", "ngqura"}
