from app.ingestion.catalogue import CatalogueEntry, _clean_entries, discover_charges, outline_text
from app.ingestion.structure import build_sections
from app.llm.client import FakeLLMClient
from tests.unit.ingestion_builders import document, heading, line, page, table


def _entry(charge_id: str, *refs: str) -> CatalogueEntry:
    return CatalogueEntry(
        charge_id=charge_id,
        name=charge_id.replace("_", " "),
        section_refs=list(refs),
        payer="vessel",
        trigger="per_service",
        description="A charge.",
    )


def _sections():
    return build_sections(
        document(
            page(
                heading("4.1 PORT FEES", 50),
                line("Fees on vessels entering the port.", 60),
                heading("4.1.1 PORT DUES", 70),
                table("|Basis|Fee|\n|---|---|\n|per 100 tons|10|", 80),
                heading("4.11 ADMINISTRATIVE FEES", 120),
                line("4.11.1 Amending fee … 459.37", 130),
            )
        )
    )


def test_outline_lists_refs_titles_and_opening_text():
    assert outline_text(_sections()) == (
        "4.1 | 4.1 PORT FEES | Fees on vessels entering the port.\n"
        "  4.1.1 | 4.1.1 PORT DUES | |Basis|Fee|\n"
        "4.11 | 4.11 ADMINISTRATIVE FEES | 4.11.1 Amending fee … 459.37"
    )


def test_refs_resolve_to_the_nearest_existing_section():
    known = {"4.1", "4.1.1", "4.11"}
    cleaned = _clean_entries(
        [
            _entry("amending_fee", "4.11.1"),  # an item, not a section → 4.11
            _entry("port_dues", "4.1.1", "4.1.1.", "9.9"),  # duplicate and unknown refs
            _entry("phantom", "12.3"),  # nothing left → dropped
        ],
        known,
    )
    assert [(entry.charge_id, entry.section_refs) for entry in cleaned] == [
        ("amending_fee", ["4.11"]),
        ("port_dues", ["4.1.1"]),
    ]


def test_duplicate_charge_ids_are_suffixed():
    cleaned = _clean_entries(
        [_entry("fee", "4.1"), _entry("fee", "4.1.1"), _entry("fee", "4.11")],
        {"4.1", "4.1.1", "4.11"},
    )
    assert [entry.charge_id for entry in cleaned] == ["fee", "fee_2", "fee_3"]


async def test_discover_charges_sends_the_outline_and_cleans_the_answer():
    llm = FakeLLMClient()
    llm.script(
        "charge_catalogue",
        {"charges": [_entry("port_dues", "4.1.1").model_dump(), _entry("x", "9.9").model_dump()]},
    )
    charges, usage = await discover_charges(llm, _sections())

    assert [charge.charge_id for charge in charges] == ["port_dues"]
    assert usage.prompt_tokens == FakeLLMClient.usage.prompt_tokens
    assert "4.1.1 | 4.1.1 PORT DUES" in llm.calls[0].messages[1].content
