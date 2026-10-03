from app.ingestion.chunker import MAX_CHUNK_CHARS, build_chunks, join_lines
from app.ingestion.structure import build_sections, table_count
from app.models import ChunkKind, SectionKind
from tests.unit.ingestion_builders import document, heading, line, page, table


def _outline(sections) -> list[tuple[str, int, str | None]]:
    return [
        (section.ref, section.level, section.parent.ref if section.parent else None)
        for section in sections
    ]


def test_numbered_and_keyword_headings_form_a_tree():
    sections = build_sections(
        document(
            page(
                heading("SECTION 2", 50, size=11),
                heading("MARINE", 62, size=9.0),
                heading("SERVICES", 74, size=9.0),
                heading("2.1 PILOTAGE", 100),
                line("Pilotage is compulsory.", 110),
                heading("2.1.1 PILOTAGE FEES", 130),
                line("Per service.", 140),
                heading("2.2 TOWAGE", 160),
                line("Per tug.", 170),
            )
        )
    )
    assert _outline(sections) == [
        ("2", 1, None),
        ("2.1", 2, "2"),
        ("2.1.1", 3, "2.1"),
        ("2.2", 2, "2"),
    ]
    assert sections[0].title == "SECTION 2 MARINE SERVICES"
    assert sections[2].path == "SECTION 2 MARINE SERVICES > 2.1 PILOTAGE > 2.1.1 PILOTAGE FEES"


def test_multi_line_numbered_heading_keeps_its_continuation():
    sections = build_sections(
        document(
            page(
                heading("5.1 FEES PAYABLE FOR LICENCES ISSUED BY", 100, size=11),
                heading("THE AUTHORITY", 114, size=11),
                line("Licence fees are payable yearly.", 140),
            )
        )
    )
    assert sections[0].title == "5.1 FEES PAYABLE FOR LICENCES ISSUED BY THE AUTHORITY"
    assert len(sections[0].blocks) == 1


def test_lists_dates_and_small_entries_are_not_headings():
    sections = build_sections(
        document(
            page(
                heading("1.1 LIGHT DUES", 50),
                line("1. Vessel does not leave the coast.", 60),  # plain list item
                heading("1 April 2024", 70),  # bold date
                line("2.1 VTS charges", 80, size=8.5, bold=True),  # smaller than body
                heading("3.1 Basic fee … 117.08", 90),  # a priced line, not a heading
            )
        )
    )
    assert [section.ref for section in sections] == ["1.1"]
    assert len(sections[0].blocks) == 4


def test_numbering_that_does_not_fit_is_rebased_under_the_current_section():
    sections = build_sections(
        document(
            page(
                heading("5.2 PERMITS", 50),
                heading("2.1 Port rule licences", 60, size=9.0),
                heading("2.1.1 Bunkering", 70, size=9.0),
                heading("2.2 Registrations", 80, size=9.0),
                heading("5.3 HULL CLEANING", 90),
            )
        )
    )
    assert _outline(sections) == [
        ("5.2", 1, None),
        ("5.2.2.1", 2, "5.2"),
        ("5.2.2.1.1", 3, "5.2.2.1"),
        ("5.2.2.2", 2, "5.2"),
        ("5.3", 1, None),
    ]


def test_unnumbered_headings_are_levelled_by_font_size():
    sections = build_sections(
        document(
            page(
                heading("Harbour Dues", 50, size=14),
                heading("Vessels", 60, size=12),
                line("Charged per call.", 70),
                heading("Cargo", 80, size=12),
                heading("Marine Services", 90, size=14),
            )
        )
    )
    assert _outline(sections) == [
        ("s-1", 1, None),
        ("s-2", 2, "s-1"),
        ("s-3", 2, "s-1"),
        ("s-4", 1, None),
    ]


def test_unnumbered_heading_inside_numbered_section_is_a_subsection():
    sections = build_sections(
        document(page(heading("4.2 SMALL VESSELS", 50), heading("Visiting yachts", 60, size=11)))
    )
    assert _outline(sections) == [("4.2", 1, None), ("s-2", 2, "4.2")]


def test_contents_definitions_and_untitled_content():
    contents = page(
        *[line(f"{n}.1 Charge number {n} {n + 4}", 50 + 10 * n) for n in range(1, 8)], number=1
    )
    body = page(
        line("Schedule of fees", 40),
        heading("Definitions", 60, size=11),
        line("“Act” means the Ports Act.", 70),
        line("“Agent” means a vessel's agent.", 80),
        number=2,
    )
    sections = build_sections(document(contents, body))
    assert [(s.title, s.kind) for s in sections] == [
        ("Contents", SectionKind.CONTENTS),
        ("Schedule of fees", SectionKind.BODY),
        ("Definitions", SectionKind.DEFINITIONS),
    ]


def test_duplicate_refs_are_made_unique():
    sections = build_sections(
        document(
            page(heading("3.1 PILOTAGE", 50), heading("3 GENERAL", 60), heading("3.1 RULES", 70))
        )
    )
    assert len({section.ref for section in sections}) == len(sections)


# -- chunker -------------------------------------------------------------------------


def test_table_chunk_carries_its_introduction_and_breadcrumb():
    sections = build_sections(
        document(
            page(
                heading("3.6 TUGS", 50),
                line("Fees are payable per service", 60),
                line("based on the vessel's tonnage:", 70),
                table("|Tonnage|Fee|\n|---|---|\n|Up to 2 000|100|", 80),
                line("A surcharge applies at night.", 120),
                label="15",
            )
        )
    )
    chunks = build_chunks(sections)
    assert table_count(sections) == 1
    assert [chunk.kind for chunk in chunks] == [ChunkKind.TEXT, ChunkKind.TABLE, ChunkKind.TEXT]
    assert chunks[1].content == (
        "3.6 TUGS\n\nFees are payable per service\nbased on the vessel's tonnage:\n\n"
        "|Tonnage|Fee|\n|---|---|\n|Up to 2 000|100|"
    )
    assert chunks[1].printed_page == "15"
    assert chunks[1].token_count > 0


def test_definitions_are_chunked_per_term_and_contents_not_at_all():
    contents = page(*[line(f"{n}.1 Item {n} {n}", 50 + 10 * n) for n in range(1, 8)], number=1)
    definitions = page(
        heading("Definitions", 50, size=11),
        line("“Act” means the Ports", 60),
        line("Act of 2005.", 70),
        line("“Agent” means a vessel's agent.", 80),
        number=2,
    )
    chunks = build_chunks(build_sections(document(contents, definitions)))
    assert [(chunk.kind, chunk.content) for chunk in chunks] == [
        (ChunkKind.DEFINITION, "Definitions\n\n“Act” means the Ports\nAct of 2005."),
        (ChunkKind.DEFINITION, "Definitions\n\n“Agent” means a vessel's agent."),
    ]


def test_long_sections_are_split():
    long_line = "x" * 1500
    sections = build_sections(
        document(page(heading("7.1 CARGO", 50), *[line(long_line, 60 + i) for i in range(6)]))
    )
    chunks = build_chunks(sections)
    assert len(chunks) == 3
    assert all(len(chunk.content) <= MAX_CHUNK_CHARS + 100 for chunk in chunks)


def test_join_lines_mends_hyphenated_words_only():
    assert join_lines(["the vessel de-", "parts from", "Self-", "Propelled"]) == (
        "the vessel departs from\nSelf-\nPropelled"
    )
