"""Hybrid search and section access over the ingested TNPA document, with
the fake (word-hashing) embedder: the lexical ranking is real, the semantic
one is noise, so these tests check recall@5 like the retrieval eval does.
Real-embedding quality is measured by `make eval-retrieval`."""

from app.db import async_session_factory
from app.llm.embeddings import FakeEmbedder
from app.models import ChunkKind
from app.retrieval.context import chunk_body, find_definitions, outline, read_section
from app.retrieval.search import TariffSearch


def _search() -> TariffSearch:
    return TariffSearch(async_session_factory, FakeEmbedder(), top_k=5, rrf_k=60)


async def test_search_finds_the_section_that_defines_a_charge(ingested_tnpa):
    search = _search()
    expectations = {
        "running of vessel lines": "3.9",
        "tug assistance fee by vessel tonnage": "3.6",
        "port dues per 24 hour period pro rata": "4.1.1",
        "pilotage exemption certificate fees": "3.5",
    }
    for query, ref in expectations.items():
        hits = await search.search(ingested_tnpa.document_id, query)
        assert ref in [hit.section_ref for hit in hits], query


async def test_hits_carry_provenance_and_both_ranks(ingested_tnpa):
    (hit, *_) = await _search().search(ingested_tnpa.document_id, "running of vessel lines", k=1)
    assert hit.section_ref == "3.9"
    assert hit.section_title == "3.9 RUNNING OF VESSEL LINES"
    assert hit.page == 10 and hit.printed_page == "19"
    assert hit.content.startswith("SECTION 3 MARINE SERVICES > 3.9 RUNNING OF VESSEL LINES")
    assert hit.semantic_rank is not None and hit.lexical_rank is not None
    assert hit.score > 0


async def test_query_without_words_falls_back_to_semantic_ranking(ingested_tnpa):
    hits = await _search().search(ingested_tnpa.document_id, "…", k=3)
    assert len(hits) == 3
    assert all(hit.lexical_rank is None for hit in hits)


async def test_read_section_returns_text_tables_and_children(ingested_tnpa):
    async with async_session_factory() as session:
        towage = await read_section(session, ingested_tnpa.document_id, "3.6")
        port_fees = await read_section(session, ingested_tnpa.document_id, "4.1")
        missing = await read_section(session, ingested_tnpa.document_id, "99.9")

    assert towage.title == "3.6 TUGS/VESSEL ASSISTANCE AND/OR ATTENDANCE"
    assert "73 118.07" in towage.text and "A surcharge of 25% is payable" in towage.text
    assert "SECTION 3 MARINE SERVICES >" not in towage.text  # breadcrumbs stripped
    assert len(towage.chunk_ids) >= 3
    assert [ref for ref, _ in port_fees.children] == ["4.1.1", "4.1.2"]
    assert missing is None


async def test_outline_with_prefix_and_definitions(ingested_tnpa):
    async with async_session_factory() as session:
        marine = await outline(session, ingested_tnpa.document_id, prefix="3.")
        definitions = await find_definitions(session, ingested_tnpa.document_id, "TONNAGE")

    assert [section.ref for section in marine][:3] == ["3.1", "3.2", "3.3"]
    assert definitions
    assert all(chunk.kind == ChunkKind.DEFINITION for chunk in definitions)
    assert any("gross tonnage" in chunk.content for chunk in definitions)


def test_chunk_body_strips_the_breadcrumb():
    assert chunk_body("A > B\n\nbody text\n\nmore") == "body text\n\nmore"
    assert chunk_body("no breadcrumb") == "no breadcrumb"
