import hashlib
import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.config import settings
from app.models import Chunk, ChunkKind, DocumentSection, DocumentStatus, TariffDocument


def _document(content: bytes = b"%PDF-1.7 test", status: str = DocumentStatus.PENDING):
    return TariffDocument(
        source_filename="tariff.pdf",
        content=content,
        checksum=hashlib.sha256(content).hexdigest(),
        status=status,
    )


def _unit_vector(index: int) -> list[float]:
    vector = [0.0] * settings.embedding_dimensions
    vector[index] = 1.0
    return vector


async def _document_with_chunks(db_session) -> TariffDocument:
    document = _document()
    db_session.add(document)
    await db_session.flush()

    section = DocumentSection(
        document_id=document.id,
        ref="3.6",
        title="Tug assistance",
        path="MARINE SERVICES > 3.6 Tug assistance",
        level=2,
        page_start=8,
        page_end=8,
        ordinal=1,
    )
    db_session.add(section)
    await db_session.flush()

    db_session.add_all(
        [
            Chunk(
                document_id=document.id,
                section_id=section.id,
                kind=ChunkKind.TABLE,
                content="Fees per service based on the vessel's tonnage",
                page=8,
                ordinal=1,
                token_count=8,
                embedding=_unit_vector(0),
            ),
            Chunk(
                document_id=document.id,
                section_id=section.id,
                kind=ChunkKind.TEXT,
                content="A surcharge is payable outside ordinary working hours",
                page=8,
                ordinal=2,
                token_count=9,
                embedding=_unit_vector(1),
            ),
        ]
    )
    await db_session.commit()
    return document


async def test_chunk_full_text_vector_is_generated(db_session):
    await _document_with_chunks(db_session)

    matches = await db_session.scalars(
        select(Chunk.content).where(Chunk.tsv.op("@@")(func.plainto_tsquery("english", "tonnage")))
    )
    assert matches.all() == ["Fees per service based on the vessel's tonnage"]


async def test_chunk_cosine_search_orders_by_similarity(db_session):
    await _document_with_chunks(db_session)

    nearest = await db_session.scalars(
        select(Chunk.ordinal).order_by(Chunk.embedding.cosine_distance(_unit_vector(1)))
    )
    assert nearest.all() == [2, 1]


async def test_duplicate_checksum_is_rejected(db_session):
    db_session.add(_document())
    await db_session.commit()

    db_session.add(_document())
    with pytest.raises(IntegrityError):
        await db_session.commit()


async def test_unknown_status_is_rejected(db_session):
    db_session.add(_document(status="almost-ready"))
    with pytest.raises(IntegrityError, match="ck_tariff_document_status"):
        await db_session.commit()


async def test_deleting_a_document_cascades_to_sections_and_chunks(db_session):
    document = await _document_with_chunks(db_session)

    await db_session.delete(document)
    await db_session.commit()

    assert await db_session.scalar(select(func.count()).select_from(DocumentSection)) == 0
    assert await db_session.scalar(select(func.count()).select_from(Chunk)) == 0


async def test_ready_reports_ready_document_count(client, db_session):
    db_session.add(_document(content=b"%PDF ready", status=DocumentStatus.READY))
    db_session.add(_document(content=b"%PDF parsing", status=DocumentStatus.PARSING))
    await db_session.commit()

    response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "documents_ready": 1}


async def test_document_ids_are_uuids(db_session):
    document = _document()
    db_session.add(document)
    await db_session.commit()
    assert isinstance(document.id, uuid.UUID)
