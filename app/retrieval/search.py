"""Hybrid search over one tariff document's chunks.

Two rankings are fused with Reciprocal Rank Fusion (score = Σ 1 / (k + rank)):
- semantic: pgvector cosine distance to the query embedding;
- lexical: Postgres full-text rank, with the query's words OR-ed together.
Lexical search matters here: tariff text is full of exact tokens ("or part
thereof", port names, "per 100 tons") that embeddings blur.
"""

import re
import uuid
from dataclasses import dataclass

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import settings
from app.llm.embeddings import Embedder
from app.models import Chunk, DocumentSection

_WORD = re.compile(r"[^\W_]+")
_CANDIDATE_FACTOR = 4
_MIN_CANDIDATES = 20


@dataclass(frozen=True)
class SearchHit:
    chunk_id: int
    section_ref: str
    section_title: str
    page: int
    printed_page: str | None
    kind: str
    content: str
    score: float
    semantic_rank: int | None
    lexical_rank: int | None


class TariffSearch:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        *,
        top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> None:
        self._sessions = session_factory
        self._embedder = embedder
        self.top_k = top_k or settings.retrieval_top_k
        self.rrf_k = rrf_k or settings.retrieval_rrf_k

    async def search(
        self, document_id: uuid.UUID, query: str, k: int | None = None
    ) -> list[SearchHit]:
        k = k or self.top_k
        candidates = max(k * _CANDIDATE_FACTOR, _MIN_CANDIDATES)
        (query_vector,) = await self._embedder.embed([query], name="embed_query")

        async with self._sessions() as session:
            semantic = await _semantic_ranking(session, document_id, query_vector, candidates)
            lexical = await _lexical_ranking(session, document_id, query, candidates)
            scores: dict[int, float] = {}
            for ranking in (semantic, lexical):
                for rank, chunk_id in enumerate(ranking, start=1):
                    scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
            best = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)[:k]
            rows = await session.execute(
                select(Chunk, DocumentSection)
                .join(DocumentSection, Chunk.section_id == DocumentSection.id)
                .where(Chunk.id.in_(best))
            )
            by_id = {chunk.id: (chunk, section) for chunk, section in rows}

        hits = []
        for chunk_id in best:
            chunk, section = by_id[chunk_id]
            hits.append(
                SearchHit(
                    chunk_id=chunk.id,
                    section_ref=section.ref,
                    section_title=section.title,
                    page=chunk.page,
                    printed_page=chunk.printed_page,
                    kind=chunk.kind,
                    content=chunk.content,
                    score=scores[chunk_id],
                    semantic_rank=_rank_of(chunk_id, semantic),
                    lexical_rank=_rank_of(chunk_id, lexical),
                )
            )
        return hits


async def _semantic_ranking(
    session: AsyncSession, document_id: uuid.UUID, vector: list[float], limit: int
) -> list[int]:
    # Without iterative scans, an HNSW search filtered to one document can
    # return fewer rows than asked for once the index holds many documents.
    await session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    result = await session.scalars(
        select(Chunk.id)
        .where(Chunk.document_id == document_id, Chunk.embedding.is_not(None))
        .order_by(Chunk.embedding.cosine_distance(vector))
        .limit(limit)
    )
    return list(result)


async def _lexical_ranking(
    session: AsyncSession, document_id: uuid.UUID, query: str, limit: int
) -> list[int]:
    words = _WORD.findall(query)
    if not words:
        return []
    # OR the words: a long natural-language query should still match chunks
    # that contain most, not all, of its terms. Stop words are dropped by
    # the english configuration.
    ts_query = func.websearch_to_tsquery("english", " or ".join(words))
    rank = func.ts_rank_cd(Chunk.tsv, ts_query)
    result = await session.scalars(
        select(Chunk.id)
        .where(Chunk.document_id == document_id, Chunk.tsv.op("@@")(ts_query))
        .order_by(rank.desc(), Chunk.ordinal)
        .limit(limit)
    )
    return list(result)


def _rank_of(chunk_id: int, ranking: list[int]) -> int | None:
    try:
        return ranking.index(chunk_id) + 1
    except ValueError:
        return None
