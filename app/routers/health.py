from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import DocumentStatus, TariffDocument
from app.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse, response_model_exclude_none=True)
async def ready(response: Response, session: AsyncSession = Depends(get_session)) -> ReadyResponse:
    # A readiness probe: any failure to reach the database means "not
    # ready", regardless of its specific cause. Having no ingested documents
    # is still "ready": tariffs can be uploaded through the API.
    try:
        documents_ready = await session.scalar(
            select(func.count())
            .select_from(TariffDocument)
            .where(TariffDocument.status == DocumentStatus.READY)
        )
    except Exception:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(status="unavailable")
    return ReadyResponse(status="ready", documents_ready=documents_ready)
