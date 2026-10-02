import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.db import async_session_factory, engine
from app.errors import AppError
from app.llm.client import build_llm_clients
from app.llm.embeddings import build_embedder
from app.llm.resilience import ResiliencePolicy
from app.logging_conf import configure_logging
from app.middleware import RequestContextMiddleware
from app.routers import calculations, documents, health, rules
from app.schemas import ErrorDetail, ErrorResponse
from app.services.calculations import CalculationService
from app.services.rulebook import RulebookService

configure_logging()
logger = structlog.get_logger("app.lifespan")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # One policy per process: the chat client and the embedder share its
    # semaphore. A missing API key fails here, at startup, not mid-request.
    policy = ResiliencePolicy.from_settings()
    app.state.llm_client, app.state.compile_llm_client = build_llm_clients(policy)
    app.state.embedder = build_embedder(policy)
    app.state.rulebook = RulebookService(
        async_session_factory, app.state.compile_llm_client, app.state.embedder
    )
    app.state.calculations = CalculationService(
        async_session_factory, app.state.llm_client, app.state.rulebook
    )
    logger.info(
        "startup_complete",
        app_env=settings.app_env,
        llm_provider=settings.llm_provider,
        llm_model=app.state.llm_client.model_name,
        compile_model=app.state.compile_llm_client.model_name,
        embedding_model=app.state.embedder.model_name,
    )

    yield

    logger.info("shutdown_started")
    app.state.llm_client.shutdown()
    await engine.dispose()
    logger.info("shutdown_complete")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())


def _error_response(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(code=code, message=message, request_id=_request_id(request))
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def _validation_error_message(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "Invalid request"
    first = errors[0]
    loc = ".".join(str(part) for part in first["loc"] if part != "body")
    return f"{loc}: {first['msg']}" if loc else first["msg"]


def _http_error_code_and_message(exc: StarletteHTTPException) -> tuple[str, str]:
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        return exc.detail["code"], exc.detail.get("message", exc.detail["code"])
    return "HTTP_ERROR", str(exc.detail)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Port Tariff Agent",
        description=(
            "Reads port tariff documents and calculates the dues payable by a vessel call. "
            "Use this page to upload tariff PDFs and run calculations."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(rules.router)
    app.include_router(calculations.router)

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return _error_response(request, exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(request, 422, "VALIDATION_ERROR", _validation_error_message(exc))

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code, message = _http_error_code_and_message(exc)
        return _error_response(request, exc.status_code, code, message)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Never leak internals in the body; the traceback goes to the logs,
        # correlated by request_id.
        logger.exception("unhandled_exception", path=request.url.path)
        return _error_response(request, 500, "INTERNAL_ERROR", "Internal server error")

    return app


app = create_app()
