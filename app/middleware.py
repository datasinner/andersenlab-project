import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger("app.request")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request_id, binds it into structlog's contextvars so every
    log line emitted anywhere while handling this request carries it, and
    logs a start/finish pair with timing (the "access log")."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.monotonic()
        logger.info("request_started", method=request.method, path=request.url.path)

        response = await call_next(request)

        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "request_finished",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response
