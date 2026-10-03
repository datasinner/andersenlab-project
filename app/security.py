"""Optional API-key protection for the /v1 routes (set API_AUTH_KEY).

Declared as an OpenAPI security scheme, so Swagger's Authorize button sends
the key. Health checks stay open."""

import hmac

from fastapi import Security
from fastapi.security import APIKeyHeader

from app.config import settings
from app.errors import AppError

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Required when the server sets API_AUTH_KEY.",
)


class UnauthorizedError(AppError):
    status_code = 401
    code = "UNAUTHORIZED"


async def require_api_key(key: str | None = Security(api_key_header)) -> None:
    expected = settings.api_auth_key
    if not expected:
        return
    if key is None or not hmac.compare_digest(key.encode(), expected.encode()):
        raise UnauthorizedError("Missing or wrong X-API-Key header")
