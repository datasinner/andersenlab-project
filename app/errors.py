class AppError(Exception):
    """Base class for errors that map to a specific HTTP status and error code.

    Services raise subclasses of this; the exception handler in app.main turns
    them into the standard {"error": {"code", "message", "request_id"}} body,
    so routers never build error responses by hand.
    """

    status_code: int = 500
    code: str = "INTERNAL_ERROR"

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.code
        super().__init__(self.message)
