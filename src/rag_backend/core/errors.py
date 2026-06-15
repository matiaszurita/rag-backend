import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class ConflictError(AppError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "conflict",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, status_code=409, details=details)


class NotFoundError(AppError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "not_found",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, status_code=404, details=details)


class AuthenticationError(AppError):
    def __init__(
        self,
        message: str = "Authentication required",
        *,
        code: str = "authentication_error",
    ) -> None:
        super().__init__(message, code=code, status_code=401)


class AuthorizationError(AppError):
    def __init__(self, message: str = "Not allowed", *, code: str = "authorization_error") -> None:
        super().__init__(message, code=code, status_code=403)


def _error_payload(error: AppError) -> dict[str, Any]:
    return {
        "error": {
            "code": error.code,
            "message": error.message,
            "details": error.details,
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, error: AppError) -> JSONResponse:
        return JSONResponse(status_code=error.status_code, content=_error_payload(error))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        payload = {
            "error": {
                "code": "validation_error",
                "message": "Request validation failed",
                "details": {"errors": error.errors()},
            }
        }
        return JSONResponse(status_code=422, content=payload)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, error: Exception) -> JSONResponse:
        logger.exception("Unhandled application error", exc_info=error)
        payload = {
            "error": {
                "code": "internal_server_error",
                "message": "Unexpected server error",
                "details": {},
            }
        }
        return JSONResponse(status_code=500, content=payload)
