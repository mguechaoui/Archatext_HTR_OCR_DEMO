"""Maps application exceptions to HTTP responses, registered on the
FastAPI app in app/main.py. Keeps route handlers free of repetitive
try/except blocks."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    EngineUnavailableError,
    InvalidImageError,
    OCRPlatformError,
    PageNotFoundError,
    SegmentationNotRunError,
    UnsupportedFileTypeError,
)

logger = logging.getLogger(__name__)

_STATUS_MAP: dict[type[OCRPlatformError], int] = {
    UnsupportedFileTypeError: 415,
    InvalidImageError: 422,
    PageNotFoundError: 404,
    SegmentationNotRunError: 409,
    EngineUnavailableError: 503,
}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(OCRPlatformError)
    async def handle_platform_error(request: Request, exc: OCRPlatformError):
        status_code = _STATUS_MAP.get(type(exc), 400)
        logger.warning("%s on %s: %s", type(exc).__name__, request.url.path, exc)
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})
