"""Application-level exceptions, mapped to HTTP responses by the API layer
(see app/core/error_handlers.py). Keeping these separate from FastAPI's
HTTPException means services stay framework-agnostic."""
from __future__ import annotations


class OCRPlatformError(Exception):
    """Base class for all application-level errors."""


class UnsupportedFileTypeError(OCRPlatformError):
    """Raised when an uploaded file's extension isn't supported."""


class InvalidImageError(OCRPlatformError):
    """Raised when an uploaded file can't be decoded as an image, or
    violates a size constraint."""


class PageNotFoundError(OCRPlatformError):
    """Raised when a page_id doesn't exist in the page store."""


class SegmentationNotRunError(OCRPlatformError):
    """Raised when OCR (or any step needing segments) is requested before
    segmentation has completed for a page."""


class EngineUnavailableError(OCRPlatformError):
    """Raised when the requested OCR engine failed to load (e.g. missing
    checkpoint) and therefore can't serve requests."""
