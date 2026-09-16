"""
Singleton accessor for the segmentation service.

FastAPI's dependency-injection system calls `get_segmentation_service()` for
every request that needs it, but the underlying model is only ever loaded
once — at application startup, via `init_segmentation_service()` in
`app.core.lifespan`.
"""
from __future__ import annotations

from app.segmentation.service import SegmentationService

_service: SegmentationService | None = None


def init_segmentation_service() -> SegmentationService:
    global _service
    if _service is None:
        _service = SegmentationService()
        _service.load()
    return _service


def get_segmentation_service() -> SegmentationService:
    if _service is None:
        raise RuntimeError(
            "SegmentationService not initialized. "
            "init_segmentation_service() must run during app startup."
        )
    return _service
