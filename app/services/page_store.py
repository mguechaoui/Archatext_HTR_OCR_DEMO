"""
Page state store.

Tracks every uploaded page through the pipeline: upload -> segmentation ->
OCR -> export. This is an in-memory store, which is fine for a single-
process demo deployment. It's deliberately accessed only through the
`PageStore` interface below, so swapping in a real database (e.g. Redis for
multi-worker deployments, or Postgres for persistence) later means
reimplementing this one class — no other module touches the underlying
storage directly.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from app.models import OCREngine, PageOCRResult, PageSegmentation
from app.schemas.status import StageStatus


@dataclass
class PageState:
    page_id: str
    original_image_path: str
    upload_status: StageStatus = StageStatus.COMPLETED
    segmentation_status: StageStatus = StageStatus.NOT_STARTED
    ocr_status: StageStatus = StageStatus.NOT_STARTED
    error_message: str | None = None

    segmentation: PageSegmentation | None = None
    overlay_image_path: str | None = None
    raw_segmentation_result: object | None = None  # kraken's raw result, for re-cropping etc.

    ocr_results: dict[OCREngine, PageOCRResult] = field(default_factory=dict)
    last_ocr_engine: OCREngine | None = None


class PageStore:
    """Thread-safe in-memory store keyed by page_id."""

    def __init__(self):
        self._pages: dict[str, PageState] = {}
        self._lock = threading.Lock()

    def create(self, page_id: str, original_image_path: str) -> PageState:
        with self._lock:
            state = PageState(page_id=page_id, original_image_path=original_image_path)
            self._pages[page_id] = state
            return state

    def get(self, page_id: str) -> PageState | None:
        with self._lock:
            return self._pages.get(page_id)

    def get_or_raise(self, page_id: str) -> PageState:
        state = self.get(page_id)
        if state is None:
            raise KeyError(f"Unknown page_id: {page_id}")
        return state

    def update(self, page_id: str, **fields) -> PageState:
        with self._lock:
            state = self._pages.get(page_id)
            if state is None:
                raise KeyError(f"Unknown page_id: {page_id}")
            for key, value in fields.items():
                setattr(state, key, value)
            return state


_store: PageStore | None = None


def get_page_store() -> PageStore:
    global _store
    if _store is None:
        _store = PageStore()
    return _store
