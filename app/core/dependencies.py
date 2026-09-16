"""
FastAPI dependency providers.

Routes depend on these functions via `Depends(...)`, never on concrete
service classes directly. This is what the prompt's "dependency injection
where appropriate" requirement refers to in practice: it keeps route
handlers testable (swap a dependency in tests) and keeps the wiring in one
place.
"""
from __future__ import annotations

from app.services.export_service import ExportService
from app.services.ocr_orchestrator import OCROrchestrator
from app.services.page_store import PageStore, get_page_store
from app.services.segmentation_orchestrator import SegmentationOrchestrator
from app.services.upload_service import UploadService


def get_store() -> PageStore:
    return get_page_store()


def get_upload_service() -> UploadService:
    return UploadService(page_store=get_page_store())


def get_segmentation_orchestrator() -> SegmentationOrchestrator:
    return SegmentationOrchestrator(page_store=get_page_store())


def get_ocr_orchestrator() -> OCROrchestrator:
    return OCROrchestrator(page_store=get_page_store())


def get_export_service() -> ExportService:
    return ExportService(page_store=get_page_store())
