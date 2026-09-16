"""Export endpoints: download page results in TXT, JSON, ALTO XML, PAGE
XML, or CSV format."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.core.dependencies import get_export_service
from app.models import OCREngine
from app.services.export_service import ExportService

router = APIRouter(prefix="/pages", tags=["export"])

_MEDIA_TYPES = {
    "txt": "text/plain",
    "json": "application/json",
    "alto_xml": "application/xml",
    "page_xml": "application/xml",
    "csv": "text/csv",
}


@router.get("/{page_id}/export/txt")
async def export_txt(
    page_id: str, engine: OCREngine | None = None, export_service: ExportService = Depends(get_export_service)
) -> FileResponse:
    path = export_service.export_txt(page_id, engine)
    return FileResponse(path, media_type=_MEDIA_TYPES["txt"], filename=path.name)


@router.get("/{page_id}/export/json")
async def export_json(
    page_id: str, engine: OCREngine | None = None, export_service: ExportService = Depends(get_export_service)
) -> FileResponse:
    path = export_service.export_json(page_id, engine)
    return FileResponse(path, media_type=_MEDIA_TYPES["json"], filename=path.name)


@router.get("/{page_id}/export/csv")
async def export_csv(
    page_id: str, engine: OCREngine | None = None, export_service: ExportService = Depends(get_export_service)
) -> FileResponse:
    path = export_service.export_csv(page_id, engine)
    return FileResponse(path, media_type=_MEDIA_TYPES["csv"], filename=path.name)


@router.get("/{page_id}/export/alto")
async def export_alto_xml(
    page_id: str, engine: OCREngine | None = None, export_service: ExportService = Depends(get_export_service)
) -> FileResponse:
    path = export_service.export_alto_xml(page_id, engine)
    return FileResponse(path, media_type=_MEDIA_TYPES["alto_xml"], filename=path.name)


@router.get("/{page_id}/export/page-xml")
async def export_page_xml(
    page_id: str, engine: OCREngine | None = None, export_service: ExportService = Depends(get_export_service)
) -> FileResponse:
    path = export_service.export_page_xml(page_id, engine)
    return FileResponse(path, media_type=_MEDIA_TYPES["page_xml"], filename=path.name)
