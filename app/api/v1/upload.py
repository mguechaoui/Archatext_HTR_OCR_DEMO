"""Upload endpoint: accepts a manuscript page image."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.dependencies import get_upload_service
from app.schemas import UploadResponse
from app.services.upload_service import UploadService

router = APIRouter(prefix="/pages", tags=["upload"])


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_page(
    file: UploadFile = File(..., description="Manuscript page image (jpg/png/tif)"),
    upload_service: UploadService = Depends(get_upload_service),
) -> UploadResponse:
    page_id, dest_path, image = await upload_service.save_upload(file)
    return UploadResponse(
        page_id=page_id,
        filename=dest_path.name,
        original_image_url=f"/static/uploads/{dest_path.name}",
        width=image.width,
        height=image.height,
    )
