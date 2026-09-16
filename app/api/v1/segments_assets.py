"""Serves the binary image assets produced by segmentation: individual
segment crops and the full-page overlay visualization."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.core.dependencies import get_store
from app.core.exceptions import PageNotFoundError, SegmentationNotRunError
from app.services.page_store import PageStore

router = APIRouter(prefix="/pages", tags=["segmentation-assets"])


@router.get("/{page_id}/segmentation/overlay")
async def get_overlay_image(page_id: str, store: PageStore = Depends(get_store)) -> FileResponse:
    state = store.get(page_id)
    if state is None:
        raise PageNotFoundError(f"No page found for page_id={page_id}")
    if not state.overlay_image_path:
        raise SegmentationNotRunError(f"No overlay available for page_id={page_id}")
    return FileResponse(state.overlay_image_path, media_type="image/png")


@router.get("/{page_id}/segments/{segment_id}/crop")
async def get_segment_crop(
    page_id: str, segment_id: int, store: PageStore = Depends(get_store)
) -> FileResponse:
    state = store.get(page_id)
    if state is None:
        raise PageNotFoundError(f"No page found for page_id={page_id}")
    if state.segmentation is None:
        raise SegmentationNotRunError(f"Segmentation has not been run for page_id={page_id}")

    segment = next((s for s in state.segmentation.segments if s.segment_id == segment_id), None)
    if segment is None or not segment.crop_path:
        raise PageNotFoundError(f"No crop found for page_id={page_id}, segment_id={segment_id}")

    return FileResponse(segment.crop_path, media_type="image/png")
