"""Status endpoint: lets the frontend poll where a page is in the pipeline."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.dependencies import get_store
from app.core.exceptions import PageNotFoundError
from app.schemas import PageStatusResponse
from app.services.page_store import PageStore

router = APIRouter(prefix="/pages", tags=["status"])


@router.get("/{page_id}/status", response_model=PageStatusResponse)
async def get_page_status(page_id: str, store: PageStore = Depends(get_store)) -> PageStatusResponse:
    state = store.get(page_id)
    if state is None:
        raise PageNotFoundError(f"No page found for page_id={page_id}")

    return PageStatusResponse(
        page_id=page_id,
        upload=state.upload_status,
        segmentation=state.segmentation_status,
        ocr=state.ocr_status,
        ocr_engine_used=state.last_ocr_engine.value if state.last_ocr_engine else None,
        error_message=state.error_message,
    )
