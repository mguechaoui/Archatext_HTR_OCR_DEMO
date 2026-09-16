"""Segmentation endpoints: run the pipeline and fetch its results."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from app.core.dependencies import get_segmentation_orchestrator, get_store
from app.core.exceptions import PageNotFoundError, SegmentationNotRunError
from app.schemas import BoundingBoxSchema, PageSegmentationSchema, SegmentSchema
from app.services.page_store import PageStore
from app.services.segmentation_orchestrator import SegmentationOrchestrator

router = APIRouter(prefix="/pages", tags=["segmentation"])


def _to_schema(page_id: str, state) -> PageSegmentationSchema:
    seg = state.segmentation
    return PageSegmentationSchema(
        page_id=page_id,
        image_width=seg.image_width,
        image_height=seg.image_height,
        total_segments=seg.total_segments,
        original_image_url=f"/static/uploads/{Path(state.original_image_path).name}",
        overlay_image_url=f"/api/v1/pages/{page_id}/segmentation/overlay" if state.overlay_image_path else None,
        segments=[
            SegmentSchema(
                segment_id=s.segment_id,
                order=s.order,
                baseline=s.baseline,
                boundary_polygon=s.boundary_polygon,
                bounding_box=BoundingBoxSchema(**s.bounding_box.__dict__) if s.bounding_box else None,
                crop_url=f"/api/v1/pages/{page_id}/segments/{s.segment_id}/crop" if s.crop_path else None,
                confidence=s.confidence,
            )
            for s in seg.segments
        ],
    )


@router.post("/{page_id}/segmentation/run", response_model=PageSegmentationSchema)
async def run_segmentation(
    page_id: str,
    orchestrator: SegmentationOrchestrator = Depends(get_segmentation_orchestrator),
    store: PageStore = Depends(get_store),
) -> PageSegmentationSchema:
    orchestrator.run(page_id)
    state = store.get_or_raise(page_id)
    return _to_schema(page_id, state)


@router.get("/{page_id}/segmentation", response_model=PageSegmentationSchema)
async def get_segmentation(
    page_id: str,
    store: PageStore = Depends(get_store),
) -> PageSegmentationSchema:
    state = store.get(page_id)
    if state is None:
        raise PageNotFoundError(f"No page found for page_id={page_id}")
    if state.segmentation is None:
        raise SegmentationNotRunError(f"Segmentation has not been run yet for page_id={page_id}")
    return _to_schema(page_id, state)
