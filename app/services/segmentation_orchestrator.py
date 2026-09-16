"""Application-layer orchestration for the segmentation pipeline step."""
from __future__ import annotations

import logging

from PIL import Image

from app.config import settings
from app.core.exceptions import PageNotFoundError
from app.models import PageSegmentation
from app.schemas.status import StageStatus
from app.segmentation.registry import get_segmentation_service
from app.services.page_store import PageStore

logger = logging.getLogger(__name__)


class SegmentationOrchestrator:
    """Coordinates: load page -> run segmentation -> crop segments ->
    render overlay -> persist results into the page store."""

    def __init__(self, page_store: PageStore):
        self.page_store = page_store

    def run(self, page_id: str) -> PageSegmentation:
        state = self.page_store.get(page_id)
        if state is None:
            raise PageNotFoundError(f"No page found for page_id={page_id}")

        self.page_store.update(page_id, segmentation_status=StageStatus.IN_PROGRESS)

        try:
            image = Image.open(state.original_image_path).convert("RGB")
            service = get_segmentation_service()

            page_segmentation, raw_result = service.segment_image(image, page_id=page_id)
            page_segmentation.source_image_path = state.original_image_path

            crops_dir = settings.output_dir / page_id / "crops"
            page_segmentation = service.crop_segments(image, page_segmentation, crops_dir)

            overlay_path = settings.output_dir / page_id / "overlay.png"
            service.render_overlay(image, raw_result, overlay_path)
            page_segmentation.overlay_image_path = str(overlay_path)

            self.page_store.update(
                page_id,
                segmentation=page_segmentation,
                overlay_image_path=str(overlay_path),
                raw_segmentation_result=raw_result,
                segmentation_status=StageStatus.COMPLETED,
            )
            logger.info(
                "Segmentation complete for page_id=%s: %d segments",
                page_id,
                page_segmentation.total_segments,
            )
            return page_segmentation

        except Exception as e:
            self.page_store.update(
                page_id, segmentation_status=StageStatus.FAILED, error_message=str(e)
            )
            raise
