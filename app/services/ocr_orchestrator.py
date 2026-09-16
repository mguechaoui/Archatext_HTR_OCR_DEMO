"""Application-layer orchestration for the OCR pipeline step."""
from __future__ import annotations

import logging
from collections.abc import Generator

from PIL import Image

from app.core.exceptions import EngineUnavailableError, PageNotFoundError, SegmentationNotRunError
from app.models import OCREngine, PageOCRResult, SegmentOCRResult
from app.ocr.registry import get_ocr_engine, list_available_engines
from app.schemas.status import StageStatus
from app.services.page_store import PageStore

logger = logging.getLogger(__name__)


class OCROrchestrator:
    def __init__(self, page_store: PageStore):
        self.page_store = page_store

    def _validate(self, page_id: str, engine_id: OCREngine):
        state = self.page_store.get(page_id)
        if state is None:
            raise PageNotFoundError(f"No page found for page_id={page_id}")
        if state.segmentation is None or not state.segmentation.segments:
            raise SegmentationNotRunError(
                f"Segmentation has not been run for page_id={page_id}. Run segmentation first."
            )
        if engine_id not in list_available_engines():
            raise EngineUnavailableError(
                f"OCR engine '{engine_id.value}' is not available."
            )
        return state

    def run(self, page_id: str, engine_id: OCREngine) -> PageOCRResult:
        """Blocking: run all segments and return full result."""
        segment_results = []
        total_time_ms = 0.0
        for seg_result, total_ms in self.run_streaming(page_id, engine_id):
            if seg_result is not None:
                segment_results.append(seg_result)
                total_time_ms = total_ms

        state = self.page_store.get(page_id)
        page_result = PageOCRResult(
            page_id=page_id,
            engine=engine_id,
            segment_results=segment_results,
            total_inference_time_ms=round(total_time_ms, 2),
        )
        state.ocr_results[engine_id] = page_result
        self.page_store.update(
            page_id,
            ocr_status=StageStatus.COMPLETED,
            last_ocr_engine=engine_id,
        )
        return page_result

    def run_streaming(
        self, page_id: str, engine_id: OCREngine
    ) -> Generator[tuple[SegmentOCRResult | None, float], None, None]:
        """
        Yields (SegmentOCRResult, cumulative_ms) for each segment as it
        finishes, then yields (None, total_ms) as the final sentinel.
        """
        state = self._validate(page_id, engine_id)
        engine = get_ocr_engine(engine_id)
        self.page_store.update(page_id, ocr_status=StageStatus.IN_PROGRESS)

        total_time_ms = 0.0
        segment_results = []

        try:
            for segment in state.segmentation.segments:
                if not segment.crop_path:
                    continue
                image = Image.open(segment.crop_path).convert("RGB")
                prediction, elapsed_ms = engine.predict_timed(image, segment=segment)
                total_time_ms += elapsed_ms

                seg_result = SegmentOCRResult(
                    segment_id=segment.segment_id,
                    text=prediction.text,
                    confidence=prediction.confidence,
                    inference_time_ms=round(elapsed_ms, 2),
                    engine=engine_id,
                )
                segment_results.append(seg_result)

                logger.info(
                    "Segment %d done in %.1fms: %s",
                    segment.segment_id, elapsed_ms, prediction.text[:40]
                )
                yield seg_result, total_time_ms

            # final sentinel
            yield None, total_time_ms

        except Exception as e:
            self.page_store.update(page_id, ocr_status=StageStatus.FAILED, error_message=str(e))
            raise