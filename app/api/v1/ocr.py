"""OCR endpoints: list engines, run OCR, fetch results and reconstructed text."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.dependencies import get_ocr_orchestrator, get_store
from app.core.exceptions import PageNotFoundError
from app.models import OCREngine
from app.ocr.registry import list_available_engines, list_registered_engines
from app.schemas import (
    AvailableEngineSchema,
    AvailableEnginesResponse,
    PageOCRResultSchema,
    SegmentOCRResultSchema,
)
from app.services.ocr_orchestrator import OCROrchestrator
from app.services.page_store import PageStore

router = APIRouter(prefix="/pages", tags=["ocr"])

_ENGINE_LABELS = {
    OCREngine.MY_OCR_MODEL: "Fine-tuned ArMan Model",
    OCREngine.HATFORMER: "HATFormer",
}


def _to_schema(result) -> PageOCRResultSchema:
    return PageOCRResultSchema(
        page_id=result.page_id,
        engine=result.engine,
        total_inference_time_ms=result.total_inference_time_ms,
        segment_results=[
            SegmentOCRResultSchema(
                segment_id=r.segment_id,
                text=r.text,
                confidence=r.confidence,
                inference_time_ms=r.inference_time_ms,
            )
            for r in result.segment_results
        ],
        page_text=result.page_text,
    )


@router.get("/ocr/engines", response_model=AvailableEnginesResponse)
async def list_engines() -> AvailableEnginesResponse:
    available = set(list_available_engines())
    return AvailableEnginesResponse(
        engines=[
            AvailableEngineSchema(
                engine=engine_id,
                available=engine_id in available,
                label=_ENGINE_LABELS.get(engine_id, engine_id.value),
            )
            for engine_id in list_registered_engines()
        ]
    )


@router.post("/{page_id}/ocr/run", response_model=PageOCRResultSchema)
async def run_ocr(
    page_id: str,
    engine: OCREngine,
    orchestrator: OCROrchestrator = Depends(get_ocr_orchestrator),
) -> PageOCRResultSchema:
    result = orchestrator.run(page_id, engine)
    return _to_schema(result)


@router.get("/{page_id}/ocr/stream")
async def stream_ocr(
    page_id: str,
    engine: OCREngine,
    orchestrator: OCROrchestrator = Depends(get_ocr_orchestrator),
) -> StreamingResponse:
    """
    Server-Sent Events endpoint. Yields one JSON event per segment as soon
    as it is recognized, so the frontend can display results line by line
    without waiting for the full page to finish.

    Event format:
        data: {"segment_id": 0, "text": "...", "confidence": 0.95, "done": false}
        data: {"done": true, "total_inference_time_ms": 1234.5}
    """
    def generate():
        for segment_result, total_ms in orchestrator.run_streaming(page_id, engine):
            if segment_result is None:
                # final event
                payload = json.dumps({"done": True, "total_inference_time_ms": round(total_ms, 2)})
            else:
                payload = json.dumps({
                    "segment_id": segment_result.segment_id,
                    "text": segment_result.text,
                    "confidence": segment_result.confidence,
                    "inference_time_ms": round(segment_result.inference_time_ms, 2),
                    "done": False,
                })
            yield f"data: {payload}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/{page_id}/ocr", response_model=PageOCRResultSchema)
async def get_ocr_result(
    page_id: str,
    engine: OCREngine | None = None,
    store: PageStore = Depends(get_store),
) -> PageOCRResultSchema:
    state = store.get(page_id)
    if state is None:
        raise PageNotFoundError(f"No page found for page_id={page_id}")

    engine_id = engine or state.last_ocr_engine
    if engine_id is None or engine_id not in state.ocr_results:
        raise PageNotFoundError(f"No OCR results found for page_id={page_id}")

    return _to_schema(state.ocr_results[engine_id])


@router.get("/{page_id}/text")
async def get_page_text(
    page_id: str,
    engine: OCREngine | None = None,
    store: PageStore = Depends(get_store),
) -> dict:
    state = store.get(page_id)
    if state is None:
        raise PageNotFoundError(f"No page found for page_id={page_id}")

    engine_id = engine or state.last_ocr_engine
    if engine_id is None or engine_id not in state.ocr_results:
        raise PageNotFoundError(f"No OCR results found for page_id={page_id}")

    return {"page_id": page_id, "engine": engine_id.value, "page_text": state.ocr_results[engine_id].page_text}