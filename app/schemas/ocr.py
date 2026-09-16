"""Pydantic schemas for the OCR endpoints."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.models import OCREngine


class OCRRequest(BaseModel):
    page_id: str = Field(..., description="Page ID returned from the upload endpoint")
    engine: OCREngine = Field(..., description="Which OCR engine to run")


class SegmentOCRResultSchema(BaseModel):
    segment_id: int
    text: str
    confidence: float | None = None
    inference_time_ms: float


class PageOCRResultSchema(BaseModel):
    page_id: str
    engine: OCREngine
    total_inference_time_ms: float
    segment_results: list[SegmentOCRResultSchema]
    page_text: str


class AvailableEngineSchema(BaseModel):
    engine: OCREngine
    available: bool
    label: str


class AvailableEnginesResponse(BaseModel):
    engines: list[AvailableEngineSchema]
