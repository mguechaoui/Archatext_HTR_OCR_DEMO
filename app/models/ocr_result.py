"""Domain entities for OCR recognition results."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class OCREngine(str, Enum):
    """Registry of supported OCR engine identifiers.

    Adding a new engine means adding one value here and one new BaseOCR
    subclass — nothing else in the domain layer changes.
    """

    MY_OCR_MODEL = "my_ocr_model"
    HATFORMER = "hatformer"


@dataclass
class SegmentOCRResult:
    """OCR output for a single segment (line)."""

    segment_id: int
    text: str
    confidence: float | None
    inference_time_ms: float
    engine: OCREngine


@dataclass
class PageOCRResult:
    """Aggregated OCR output for an entire page, in reading order."""

    page_id: str
    engine: OCREngine
    segment_results: list[SegmentOCRResult] = field(default_factory=list)
    total_inference_time_ms: float = 0.0

    @property
    def page_text(self) -> str:
        """Reconstruct full page text in reading order (segment order is
        assumed to already reflect reading order, as produced by the
        segmentation service)."""
        return "\n".join(r.text for r in self.segment_results)
