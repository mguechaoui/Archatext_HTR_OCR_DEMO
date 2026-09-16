"""
Abstract OCR interface.

Every OCR engine (the fine-tuned kraken model, HATFormer, and any future
model) implements this interface. The API and service layers only ever
talk to `BaseOCR` — they never import a concrete engine class directly.
Adding a new engine means:

  1. Write a new `BaseOCR` subclass in `app/ocr/engines/`.
  2. Register it in `app/ocr/registry.py`.

Nothing else in the codebase changes.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image

from app.models import OCREngine

if TYPE_CHECKING:
    from app.models.segmentation_result import Segment


@dataclass
class OCRPrediction:
    """Return type for a single-image inference call."""

    text: str
    confidence: float | None = None


class BaseOCR(ABC):
    """Common interface for all OCR engine adapters."""

    #: Identifier used in API requests/responses and in the model registry.
    engine_id: OCREngine

    @abstractmethod
    def load(self) -> None:
        """Load model weights into memory. Called once at startup."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: Image.Image, segment: Segment | None = None) -> OCRPrediction:
        """Run OCR on a single cropped line/segment image.

        `segment` carries the original baseline and boundary_polygon from the
        segmentation step. Engines that need the real geometry (e.g. kraken
        baseline models) should use it; engines that don't (e.g. HATFormer)
        can ignore it.
        """
        raise NotImplementedError

    def predict_batch(self, images: list[Image.Image]) -> list[OCRPrediction]:
        """Default batch implementation: predict one-by-one. Engines that
        support real batching (e.g. a transformer with a batched forward
        pass) should override this for better throughput."""
        return [self.predict(img) for img in images]

    def predict_timed(
        self, image: Image.Image, segment: Segment | None = None
    ) -> tuple[OCRPrediction, float]:
        """Convenience wrapper used by the OCR service to capture
        per-segment inference time without every engine reimplementing
        timing logic."""
        start = time.perf_counter()
        prediction = self.predict(image, segment=segment)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return prediction, elapsed_ms