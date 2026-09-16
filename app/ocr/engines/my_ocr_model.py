"""
`MyOCRModel` — adapter around the fine-tuned kraken recognition checkpoint
produced by the ArMan training pipeline (`train_arman_working.py`).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from app.config import settings
from app.models import OCREngine
from app.ocr.base import BaseOCR, OCRPrediction
from app.utils.device import resolve_device

if TYPE_CHECKING:
    from app.models.segmentation_result import Segment

logger = logging.getLogger(__name__)

MODEL_INPUT_HEIGHT = 120


class MyOCRModel(BaseOCR):
    engine_id = OCREngine.MY_OCR_MODEL

    def __init__(self, checkpoint_path: Path | None = None, device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path or settings.myocr_checkpoint_path)
        self.device = resolve_device(device or settings.device)
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"MyOCRModel checkpoint not found at '{self.checkpoint_path}'. "
                "Set OCR_MYOCR_CHECKPOINT_PATH to your fine-tuned .mlmodel file."
            )
        from kraken.lib import models
        logger.info("Loading MyOCRModel checkpoint: %s (device=%s)", self.checkpoint_path, self.device)
        self._model = models.load_any(str(self.checkpoint_path), device=self.device)
        logger.info("MyOCRModel ready")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict(self, image: Image.Image, segment: Segment | None = None) -> OCRPrediction:
        if self._model is None:
            self.load()

        from kraken.containers import BaselineLine, Segmentation
        from kraken.rpred import rpred

        # Grayscale + resize to fixed height=120 (model spec: [1,120,0,1...])
        image = image.convert("L")
        w, h = image.size
        new_w = max(1, int(w * MODEL_INPUT_HEIGHT / h))
        image = image.resize((new_w, MODEL_INPUT_HEIGHT), Image.BICUBIC)
        w, h = image.size  # h == 120 now

        # kraken's bounds check (from extract_polygons source):
        #   pl.max(axis=0) -> [max_x, max_y]
        #   [::-1]         -> [max_y, max_x]
        #   compared to imshape = [height, width] = [h, w]
        # So it fails if max_y >= h OR max_x >= w.
        # Using w-1 and h-1 guarantees both checks pass.
        W = w - 1
        H = h - 1

        line = BaselineLine(
            id="line_0",
            baseline=[(0, H // 2), (W, H // 2)],
            boundary=[(0, 0), (W, 0), (W, H), (0, H)],
        )
        bounds = Segmentation(
            type="baselines",
            imagename="",
            text_direction="horizontal-rl",
            script_detection=False,
            lines=[line],
        )

        records = list(rpred(network=self._model, im=image, bounds=bounds))
        if not records:
            return OCRPrediction(text="", confidence=None)

        record = records[0]
        text = str(record.prediction)
        confidence = None
        if getattr(record, "confidences", None):
            confidences = list(record.confidences)
            if confidences:
                confidence = float(sum(confidences) / len(confidences))

        return OCRPrediction(text=text, confidence=confidence)