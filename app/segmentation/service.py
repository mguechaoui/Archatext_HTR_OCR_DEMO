"""
Segmentation service — refactored from the original `seg_muhref.py` research
script.

What changed vs. the original script:
  - Model loading happens once (at app startup, via the singleton in
    `app.segmentation.registry`) instead of once per script run.
  - `segment()` now returns our own `PageSegmentation` domain object instead
    of writing JSON/PNG files directly inside the segmentation function.
  - Cropping and overlay rendering are kept as separate, optional steps so
    the API layer can decide what it actually needs (e.g. skip the
    matplotlib overlay if only bounding boxes are requested).

What stayed identical:
  - The actual call into kraken: `vgsl.TorchVGSLModel.load_model(...)` and
    `blla.segment(image, model=model, text_direction=...)`. This is the
    exact inference path from the original script — we do not touch it,
    per the "preserve original inference behavior exactly" requirement.
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings
from app.models import BoundingBox, PageSegmentation, Segment

logger = logging.getLogger(__name__)


class SegmentationService:
    """Wraps the kraken `blla` baseline segmenter behind a stable interface.

    Instantiate once (see `app.segmentation.registry.get_segmentation_service`)
    and reuse across requests — model loading is the expensive part.
    """

    def __init__(self, model_path: Path | None = None, text_direction: str | None = None):
        self.model_path = Path(model_path or settings.segmentation_model_path)
        self.text_direction = text_direction or settings.segmentation_text_direction
        self._model = None  # lazy-loaded kraken VGSL model

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """Load the segmentation model into memory. Call once at startup."""
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Segmentation model not found at '{self.model_path}'. "
                "Set OCR_SEGMENTATION_MODEL_PATH or place the .mlmodel there."
            )

        # Imported lazily so the rest of the app can be imported/tested
        # without kraken installed.
        from kraken.lib import vgsl

        logger.info("Loading segmentation model: %s", self.model_path)
        self._model = vgsl.TorchVGSLModel.load_model(str(self.model_path))
        logger.info("Segmentation model ready")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def segment_image(self, image: Image.Image, page_id: str | None = None) -> tuple[PageSegmentation, object]:
        """
        Run line segmentation on a single page image.

        Returns a tuple of (PageSegmentation domain object, raw kraken
        result), since downstream steps (cropping, overlay rendering) need
        the raw kraken `boundary`/`baseline` geometry that doesn't belong
        in the API-facing domain object verbatim.
        """
        if self._model is None:
            self.load()

        from kraken import blla

        page_id = page_id or uuid.uuid4().hex

        # This is the exact call from the original script:
        #   blla.segment(image, model=model, text_direction="horizontal-rl")
        result = blla.segment(image, model=self._model, text_direction=self.text_direction)

        segments: list[Segment] = []
        for i, line in enumerate(result.lines):
            boundary = [tuple(pt) for pt in (line.boundary or [])]
            baseline = [tuple(pt) for pt in (line.baseline or [])]
            bbox = self._bbox_from_polygon(boundary, image.size)

            segments.append(
                Segment(
                    segment_id=i,
                    order=i,
                    baseline=baseline,
                    boundary_polygon=boundary,
                    bounding_box=bbox,
                    confidence=getattr(line, "score", None),
                )
            )

        page_segmentation = PageSegmentation(
            page_id=page_id,
            source_image_path="",  # filled in by the caller, which knows the saved path
            image_width=image.width,
            image_height=image.height,
            segments=segments,
        )
        return page_segmentation, result

    @staticmethod
    def _bbox_from_polygon(
        boundary: list[tuple[float, float]], image_size: tuple[int, int]
    ) -> BoundingBox | None:
        if not boundary:
            return None
        pts = np.array(boundary)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        w, h = image_size
        return BoundingBox(
            x0=max(0, int(x0)),
            y0=max(0, int(y0)),
            x1=min(w, int(x1)),
            y1=min(h, int(y1)),
        )

    # ------------------------------------------------------------------ #
    # Post-processing (cropping / overlay) — identical logic to the
    # original script, just parameterized on output paths instead of
    # being hardcoded to a single output directory.
    # ------------------------------------------------------------------ #
    def crop_segments(
        self, image: Image.Image, page_segmentation: PageSegmentation, out_dir: Path
    ) -> PageSegmentation:
        """Crop every segment's bounding box out of the page image and save
        it to disk, recording the crop path on each Segment."""
        out_dir.mkdir(parents=True, exist_ok=True)
        arr = np.array(image)

        new_segments = []
        for seg in page_segmentation.segments:
            if seg.bounding_box is None:
                new_segments.append(seg)
                continue
            x0, y0, x1, y1 = seg.bounding_box.as_tuple()
            crop = arr[y0:y1, x0:x1]
            if crop.size == 0:
                new_segments.append(seg)
                continue
            crop_path = out_dir / f"segment_{seg.segment_id:03d}.png"
            Image.fromarray(crop).save(crop_path)
            new_segments.append(
                Segment(
                    segment_id=seg.segment_id,
                    order=seg.order,
                    baseline=seg.baseline,
                    boundary_polygon=seg.boundary_polygon,
                    bounding_box=seg.bounding_box,
                    crop_path=str(crop_path),
                    confidence=seg.confidence,
                )
            )

        page_segmentation.segments = new_segments
        return page_segmentation

    def render_overlay(
        self, image: Image.Image, raw_result, out_path: Path
    ) -> Path:
        """Render the same matplotlib overlay visualization as the original
        script (polygons + baselines + line numbers) and save it to
        `out_path`."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon

        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 18))
        ax.imshow(image)
        ax.set_title(f"Lines detected: {len(raw_result.lines)}", fontsize=13)
        ax.axis("off")
        colors = plt.cm.Set2(np.linspace(0, 1, max(len(raw_result.lines), 1)))
        for i, line in enumerate(raw_result.lines):
            c = colors[i % len(colors)]
            if line.boundary:
                pts = np.array(line.boundary)
                poly = Polygon(
                    pts, closed=True, linewidth=1.5, edgecolor=c, facecolor=(*c[:3], 0.15)
                )
                ax.add_patch(poly)
            if line.baseline:
                bl = np.array(line.baseline)
                ax.plot(bl[:, 0], bl[:, 1], color="red", linewidth=1.2, alpha=0.85)
                mid = bl[len(bl) // 2]
                ax.text(
                    mid[0], mid[1] - 8, str(i + 1), fontsize=7, color="white",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="navy", alpha=0.75),
                )
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path


def time_call(fn, *args, **kwargs):
    """Small helper to time a call in milliseconds. Used by callers that
    want inference timing without duplicating boilerplate."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms
