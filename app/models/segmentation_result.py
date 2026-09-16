"""
Domain entities for the segmentation pipeline.

These are plain dataclasses, intentionally independent of both kraken's
internal types and the API's Pydantic schemas. The segmentation service
converts kraken's `Segmentation` result into these; the API layer converts
these into response schemas. Neither side needs to know about the other's
representation.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BoundingBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class Segment:
    """A single detected line/region on a manuscript page."""

    segment_id: int
    order: int
    baseline: list[tuple[float, float]] = field(default_factory=list)
    boundary_polygon: list[tuple[float, float]] = field(default_factory=list)
    bounding_box: BoundingBox | None = None
    crop_path: str | None = None
    confidence: float | None = None


@dataclass
class PageSegmentation:
    """Result of running the segmentation pipeline on one manuscript page."""

    page_id: str
    source_image_path: str
    image_width: int
    image_height: int
    segments: list[Segment] = field(default_factory=list)
    overlay_image_path: str | None = None

    @property
    def total_segments(self) -> int:
        return len(self.segments)
