"""Pydantic schemas for the segmentation endpoints."""
from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBoxSchema(BaseModel):
    x0: int
    y0: int
    x1: int
    y1: int


class SegmentSchema(BaseModel):
    segment_id: int
    order: int
    baseline: list[tuple[float, float]] = Field(default_factory=list)
    boundary_polygon: list[tuple[float, float]] = Field(default_factory=list)
    bounding_box: BoundingBoxSchema | None = None
    crop_url: str | None = Field(default=None, description="URL to fetch this segment's cropped image")
    confidence: float | None = None


class PageSegmentationSchema(BaseModel):
    page_id: str
    image_width: int
    image_height: int
    total_segments: int
    original_image_url: str
    overlay_image_url: str | None = None
    segments: list[SegmentSchema]


class SegmentationRequest(BaseModel):
    page_id: str = Field(..., description="Page ID returned from the upload endpoint")


class UploadResponse(BaseModel):
    page_id: str
    filename: str
    original_image_url: str
    width: int
    height: int
