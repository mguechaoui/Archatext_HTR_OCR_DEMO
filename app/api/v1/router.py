"""Combines every v1 route module into a single router."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import export, ocr, segmentation, segments_assets, status, upload

api_router = APIRouter()
api_router.include_router(upload.router)
api_router.include_router(segmentation.router)
api_router.include_router(segments_assets.router)
api_router.include_router(ocr.router)
api_router.include_router(status.router)
api_router.include_router(export.router)
