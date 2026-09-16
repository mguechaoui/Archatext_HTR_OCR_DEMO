"""Pydantic schemas for status tracking and export endpoints."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class StageStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class PageStatusResponse(BaseModel):
    page_id: str
    upload: StageStatus
    segmentation: StageStatus
    ocr: StageStatus
    ocr_engine_used: str | None = None
    error_message: str | None = None


class ExportFormat(str, Enum):
    TXT = "txt"
    JSON = "json"
    ALTO_XML = "alto_xml"
    PAGE_XML = "page_xml"
    CSV = "csv"
