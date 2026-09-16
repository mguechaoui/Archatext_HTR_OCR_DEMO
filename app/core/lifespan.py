"""
Application lifespan.

This is where the "load models once at startup, not per request"
requirement is implemented. FastAPI's lifespan context manager runs once
when the app starts and once when it shuts down — everything expensive
(the segmentation model, every OCR engine) is loaded here, before the app
starts accepting requests.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.ocr.registry import init_ocr_engines, list_available_engines, list_registered_engines
from app.segmentation.registry import init_segmentation_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading models from %s ...", settings.model_dir)

    try:
        init_segmentation_service()
        app.state.segmentation_ready = True
    except Exception as e:
        logger.warning("Segmentation model not loaded at startup: %s", e)
        app.state.segmentation_ready = False

    init_ocr_engines(eager=settings.eager_load_models)

    available = [e.value for e in list_available_engines()]
    registered = [e.value for e in list_registered_engines()]
    logger.info("OCR engines registered=%s available=%s", registered, available)

    # Recorded on app.state so /health/ready can answer without re-probing
    # the registries on every probe request.
    app.state.ocr_available = available
    app.state.ocr_registered = registered

    yield

    logger.info("Shutting down.")
