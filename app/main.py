"""
Application entrypoint.

Run with:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.config import settings
from app.core.error_handlers import register_exception_handlers
from app.core.lifespan import lifespan
from app.utils.logging import configure_logging

configure_logging(level=logging.DEBUG if settings.debug else logging.INFO)

app = FastAPI(
    title=settings.app_name,
    description="OCR platform for historical Arabic manuscript analysis: "
    "segmentation + pluggable OCR engines.",
    version="1.0.0",
    lifespan=lifespan,
)

register_exception_handlers(app)

# Only mounted when a separately-hosted frontend needs it (e.g. the static UI
# on Vercel calling this API on Azure). Default is empty: the container serves
# its own frontend from the same origin, so no CORS is required at all.
if settings.cors_origin_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=settings.api_v1_prefix)

@app.get("/health", tags=["ops"])
async def health_check() -> dict:
    """Liveness: the process is up and the event loop is responsive.

    Deliberately says nothing about models — a liveness probe that fails on a
    missing checkpoint makes the orchestrator restart-loop a container that
    restarting will never fix.
    """
    return {"status": "ok"}


@app.get("/health/ready", tags=["ops"])
async def readiness_check(response: Response) -> dict:
    """Readiness: is this replica actually able to do the job?

    Returns 503 until the segmentation model is loaded, because segmentation
    is the entry point of every pipeline in this app — an instance without it
    can serve the UI but cannot serve a request. OCR engines are reported but
    do not gate readiness: the registry is explicitly designed to degrade to
    'this engine 503s, the rest of the platform works'.
    """
    seg_ready = bool(getattr(app.state, "segmentation_ready", False))
    available = list(getattr(app.state, "ocr_available", []))
    registered = list(getattr(app.state, "ocr_registered", []))

    if not seg_ready:
        response.status_code = 503

    return {
        "status": "ready" if seg_ready else "not_ready",
        "segmentation": "loaded" if seg_ready else "unavailable",
        "model_dir": str(settings.model_dir),
        "device": settings.device,
        "ocr_engines": {
            "registered": registered,
            "available": available,
            "degraded": sorted(set(registered) - set(available)),
        },
    }


@app.get("/version", tags=["ops"])
async def version_info() -> dict:
    """Surfaces the build and the weights version together.

    These two version independently by design — the image is weights-free and
    the checkpoints are pulled at boot — so debugging a bad prediction starts
    with knowing which pair actually produced it.
    """
    import os

    return {
        "app": settings.app_name,
        "api_version": app.version,
        "git_sha": os.getenv("GIT_SHA", "unknown"),
        "image_tag": os.getenv("IMAGE_TAG", "unknown"),
        "model_version": os.getenv("OCR_MODEL_VERSION", "unset"),
    }


# ---------------------------------------------------------------------------
# Static mounts MUST be registered last.
#
# Starlette matches routes in registration order, and a Mount at "/" matches
# *every* path. Registering it before the ops endpoints means StaticFiles
# swallows /health and /health/ready and answers 404 — which reads to a
# platform health probe as "this container is dead", and it restart-loops a
# perfectly healthy app. Mounting after the API router and the ops routes is
# what keeps them reachable.
# ---------------------------------------------------------------------------

# Serve uploaded originals so the frontend can preview them directly.
app.mount("/static/uploads", StaticFiles(directory=str(settings.upload_dir)), name="uploads")
# Serve the frontend itself.
app.mount("/", StaticFiles(directory=str(settings.static_dir), html=True), name="frontend")
