"""
OCR plugin registry.

This is the single place that knows about concrete `BaseOCR` subclasses.
Everything else in the app (API routes, the OCR service) depends only on
`BaseOCR` and asks this registry for "the engine called X" — it never
imports `MyOCRModel` or `HATFormerOCR` directly.

To add a new OCR engine:
    1. Implement a new `BaseOCR` subclass under `app/ocr/engines/`.
    2. Add one line to `_ENGINE_CLASSES` below.
That's the entire integration surface.
"""
from __future__ import annotations

import logging

from app.config import settings
from app.models import OCREngine
from app.ocr.base import BaseOCR
from app.ocr.engines.hatformer import HATFormerOCR
from app.ocr.engines.my_ocr_model import MyOCRModel

logger = logging.getLogger(__name__)

_ENGINE_CLASSES: dict[OCREngine, type[BaseOCR]] = {
    OCREngine.MY_OCR_MODEL: MyOCRModel,
    OCREngine.HATFORMER: HATFormerOCR,
}

_instances: dict[OCREngine, BaseOCR] = {}


def _selected_engines() -> dict[OCREngine, type[BaseOCR]]:
    """Filter the registry by OCR_ENABLED_OCR_ENGINES.

    Deployment concern, not an architectural one: on a memory-constrained
    instance we want to ship the same image everywhere and choose per
    environment which engines it actually instantiates. An engine that is
    excluded here is simply absent from the registry, so it never appears in
    the frontend's engine list and never allocates.
    """
    selected = settings.enabled_engine_list
    if selected is None:
        return dict(_ENGINE_CLASSES)

    wanted = {s.lower() for s in selected}
    filtered = {k: v for k, v in _ENGINE_CLASSES.items() if k.value.lower() in wanted}
    unknown = wanted - {k.value.lower() for k in _ENGINE_CLASSES}
    if unknown:
        logger.warning("OCR_ENABLED_OCR_ENGINES names unknown engine(s): %s", sorted(unknown))
    if not filtered:
        logger.error("OCR_ENABLED_OCR_ENGINES matched no engines; no OCR will be available")
    return filtered


def init_ocr_engines(eager: bool = True) -> dict[OCREngine, BaseOCR]:
    """Instantiate (and optionally load) every registered engine once, at
    application startup, so requests never pay model-loading cost."""
    for engine_id, engine_cls in _selected_engines().items():
        if engine_id in _instances:
            continue
        instance = engine_cls()
        if eager:
            try:
                instance.load()
            except Exception as e:
                # Don't crash the whole app if one engine fails to load --
                # missing checkpoint, incompatible library version, bad
                # weights, whatever. Log it and let that engine 503 on use
                # instead, so the rest of the platform stays usable.
                logger.warning("Engine '%s' not loaded: %s", engine_id.value, e)
        _instances[engine_id] = instance
    return _instances


def get_ocr_engine(engine_id: OCREngine) -> BaseOCR:
    if engine_id not in _instances:
        raise RuntimeError(
            f"OCR engine '{engine_id.value}' not initialized. "
            "init_ocr_engines() must run during app startup."
        )
    return _instances[engine_id]


def list_available_engines() -> list[OCREngine]:
    """Engines that are registered AND successfully loaded — i.e. safe to
    use right now."""
    return [engine_id for engine_id, inst in _instances.items() if inst.is_loaded]


def list_registered_engines() -> list[OCREngine]:
    """All engines enabled in this deployment, regardless of load state."""
    return list(_instances.keys()) or list(_selected_engines().keys())
