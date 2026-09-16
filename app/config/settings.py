"""
Centralized application configuration.

Every path, device choice, and tunable parameter lives here and is sourced
from environment variables (or a `.env` file). Nothing in the rest of the
codebase should hardcode a path, a device string, or a batch size — import
`settings` instead.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_DIR = Path(__file__).resolve().parents[2]

# Where model weights live on disk. In the container this is /models, a
# writable volume that `scripts/fetch_models.py` populates from Azure Blob
# Storage before uvicorn starts. Locally it defaults to ./model_weights so a
# developer can just drop checkpoints in and run. Resolved at import time
# because every checkpoint path below is derived from it.
_MODEL_DIR = Path(os.getenv("OCR_MODEL_DIR", str(_BASE_DIR / "models")))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="OCR_",
        extra="ignore",
        # `model_dir` collides with pydantic's reserved "model_" namespace;
        # we want the env var to read OCR_MODEL_DIR, so opt out of the guard.
        protected_namespaces=(),
    )

    # ------------------------------------------------------------------ #
    # General
    # ------------------------------------------------------------------ #
    app_name: str = "Arabic Manuscript OCR Platform"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False

    # ------------------------------------------------------------------ #
    # Directories (all relative to project root unless given as absolute)
    # ------------------------------------------------------------------ #
    base_dir: Path = _BASE_DIR
    model_dir: Path = _MODEL_DIR
    upload_dir: Path = base_dir / "uploads" / "images"
    temp_dir: Path = base_dir / "uploads" / "temp"
    output_dir: Path = base_dir / "outputs"
    static_dir: Path = base_dir / "static"

    # ------------------------------------------------------------------ #
    # Device / performance
    # ------------------------------------------------------------------ #
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    ocr_batch_size: int = 8
    max_image_dimension: int = 6000  # safety cap for absurdly large scans

    # ------------------------------------------------------------------ #
    # Segmentation model (kraken / muharaf)
    # ------------------------------------------------------------------ #
    segmentation_model_path: Path = _MODEL_DIR / "seg" / "muharaf_seg_best.mlmodel"
    segmentation_text_direction: str = "horizontal-rl"

    # ------------------------------------------------------------------ #
    # OCR models
    # ------------------------------------------------------------------ #
    # "MyOCRModel" — fine-tuned kraken recognition checkpoint (ArMan pipeline)
    myocr_checkpoint_path: Path = _MODEL_DIR / "ocr" / "arman_run1_best.mlmodel"

    # HATFormer — HuggingFace-style image-to-text transformer checkpoint.
    # Point this at a local directory or HF hub id; the adapter loads it
    # with transformers' Auto* classes at startup.
    hatformer_checkpoint_path: str = str(_MODEL_DIR / "ocr" / "hatformer")

    # ------------------------------------------------------------------ #
    # Upload constraints
    # ------------------------------------------------------------------ #
    max_upload_size_mb: int = 50
    allowed_image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

    # ------------------------------------------------------------------ #
    # Deployment
    # ------------------------------------------------------------------ #
    # Comma-separated list of OCR engines to actually instantiate, or "*" for
    # all of them. This is the memory lever: HATFormer is a full
    # VisionEncoderDecoder and costs well over a gigabyte of RSS on its own,
    # so a small instance can ship with OCR_ENABLED_OCR_ENGINES=my_ocr_model
    # and still serve the kraken pipeline end to end.
    enabled_ocr_engines: str = "*"

    # Load every model during startup rather than on first request. Keep this
    # on: with scale-to-zero the platform holds the first request until the
    # readiness probe passes, so paying the load cost at boot turns a 60s
    # first request into a 60s cold start that the user never sees mid-call.
    eager_load_models: bool = True

    # CORS origins for a separately-hosted frontend (e.g. the static UI on
    # Vercel talking to this API on Azure). Empty = same-origin only, which
    # is the default because the container serves its own frontend.
    cors_allow_origins: str = ""

    @field_validator("cors_allow_origins", "enabled_ocr_engines", mode="before")
    @classmethod
    def _stringify(cls, v: object) -> str:
        return ",".join(v) if isinstance(v, (list, tuple)) else str(v)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]

    @property
    def enabled_engine_list(self) -> list[str] | None:
        """None means 'no filter — enable everything registered'."""
        if self.enabled_ocr_engines.strip() in ("*", ""):
            return None
        return [e.strip() for e in self.enabled_ocr_engines.split(",") if e.strip()]

    def ensure_directories(self) -> None:
        for d in (self.upload_dir, self.temp_dir, self.output_dir, self.static_dir, self.model_dir):
            d.mkdir(parents=True, exist_ok=True)
        for sub in ("json", "txt", "alto", "page_xml", "csv"):
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings


settings = get_settings()
