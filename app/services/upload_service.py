"""Handles validating and persisting uploaded manuscript page images."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import UploadFile
from PIL import Image

from app.config import settings
from app.core.exceptions import InvalidImageError, UnsupportedFileTypeError
from app.services.page_store import PageStore

logger = logging.getLogger(__name__)


class UploadService:
    def __init__(self, page_store: PageStore):
        self.page_store = page_store

    async def save_upload(self, file: UploadFile) -> tuple[str, Path, Image.Image]:
        """Validate and persist an uploaded file. Returns (page_id, saved_path, opened_image)."""
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in settings.allowed_image_extensions:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{suffix}'. Allowed: {settings.allowed_image_extensions}"
            )

        page_id = uuid.uuid4().hex
        dest_path = settings.upload_dir / f"{page_id}{suffix}"

        contents = await file.read()
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if len(contents) > max_bytes:
            raise InvalidImageError(
                f"File exceeds maximum upload size of {settings.max_upload_size_mb}MB"
            )

        dest_path.write_bytes(contents)

        try:
            image = Image.open(dest_path)
            image.load()
            image = image.convert("RGB")
        except Exception as e:
            dest_path.unlink(missing_ok=True)
            raise InvalidImageError(f"File is not a valid image: {e}") from e

        if max(image.size) > settings.max_image_dimension:
            logger.info("Downscaling oversized image %s from %s", page_id, image.size)
            image.thumbnail((settings.max_image_dimension, settings.max_image_dimension))
            image.save(dest_path)

        self.page_store.create(page_id=page_id, original_image_path=str(dest_path))
        logger.info("Saved upload page_id=%s path=%s size=%s", page_id, dest_path, image.size)
        return page_id, dest_path, image
