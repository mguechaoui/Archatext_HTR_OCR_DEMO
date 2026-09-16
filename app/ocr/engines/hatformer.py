"""
`HATFormerOCR` — adapter for the muharaf HATFormer checkpoint.

Preprocessing matches training config (configs/ocr_hatformer.yaml):
  image.height: 64
  image.canvas: 384
  image.rtl_flip: true
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from PIL import Image

from app.config import settings
from app.models import OCREngine
from app.ocr.base import BaseOCR, OCRPrediction
from app.utils.device import resolve_device

if TYPE_CHECKING:
    from app.models.segmentation_result import Segment

logger = logging.getLogger(__name__)


class HATFormerOCR(BaseOCR):
    engine_id = OCREngine.HATFORMER

    IMAGE_HEIGHT = 64
    CANVAS_SIZE = 384
    RTL_FLIP = True

    def __init__(self, checkpoint_path: str | None = None, device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path or settings.hatformer_checkpoint_path)
        self.device = resolve_device(device or settings.device)
        self._model = None
        self._processor = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return

        from transformers import (
            PreTrainedTokenizerFast,
            VisionEncoderDecoderModel,
            ViTImageProcessor,
        )

        logger.info("Loading HATFormer: %s", self.checkpoint_path)

        tokenizer_file = str(self.checkpoint_path / "tokenizer.json")
        self._tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        self._tokenizer.add_special_tokens({
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "cls_token": "<s>",
            "bos_token": "<s>",
        })

        self._processor = ViTImageProcessor.from_pretrained(str(self.checkpoint_path))
        self._model = VisionEncoderDecoderModel.from_pretrained(str(self.checkpoint_path))
        self._model.config.decoder_start_token_id = self._tokenizer.bos_token_id
        self._model.config.pad_token_id = self._tokenizer.pad_token_id
        self._model.config.vocab_size = self._model.config.decoder.vocab_size
        self._model.to(self.device)
        self._model.eval()

        logger.info("HATFormer ready")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size
        ratio = self.IMAGE_HEIGHT / h
        new_w = int(w * ratio)
        image = image.resize((new_w, self.IMAGE_HEIGHT), Image.Resampling.LANCZOS)

        if self.RTL_FLIP:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        canvas = Image.new("RGB", (self.CANVAS_SIZE, self.CANVAS_SIZE), (255, 255, 255))
        canvas.paste(image, (0, 0))

        inputs = self._processor(images=canvas, return_tensors="pt")
        return inputs.pixel_values.to(self.device)

    def predict(self, image: Image.Image, segment: Segment | None = None) -> OCRPrediction:
        if self._model is None:
            self.load()

        pixel_values = self._preprocess(image)

        with torch.no_grad():
            generated_ids = self._model.generate(
                pixel_values=pixel_values,
                num_beams=3,
                length_penalty=0,
                max_new_tokens=64,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        text = self._tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        return OCRPrediction(text=text.strip(), confidence=None)

    def predict_batch(self, images: list[Image.Image]) -> list[OCRPrediction]:
        if self._model is None:
            self.load()

        pixel_values_list = [self._preprocess(img) for img in images]
        pixel_values = torch.cat(pixel_values_list, dim=0)

        with torch.no_grad():
            generated_ids = self._model.generate(
                pixel_values=pixel_values,
                num_beams=3,
                length_penalty=0,
                max_new_tokens=64,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        texts = self._tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        return [OCRPrediction(text=t.strip(), confidence=None) for t in texts]