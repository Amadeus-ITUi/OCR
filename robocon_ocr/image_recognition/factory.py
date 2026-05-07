from __future__ import annotations

from robocon_ocr.config import OCRConfig
from robocon_ocr.image_recognition.lightweight_recognizer import LightweightMathRecognizer
from robocon_ocr.image_recognition.pix2tex_recognizer import Pix2TexMathRecognizer


def create_recognizer(config: OCRConfig):
    backend = config.backend.strip().lower()
    if backend == "pix2tex":
        return Pix2TexMathRecognizer(config)
    if backend == "lightweight":
        return LightweightMathRecognizer(config)
    raise ValueError(f"unsupported OCR backend: {config.backend}")
