from __future__ import annotations

from robocon_ocr.config import OCRConfig
from robocon_ocr.image_recognition.lightweight_recognizer import LightweightMathRecognizer


def create_recognizer(config: OCRConfig):
    backend = config.backend.strip().lower()
    if backend == "lightweight":
        return LightweightMathRecognizer(config)
    raise ValueError(f"unsupported OCR backend: {config.backend}")
