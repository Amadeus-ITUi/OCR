from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from robocon_ocr.config import PreprocessConfig


@dataclass(slots=True)
class EnhancementResult:
    denoised: Image.Image
    binary: Image.Image
    prepared_for_ocr: Image.Image


def _import_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("未安装 OpenCV。请先执行 `pip install -r requirements.txt`。") from exc
    return cv2


def enhance_for_ocr(image: Image.Image, config: PreprocessConfig) -> EnhancementResult:
    cv2 = _import_cv2()
    gray = image.convert("L")
    if config.scale_factor != 1.0:
        gray = gray.resize(
            (
                max(1, int(gray.width * config.scale_factor)),
                max(1, int(gray.height * config.scale_factor)),
            ),
            Image.Resampling.LANCZOS,
        )
    arr = np.asarray(gray)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary_image = Image.fromarray(binary.astype(np.uint8), mode="L")
    return EnhancementResult(
        denoised=gray,
        binary=binary_image,
        prepared_for_ocr=binary_image,
    )
