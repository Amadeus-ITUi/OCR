from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from robocon_ocr.config import PreprocessConfig


def crop_display_area(image: Image.Image, config: PreprocessConfig) -> Image.Image:
    gray = np.asarray(image.convert("L"))
    mask = gray >= config.white_threshold
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return image

    x0 = max(0, int(xs.min()) - config.crop_padding)
    y0 = max(0, int(ys.min()) - config.crop_padding)
    x1 = min(image.width, int(xs.max()) + config.crop_padding + 1)
    y1 = min(image.height, int(ys.max()) + config.crop_padding + 1)
    return image.crop((x0, y0, x1, y1))


def enhance_for_ocr(image: Image.Image, config: PreprocessConfig) -> Image.Image:
    gray = image.convert("L")
    enlarged = gray.resize(
        (
            int(gray.width * config.scale_factor),
            int(gray.height * config.scale_factor),
        )
    )
    arr = np.asarray(enlarged)
    binary = np.where(arr > config.binary_threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def prepare_for_ocr(image_path: Path, config: PreprocessConfig) -> tuple[Image.Image, Image.Image]:
    original = Image.open(image_path).convert("RGB")
    cropped = crop_display_area(original, config)
    prepared = enhance_for_ocr(cropped, config)
    return cropped, prepared


def save_debug_images(
    image_name: str,
    cropped: Image.Image,
    prepared: Image.Image,
    debug_dir: Path,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    cropped.save(debug_dir / f"{Path(image_name).stem}_cropped.png")
    prepared.save(debug_dir / f"{Path(image_name).stem}_prepared.png")

