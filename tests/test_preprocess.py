from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from robocon_ocr.config import PreprocessConfig
from robocon_ocr.image_recognition.preprocess import crop_foreground_text, prepare_for_ocr


def test_crop_foreground_text_finds_dark_formula_bbox():
    image = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((160, 70, 240, 130), fill="black")

    cropped = crop_foreground_text(image, PreprocessConfig(crop_padding=10))

    assert cropped.width < image.width
    assert cropped.height < image.height
    assert cropped.width >= 100
    assert cropped.height >= 80


def test_crop_foreground_text_falls_back_for_blank_image():
    image = Image.new("RGB", (320, 120), "white")

    cropped = crop_foreground_text(image, PreprocessConfig())

    assert cropped.size == image.size


def test_prepare_for_ocr_outputs_binary_image(tmp_path: Path):
    image_path = tmp_path / "formula.png"
    image = Image.new("RGB", (240, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 40), "5+2=", fill="black")
    image.save(image_path)

    cropped, prepared = prepare_for_ocr(image_path, PreprocessConfig())

    assert cropped.width <= image.width
    assert prepared.mode == "L"
    assert set(np.asarray(prepared).reshape(-1)) == {0, 255}
