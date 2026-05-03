from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from robocon_ocr.config import PreprocessConfig
from robocon_ocr.roi_tuning import DEFAULT_ROI_TUNING_VALUES
from robocon_ocr.image_recognition.preprocess import crop_foreground_text, prepare_for_ocr, prepare_image_for_ocr


def _create_board_image(size=(1280, 720), quad=None) -> Image.Image:
    image = Image.new("RGB", size, (30, 40, 50))
    draw = ImageDraw.Draw(image)
    quad = quad or [(220, 140), (1060, 140), (1060, 612), (220, 612)]
    draw.polygon(quad, fill="white")
    draw.text((460, 320), "5+2=", fill="black")
    return image


def test_crop_foreground_text_finds_white_board_bbox():
    image = _create_board_image()

    cropped = crop_foreground_text(image, PreprocessConfig())

    assert cropped.width < image.width
    assert cropped.height < image.height
    assert cropped.width / cropped.height > 1.5


def test_preprocess_config_defaults_come_from_roi_tuning():
    config = PreprocessConfig()

    assert config.white_threshold == DEFAULT_ROI_TUNING_VALUES["white_threshold"]
    assert config.edge_threshold == DEFAULT_ROI_TUNING_VALUES["edge_threshold"]
    assert config.min_roi_area_ratio == DEFAULT_ROI_TUNING_VALUES["min_roi_area_ratio"]
    assert config.rectangle_ratio_tolerance == DEFAULT_ROI_TUNING_VALUES["rectangle_ratio_tolerance"]
    assert config.quadrilateral_ratio_tolerance == DEFAULT_ROI_TUNING_VALUES["quadrilateral_ratio_tolerance"]


def test_prepare_image_for_ocr_detects_perspective_quad():
    image = _create_board_image(quad=[(180, 190), (1090, 120), (1130, 600), (250, 650)])

    result = prepare_image_for_ocr(image, PreprocessConfig())

    assert result.roi_found is True
    assert result.roi_quad is not None
    assert result.rectified.width == 1280
    assert result.rectified.height == 720
    assert result.roi_debug.best_candidate_type in {"rectangle", "quadrilateral"}
    assert result.roi_debug.failure_reason is None
    assert result.board_binary.mode == "L"
    assert result.prepared.mode == "L"


def test_prepare_for_ocr_outputs_binary_image(tmp_path: Path):
    image_path = tmp_path / "formula.png"
    image = _create_board_image(size=(960, 540))
    image.save(image_path)

    result = prepare_for_ocr(image_path, PreprocessConfig())

    assert result.roi_found is True
    assert set(np.asarray(result.board_binary).reshape(-1)) <= {0, 255}
    assert set(np.asarray(result.prepared).reshape(-1)) <= {0, 255}


def test_prepare_image_for_ocr_reports_missing_roi():
    image = Image.new("RGB", (640, 360), (50, 50, 50))

    result = prepare_image_for_ocr(image, PreprocessConfig())

    assert result.roi_found is False
    assert result.roi_debug.failure_reason in {
        "no contour candidate",
        "area below threshold",
        "edge too weak",
        "rectangle ratio mismatch",
        "quadrilateral ratio mismatch",
        "no valid roi after filtering",
    }
