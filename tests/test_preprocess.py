from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from robocon_ocr.config import PreprocessConfig
from robocon_ocr.roi_tuning import DEFAULT_ROI_TUNING_VALUES
from robocon_ocr.image_recognition.preprocess import crop_foreground_text, prepare_for_ocr, prepare_image_for_ocr
from robocon_ocr.vision_processing.expression_region import _scan_boundary_backward
from robocon_ocr.vision_processing.expression_region import extract_expression_region
from robocon_ocr.vision_processing.enhancement import enhance_for_ocr


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
    assert config.component_white_threshold == DEFAULT_ROI_TUNING_VALUES["component_white_threshold"]
    assert config.component_min_area_ratio == DEFAULT_ROI_TUNING_VALUES["component_min_area_ratio"]
    assert config.component_fill_ratio_threshold == DEFAULT_ROI_TUNING_VALUES["component_fill_ratio_threshold"]
    assert config.edge_threshold == DEFAULT_ROI_TUNING_VALUES["edge_threshold"]
    assert config.min_roi_area_ratio == DEFAULT_ROI_TUNING_VALUES["min_roi_area_ratio"]
    assert config.rectangle_ratio_tolerance == DEFAULT_ROI_TUNING_VALUES["rectangle_ratio_tolerance"]
    assert config.quadrilateral_ratio_tolerance == DEFAULT_ROI_TUNING_VALUES["quadrilateral_ratio_tolerance"]
    assert config.expression_search_top_ratio == DEFAULT_ROI_TUNING_VALUES["expression_search_top_ratio"]
    assert config.expression_search_bottom_ratio == DEFAULT_ROI_TUNING_VALUES["expression_search_bottom_ratio"]
    assert config.expression_search_left_ratio == DEFAULT_ROI_TUNING_VALUES["expression_search_left_ratio"]
    assert config.expression_search_right_ratio == DEFAULT_ROI_TUNING_VALUES["expression_search_right_ratio"]
    assert config.expression_otsu_bias == DEFAULT_ROI_TUNING_VALUES["expression_otsu_bias"]
    assert config.expression_enter_ratio == DEFAULT_ROI_TUNING_VALUES["expression_enter_ratio"]
    assert config.expression_exit_ratio == DEFAULT_ROI_TUNING_VALUES["expression_exit_ratio"]
    assert config.expression_min_consecutive_rows == DEFAULT_ROI_TUNING_VALUES["expression_min_consecutive_rows"]
    assert config.expression_min_consecutive_cols == DEFAULT_ROI_TUNING_VALUES["expression_min_consecutive_cols"]
    assert config.expression_bbox_padding_x == DEFAULT_ROI_TUNING_VALUES["expression_bbox_padding_x"]
    assert config.expression_bbox_padding_y == DEFAULT_ROI_TUNING_VALUES["expression_bbox_padding_y"]
    assert config.expression_bbox_expand_ratio_x == DEFAULT_ROI_TUNING_VALUES["expression_bbox_expand_ratio_x"]
    assert config.expression_bbox_expand_ratio_y == DEFAULT_ROI_TUNING_VALUES["expression_bbox_expand_ratio_y"]
    assert config.enhance_contrast_clip_limit == DEFAULT_ROI_TUNING_VALUES["enhance_contrast_clip_limit"]
    assert config.enhance_contrast_tile_grid_size == DEFAULT_ROI_TUNING_VALUES["enhance_contrast_tile_grid_size"]
    assert config.enhance_remove_noise_area_min == DEFAULT_ROI_TUNING_VALUES["enhance_remove_noise_area_min"]


def test_prepare_image_for_ocr_detects_perspective_quad():
    image = _create_board_image(quad=[(180, 190), (1090, 120), (1130, 600), (250, 650)])

    result = prepare_image_for_ocr(image, PreprocessConfig())

    assert result.roi_found is True
    assert result.roi_quad is not None
    assert result.rectified.width == 1280
    assert result.rectified.height == 720
    assert result.roi_debug.best_candidate_type == "rectangle"
    assert result.roi_debug.best_candidate_source == "rect_contour"
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
        "rectangle fill too low",
        "corners not rectangular",
        "no valid roi after filtering",
    }


def test_prepare_image_for_ocr_uses_rect_contour_source_for_clear_white_board():
    image = _create_board_image()

    result = prepare_image_for_ocr(image, PreprocessConfig())

    assert result.roi_found is True
    assert result.roi_debug.best_candidate_source == "rect_contour"
    assert result.roi_debug.component_count == 0


def test_extract_expression_region_ignores_black_frame():
    image = Image.new("RGB", (1280, 720), (210, 235, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 1239, 679), outline="black", width=18)
    draw.text((430, 300), "11+(4x8)=", fill="black")

    result = extract_expression_region(image, PreprocessConfig())

    assert result.region_found is True
    assert result.bbox is not None
    x0, y0, x1, y1 = result.bbox
    assert x0 > 80
    assert y0 > 80
    assert x1 < 1200
    assert y1 < 660


def test_extract_expression_region_uses_otsu_on_blue_tinted_board():
    image = Image.new("RGB", (1280, 720), (185, 220, 238))
    draw = ImageDraw.Draw(image)
    draw.text((360, 280), "11 + (4 x 8) =", fill=(20, 20, 20))

    result = extract_expression_region(image, PreprocessConfig())

    assert result.region_found is True
    assert result.otsu_threshold is not None
    assert result.bbox is not None
    assert result.cropped_region is not None
    assert result.cropped_region.width < image.width
    assert result.cropped_region.height < image.height


def test_enhance_for_ocr_applies_local_contrast_without_breaking_binary_output():
    image = Image.new("L", (96, 48), 205)
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 12, 76, 34), fill=170)
    draw.rectangle((24, 16, 30, 30), fill=110)
    draw.rectangle((42, 16, 48, 30), fill=110)
    draw.rectangle((24, 22, 48, 24), fill=110)

    result = enhance_for_ocr(
        image.convert("RGB"),
        PreprocessConfig(
            scale_factor=1.0,
            enhance_contrast_clip_limit=3.0,
            enhance_contrast_tile_grid_size=4,
        ),
    )

    assert result.denoised.mode == "L"
    assert result.binary.mode == "L"
    assert set(np.asarray(result.prepared_for_ocr).reshape(-1)) <= {0, 255}
    assert np.asarray(result.denoised).std() > np.asarray(image).std()


def test_enhance_for_ocr_removes_tiny_foreground_noise():
    image = Image.new("L", (96, 48), 220)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 12, 72, 34), fill=60)
    draw.rectangle((6, 6, 6, 6), fill=40)
    draw.rectangle((88, 10, 88, 10), fill=40)
    draw.rectangle((10, 40, 10, 40), fill=40)

    result = enhance_for_ocr(
        image.convert("RGB"),
        PreprocessConfig(
            scale_factor=1.0,
            enhance_contrast_clip_limit=0.0,
            enhance_remove_noise_area_min=4,
        ),
    )

    prepared = np.asarray(result.prepared_for_ocr)
    assert prepared[6, 6] == 255
    assert prepared[10, 88] == 255
    assert prepared[40, 10] == 255
    assert prepared[20, 30] == 0


def test_extract_expression_region_expands_final_bbox_by_ratio():
    image = Image.new("RGB", (1280, 720), (210, 235, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((420, 300, 860, 360), fill="black")

    no_expand = extract_expression_region(
        image,
        PreprocessConfig(
            expression_bbox_padding_x=0,
            expression_bbox_padding_y=0,
            expression_bbox_expand_ratio_x=0.0,
            expression_bbox_expand_ratio_y=0.0,
        ),
    )
    expanded = extract_expression_region(
        image,
        PreprocessConfig(
            expression_bbox_padding_x=0,
            expression_bbox_padding_y=0,
            expression_bbox_expand_ratio_x=0.10,
            expression_bbox_expand_ratio_y=0.10,
        ),
    )

    assert no_expand.region_found is True
    assert expanded.region_found is True
    assert no_expand.bbox is not None
    assert expanded.bbox is not None
    base_x0, base_y0, base_x1, base_y1 = no_expand.bbox
    expanded_x0, expanded_y0, expanded_x1, expanded_y1 = expanded.bbox
    assert expanded_x0 < base_x0
    assert expanded_y0 < base_y0
    assert expanded_x1 > base_x1
    assert expanded_y1 > base_y1


def test_extract_expression_region_keeps_long_expression_near_board_edges():
    image = Image.new("RGB", (1280, 720), (210, 235, 245))
    draw = ImageDraw.Draw(image)
    for x0 in (105, 220, 335, 450, 565, 680, 795, 910, 1025, 1140):
        draw.rectangle((x0, 290, x0 + 45, 380), fill="black")

    result = extract_expression_region(image, PreprocessConfig())

    assert result.region_found is True
    assert result.failure_reason is None
    assert result.bbox is not None
    x0, y0, x1, y1 = result.bbox
    assert x0 <= 105
    assert x1 >= 1185
    assert (y1 - y0) < 180
    assert result.cropped_region is not None
    assert result.cropped_region.width < image.width
    assert result.cropped_region.height < image.height


def test_scan_boundary_backward_stops_near_foreground_end():
    values = np.array([0.0, 0.0, 0.0, 0.02, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    boundary = _scan_boundary_backward(
        values,
        enter_threshold=0.01,
        exit_threshold=0.003,
        min_consecutive=3,
    )

    assert boundary == 6


def test_extract_expression_region_reports_row_top_failure_on_empty_board():
    image = Image.new("RGB", (1280, 720), (210, 235, 245))

    result = extract_expression_region(image, PreprocessConfig())

    assert result.region_found is False
    assert result.failure_reason == "row top not found"
