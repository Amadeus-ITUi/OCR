import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from robocon_ocr.camera_tuning import DEFAULT_CAMERA_TUNING
from robocon_ocr.cli import (
    LatestFrameBuffer,
    DisplayState,
    _build_status_panel,
    _build_text_panel,
    _format_roi_debug_lines,
    _fit_panel_preserve_aspect,
    build_argparser,
    build_camera_config,
    build_camera_pipeline_config,
    build_config,
    resolve_label_file,
)
from robocon_ocr.config import PipelineConfig
from robocon_ocr.image_recognition.base import OCRResult
from robocon_ocr.image_recognition.preprocess import prepare_image_for_ocr
from robocon_ocr.pipeline import run_image_pipeline
from robocon_ocr.result.expression import ParsedExpression
from robocon_ocr.result.reporter import PipelineRecord
from robocon_ocr.staged_pipeline import PipelineContext


def _create_board_image(size=(1280, 720), quad=None) -> Image.Image:
    image = Image.new("RGB", size, (30, 40, 50))
    draw = ImageDraw.Draw(image)
    quad = quad or [(220, 140), (1060, 140), (1060, 612), (220, 612)]
    draw.polygon(quad, fill="white")
    draw.text((460, 320), "5+2=", fill="black")
    return image


def test_resolve_label_file_prefers_explicit_path(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    explicit = tmp_path / "custom.tsv"
    explicit.write_text("filename\texpression\tanswer\tfont_size_px\n", encoding="utf-8")

    assert resolve_label_file(dataset_dir, explicit) == explicit


def test_build_config_auto_detects_manifest(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    manifest = dataset_dir / "problems_and_answers.txt"
    manifest.write_text("filename\texpression\tanswer\tfont_size_px\n", encoding="utf-8")

    args = build_argparser().parse_args(["dataset", str(dataset_dir)])
    config = build_config(args)

    assert config.dataset_dir == dataset_dir
    assert config.label_file == manifest
    assert config.debug_dir is None
    assert config.stop_after_stage is None
    assert config.debug_save_stages is False
    assert config.ocr.backend == "lightweight"


def test_build_config_supports_stage_stop_and_stage_debug_dir(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    debug_dir = tmp_path / "debug"

    args = build_argparser().parse_args(
        [
            "dataset",
            str(dataset_dir),
            "--debug-dir",
            str(debug_dir),
            "--stop-after-stage",
            "expression_region",
            "--debug-save-stages",
        ]
    )
    config = build_config(args)

    assert config.debug_dir == debug_dir
    assert config.stop_after_stage == "expression_region"
    assert config.debug_save_stages is True


def test_build_camera_pipeline_config_defaults_to_lightweight_backend():
    args = build_argparser().parse_args(["camera"])

    config = build_camera_pipeline_config(args)

    assert config.ocr.backend == "lightweight"


def test_build_config_accepts_explicit_ocr_backend_override(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    args = build_argparser().parse_args(["dataset", str(dataset_dir), "--ocr-backend", "lightweight"])

    config = build_config(args)

    assert config.ocr.backend == "lightweight"


def test_build_camera_config_from_args():
    args = build_argparser().parse_args(
        [
            "camera",
            "--device-index",
            "2",
            "--width",
            "1280",
            "--height",
            "720",
            "--fps",
            "60",
            "--pixel-format",
            "YUYV",
            "--warmup-frames",
            "8",
            "--capture-timeout-ms",
            "5000",
            "--interval-ms",
            "120",
            "--max-frames",
            "20",
            "--print-all",
            "--show-window",
            "--window-scale",
            "0.5",
            "--save-frame",
            "captures/frame.png",
        ]
    )

    config = build_camera_config(args)

    assert config.device_index == 2
    assert config.width == 1280
    assert config.height == 720
    assert config.fps == 60
    assert config.pixel_format == "YUYV"
    assert config.warmup_frames == 8
    assert config.capture_timeout_ms == 5000
    assert config.interval_ms == 120
    assert config.max_frames == 20
    assert config.emit_only_changes is False
    assert config.save_frame == Path("captures/frame.png")
    assert config.async_latest_frame is True
    assert config.exposure_time_absolute == DEFAULT_CAMERA_TUNING.exposure_time_absolute
    assert config.focus_absolute == DEFAULT_CAMERA_TUNING.focus_absolute


def test_fit_panel_preserve_aspect_adds_letterbox_instead_of_stretching():
    image = np.full((100, 300, 3), 255, dtype=np.uint8)

    panel = _fit_panel_preserve_aspect(image, (200, 200))

    assert panel.shape == (200, 200, 3)
    assert np.all(panel[0:60] == 0)
    assert np.all(panel[140:200] == 0)
    assert np.all(panel[70:130, 20:180] == 255)


def test_build_text_panel_supports_unicode_math_symbols():
    panel = _build_text_panel(
        ["Pipeline Status", "Raw:", "2×(9+19)×20÷5="],
        (640, 240),
    )

    assert panel.shape == (240, 640, 3)
    assert panel.mean() > 0


def test_build_status_panel_renders_expression_and_answer_summary():
    display = DisplayState(
        context=PipelineContext(
            image_name="camera_0.png",
            original=Image.new("RGB", (1280, 720), "white"),
            preprocess_config=PipelineConfig(dataset_dir=Path(".")).preprocess,
            ocr_config=PipelineConfig(dataset_dir=Path(".")).ocr,
        ),
        record=PipelineRecord(
            image_name="camera_0.png",
            ocr=OCRResult(raw_text="2×(9+19)×20÷5=", confidence=0.97, lines=["2×(9+19)×20÷5="], backend="lightweight"),
            parsed=ParsedExpression(
                normalized_text="2×(9+19)×20÷5=",
                expression="2×(9+19)×20÷5",
                answer=224,
                is_valid=True,
            ),
            label=None,
            roi_found=True,
        ),
        frame_index=1,
        stage_ms=12.3,
    )

    panel = _build_status_panel(display, (640, 360))

    assert panel.shape == (360, 640, 3)
    assert panel.mean() > 0


def test_camera_parser_defaults_match_local_camera_setup():
    args = build_argparser().parse_args(["camera"])
    config = build_camera_config(args)

    assert config.device_index == DEFAULT_CAMERA_TUNING.device_index
    assert config.width == DEFAULT_CAMERA_TUNING.width
    assert config.height == DEFAULT_CAMERA_TUNING.height
    assert config.fps == DEFAULT_CAMERA_TUNING.fps
    assert config.pixel_format == "MJPG"
    assert config.interval_ms == DEFAULT_CAMERA_TUNING.interval_ms
    assert config.auto_exposure == DEFAULT_CAMERA_TUNING.auto_exposure


def test_default_camera_tuning_uses_mjpg():
    assert DEFAULT_CAMERA_TUNING.pixel_format == "MJPG"


def test_latest_frame_buffer_keeps_only_newest_frame():
    buffer = LatestFrameBuffer()
    buffer.publish(0, np.zeros((4, 4, 3), dtype=np.uint8))
    buffer.publish(1, np.ones((4, 4, 3), dtype=np.uint8))

    frame_index, frame_bgr, _captured_at = buffer.wait_for_next(-1)

    assert frame_index == 1
    assert int(frame_bgr.mean()) == 1


def test_run_image_pipeline_on_synthetic_camera_frame(tmp_path: Path):
    class StubRecognizer:
        def __init__(self, config) -> None:
            self.config = config

        def warmup(self) -> None:
            return None

        def recognize(self, image):
            return OCRResult(raw_text="5+2=", confidence=1.0, lines=["5+2="])

    image = _create_board_image()

    import robocon_ocr.pipeline as pipeline_module

    original = pipeline_module.create_recognizer
    pipeline_module.create_recognizer = lambda _config: StubRecognizer(_config)
    try:
        record = run_image_pipeline(
            image=image,
            image_name="camera_0.png",
            config=PipelineConfig(dataset_dir=tmp_path),
        )
    finally:
        pipeline_module.create_recognizer = original

    assert record.image_name == "camera_0.png"
    assert record.roi_found is True
    assert record.parsed.expression == "5+2"


def test_run_image_pipeline_without_roi_returns_invalid_record(tmp_path: Path):
    image = Image.new("RGB", (640, 360), (20, 20, 20))

    record = run_image_pipeline(
        image=image,
        image_name="camera_1.png",
        config=PipelineConfig(dataset_dir=tmp_path),
    )

    assert record.roi_found is False
    assert record.parsed.is_valid is False
    assert record.parsed.error == "roi not found"


def test_roi_debug_lines_render_thresholds_and_reason(tmp_path: Path):
    image = Image.new("RGB", (640, 360), (20, 20, 20))
    preprocess = prepare_image_for_ocr(image, PipelineConfig(dataset_dir=tmp_path).preprocess)

    lines = _format_roi_debug_lines(preprocess.roi_debug)

    assert any(line.startswith("Reason:") for line in lines)
    assert any(line.startswith("Area:") for line in lines)
    assert any(line.startswith("Edge:") for line in lines)
    assert any(line.startswith("WhiteThr:") for line in lines)


@pytest.mark.skipif(importlib.util.find_spec("paddleocr") is None, reason="paddleocr not installed")
def test_module_cli_uses_auto_detected_label_file(tmp_path: Path):
    _create_board_image().save(tmp_path / "problem_0001.png")
    (tmp_path / "problems_and_answers.txt").write_text(
        "filename\texpression\tanswer\tfont_size_px\nproblem_0001.png\t5 + 2 =\t7\t72\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "robocon_ocr", str(tmp_path)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert "gt_expression: 5 + 2 =" in result.stdout
    assert "[summary]" in result.stdout


@pytest.mark.skipif(importlib.util.find_spec("paddleocr") is None, reason="paddleocr not installed")
def test_module_cli_runs_without_label_file(tmp_path: Path):
    _create_board_image().save(tmp_path / "problem_0001.png")

    result = subprocess.run(
        [sys.executable, "-m", "robocon_ocr", str(tmp_path)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert "[problem_0001.png]" in result.stdout
    assert "gt_expression:" not in result.stdout
    assert "[summary]" in result.stdout


@pytest.mark.skipif(importlib.util.find_spec("paddleocr") is None, reason="paddleocr not installed")
def test_legacy_script_entry_still_works(tmp_path: Path):
    _create_board_image().save(tmp_path / "problem_0001.png")
    (tmp_path / "problems_and_answers.txt").write_text(
        "filename\texpression\tanswer\tfont_size_px\nproblem_0001.png\t5 + 2 =\t7\t72\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/run_offline_pipeline.py", str(tmp_path)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert "gt_expression: 5 + 2 =" in result.stdout
    assert "[summary]" in result.stdout
