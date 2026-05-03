import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from robocon_ocr.cli import build_argparser, build_camera_config, build_config, resolve_label_file
from robocon_ocr.config import PipelineConfig
from robocon_ocr.pipeline import run_image_pipeline


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


def test_camera_parser_defaults_match_local_camera_setup():
    args = build_argparser().parse_args(["camera"])
    config = build_camera_config(args)

    assert config.device_index == 2
    assert config.width == 1280
    assert config.height == 720
    assert config.fps == 30.0
    assert config.pixel_format == "MJPG"
    assert config.interval_ms == 50


def test_run_image_pipeline_on_synthetic_camera_frame(tmp_path: Path):
    image_path = Path(__file__).resolve().parents[1] / "dataset" / "num_10_com_1" / "problem_0001.png"
    image = Image.open(image_path).convert("RGB")

    record = run_image_pipeline(
        image=image,
        image_name="camera_0.png",
        config=PipelineConfig(dataset_dir=tmp_path),
    )

    assert record.image_name == "camera_0.png"
    assert record.parsed.expression == "5+2"


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_module_cli_uses_auto_detected_label_file(tmp_path: Path):
    source_dir = Path(__file__).resolve().parents[1] / "dataset" / "num_10_com_1"
    shutil.copyfile(source_dir / "problem_0001.png", tmp_path / "problem_0001.png")
    shutil.copyfile(source_dir / "problems_and_answers.txt", tmp_path / "problems_and_answers.txt")

    result = subprocess.run(
        [sys.executable, "-m", "robocon_ocr", str(tmp_path)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert "gt_expression: 5 + 2 =" in result.stdout
    assert "expression_match: True" in result.stdout


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_module_cli_runs_without_label_file(tmp_path: Path):
    source_dir = Path(__file__).resolve().parents[1] / "dataset" / "num_10_com_1"
    shutil.copyfile(source_dir / "problem_0001.png", tmp_path / "problem_0001.png")

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


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_legacy_script_entry_still_works(tmp_path: Path):
    source_dir = Path(__file__).resolve().parents[1] / "dataset" / "num_10_com_1"
    shutil.copyfile(source_dir / "problem_0001.png", tmp_path / "problem_0001.png")
    shutil.copyfile(source_dir / "problems_and_answers.txt", tmp_path / "problems_and_answers.txt")

    result = subprocess.run(
        [sys.executable, "scripts/run_offline_pipeline.py", str(tmp_path)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert "gt_expression: 5 + 2 =" in result.stdout
    assert "[summary]" in result.stdout
