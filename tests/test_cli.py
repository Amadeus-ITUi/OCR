import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from robocon_ocr.cli import build_argparser, build_config, resolve_label_file


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

    args = build_argparser().parse_args([str(dataset_dir)])
    config = build_config(args)

    assert config.dataset_dir == dataset_dir
    assert config.label_file == manifest
    assert config.debug_dir is None


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
    image_path = tmp_path / "problem_0001.png"
    image = Image.new("RGB", (240, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 40), "5+2=", fill="black")
    image.save(image_path)

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
