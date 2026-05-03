from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class OCRConfig:
    lang: str = "eng"
    tesseract_cmd: str | None = None
    oem: int = 3
    psm: int = 11
    fallback_psm: tuple[int, ...] = (7,)
    whitelist: str = "0123456789+-xX*/()=×÷"
    preserve_interword_spaces: bool = False


@dataclass(slots=True)
class PreprocessConfig:
    foreground_threshold: int = 200
    min_foreground_pixels: int = 32
    min_crop_size: int = 8
    crop_padding: int = 24
    scale_factor: float = 2.0
    binary_threshold: int = 185


@dataclass(slots=True)
class CameraConfig:
    device_index: int = 2
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    pixel_format: str = "MJPG"
    warmup_frames: int = 5
    capture_timeout_ms: int = 3000
    interval_ms: int = 50
    max_frames: int | None = None
    emit_only_changes: bool = True
    save_frame: Path | None = None


@dataclass(slots=True)
class PipelineConfig:
    dataset_dir: Path
    label_file: Path | None = None
    debug_dir: Path | None = None
    ocr: OCRConfig = field(default_factory=OCRConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
