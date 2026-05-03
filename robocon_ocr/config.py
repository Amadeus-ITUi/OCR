from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from robocon_ocr.roi_tuning import DEFAULT_ROI_TUNING_VALUES


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
    white_threshold: int = int(DEFAULT_ROI_TUNING_VALUES["white_threshold"])
    edge_threshold: int = int(DEFAULT_ROI_TUNING_VALUES["edge_threshold"])
    min_roi_area_ratio: float = float(DEFAULT_ROI_TUNING_VALUES["min_roi_area_ratio"])
    rectangularity_min: float = 0.82
    rectangle_ratio_tolerance: float = float(DEFAULT_ROI_TUNING_VALUES["rectangle_ratio_tolerance"])
    quadrilateral_ratio_tolerance: float = float(DEFAULT_ROI_TUNING_VALUES["quadrilateral_ratio_tolerance"])
    target_aspect_ratio: float = float(DEFAULT_ROI_TUNING_VALUES["target_aspect_ratio"])
    roi_padding: int = int(DEFAULT_ROI_TUNING_VALUES["roi_padding"])
    perspective_width: int = int(DEFAULT_ROI_TUNING_VALUES["perspective_width"])
    perspective_height: int = int(DEFAULT_ROI_TUNING_VALUES["perspective_height"])
    scale_factor: float = float(DEFAULT_ROI_TUNING_VALUES["scale_factor"])


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
    async_latest_frame: bool = True
    auto_exposure: int | None = None
    exposure_dynamic_framerate: int | None = None
    exposure_time_absolute: int | None = None
    gain: int | None = None
    brightness: int | None = None
    contrast: int | None = None
    saturation: int | None = None
    sharpness: int | None = None
    gamma: int | None = None
    white_balance_automatic: int | None = None
    white_balance_temperature: int | None = None
    focus_automatic_continuous: int | None = None
    focus_absolute: int | None = None
    backlight_compensation: int | None = None
    power_line_frequency: int | None = None


@dataclass(slots=True)
class PipelineConfig:
    dataset_dir: Path
    label_file: Path | None = None
    debug_dir: Path | None = None
    ocr: OCRConfig = field(default_factory=OCRConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
