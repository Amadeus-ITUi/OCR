from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class OCRConfig:
    use_angle_cls: bool = False
    lang: str = "en"
    show_log: bool = False
    det_limit_side_len: int = 1920
    rec_batch_num: int = 6
    drop_score: float = 0.1


@dataclass(slots=True)
class PreprocessConfig:
    white_threshold: int = 235
    crop_padding: int = 24
    scale_factor: float = 2.0
    binary_threshold: int = 185


@dataclass(slots=True)
class PipelineConfig:
    dataset_dir: Path
    label_file: Path | None = None
    debug_dir: Path | None = None
    ocr: OCRConfig = field(default_factory=OCRConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

