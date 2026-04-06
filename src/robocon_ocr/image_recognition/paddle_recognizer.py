from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from robocon_ocr.config import OCRConfig


@dataclass(slots=True)
class OCRResult:
    raw_text: str
    confidence: float
    lines: list[str]


class PaddleMathRecognizer:
    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self._engine = None

    def _build_engine(self):
        try:
            from paddleocr import PaddleOCR
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "未安装 paddleocr。请先执行 `pip install -r requirements.txt`。"
            ) from exc

        return PaddleOCR(
            use_angle_cls=self.config.use_angle_cls,
            lang=self.config.lang,
            show_log=self.config.show_log,
            det_limit_side_len=self.config.det_limit_side_len,
            rec_batch_num=self.config.rec_batch_num,
            drop_score=self.config.drop_score,
        )

    @property
    def engine(self):
        if self._engine is None:
            self._engine = self._build_engine()
        return self._engine

    def recognize(self, image: Image.Image) -> OCRResult:
        result = self.engine.ocr(np.asarray(image), cls=self.config.use_angle_cls)
        if not result or not result[0]:
            return OCRResult(raw_text="", confidence=0.0, lines=[])

        lines: list[str] = []
        scores: list[float] = []
        for item in result[0]:
            text, score = item[1]
            lines.append(text)
            scores.append(float(score))

        return OCRResult(
            raw_text=" ".join(lines).strip(),
            confidence=sum(scores) / len(scores),
            lines=lines,
        )

