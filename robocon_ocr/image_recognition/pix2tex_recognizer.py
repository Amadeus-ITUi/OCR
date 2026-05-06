from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass

from PIL import Image

from robocon_ocr.config import OCRConfig
from robocon_ocr.result.expression import normalize_ocr_text, validate_ocr_text


@dataclass(slots=True)
class OCRResult:
    raw_text: str
    confidence: float
    lines: list[str]
    psm: int | None = None
    error: str | None = None


class Pix2TexMathRecognizer:
    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self._engine = None

    def _build_engine(self):
        try:
            from pix2tex.cli import LatexOCR
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "未安装 pix2tex。请先执行 `pip install -r requirements.txt`。"
            ) from exc

        kwargs = self._build_engine_kwargs(LatexOCR)
        try:
            return LatexOCR(**kwargs)
        except TypeError:
            return LatexOCR()

    def _build_engine_kwargs(self, engine_cls) -> dict[str, object]:
        try:
            signature = inspect.signature(engine_cls)
        except (TypeError, ValueError):
            return {}

        if "arguments" not in signature.parameters:
            return {}

        namespace = argparse.Namespace()
        setattr(namespace, "config", "settings/config.yaml")
        setattr(namespace, "checkpoint", self.config.model_path or "checkpoints/weights.pth")
        setattr(namespace, "no_cuda", self.config.device.lower() == "cpu")
        # pix2tex's optional image_resizer path is fragile for some camera-frame inputs
        # and can raise PIL paste mode mismatches. Disable it for stability.
        setattr(namespace, "no_resize", True)
        return {"arguments": namespace}

    @property
    def engine(self):
        if self._engine is None:
            self._engine = self._build_engine()
        return self._engine

    def warmup(self) -> None:
        _ = self.engine

    def recognize_candidates(self, image: Image.Image) -> list[OCRResult]:
        result = self.recognize(image)
        return [result]

    def recognize(self, image: Image.Image) -> OCRResult:
        try:
            raw_text = str(self.engine(image)).strip()
        except Exception as exc:
            raise RuntimeError(f"pix2tex 推理失败: {exc}") from exc

        if not raw_text:
            return OCRResult(
                raw_text="",
                confidence=0.0,
                lines=[],
                psm=None,
                error="no text detected by OCR",
            )

        normalized = normalize_ocr_text(raw_text)
        error = validate_ocr_text(normalized) if self.config.strict_charset else None
        if error is not None:
            return OCRResult(
                raw_text=raw_text,
                confidence=0.0,
                lines=[],
                psm=None,
                error=error,
            )

        return OCRResult(
            raw_text=normalized,
            confidence=1.0,
            lines=[normalized] if normalized else [],
            psm=None,
            error=None if normalized else "no text detected by OCR",
        )
