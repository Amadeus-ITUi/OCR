from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from robocon_ocr.config import OCRConfig


@dataclass(slots=True)
class OCRResult:
    raw_text: str
    confidence: float
    lines: list[str]
    psm: int | None = None
    error: str | None = None


class TesseractMathRecognizer:
    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self._engine = None

    def _build_engine(self):
        try:
            import pytesseract
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "未安装 pytesseract。请先执行 `pip install -r requirements.txt`。"
            ) from exc

        if self.config.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_cmd
        return pytesseract

    @property
    def engine(self):
        if self._engine is None:
            self._engine = self._build_engine()
        return self._engine

    def recognize(self, image: Image.Image) -> OCRResult:
        lang = self._resolve_lang()
        best_result: OCRResult | None = None

        for psm in self._psm_candidates():
            config = self._build_tesseract_config(psm)
            try:
                data = self.engine.image_to_data(
                    image,
                    lang=lang,
                    config=config,
                    output_type=self.engine.Output.DICT,
                )
            except self.engine.TesseractNotFoundError as exc:
                raise RuntimeError(
                    "未找到 tesseract 可执行程序。请先在系统中安装 tesseract-ocr，"
                    "或在 OCRConfig.tesseract_cmd 中指定其路径。"
                ) from exc

            result = self._result_from_data(data, psm)
            if best_result is None or self._is_better_result(result, best_result):
                best_result = result
            if result.raw_text:
                return result

        return best_result or OCRResult(
            raw_text="",
            confidence=0.0,
            lines=[],
            error="no text detected by OCR",
        )

    def _resolve_lang(self) -> str:
        # Tesseract commonly uses "eng" while some OCR libraries use "en".
        if self.config.lang == "en":
            return "eng"
        return self.config.lang

    def _build_tesseract_config(self, psm: int) -> str:
        options = [
            f"--oem {self.config.oem}",
            f"--psm {psm}",
        ]
        if self.config.whitelist:
            options.append(f"-c tessedit_char_whitelist={self.config.whitelist}")
        if self.config.preserve_interword_spaces:
            options.append("-c preserve_interword_spaces=1")
        return " ".join(options)

    def _psm_candidates(self) -> tuple[int, ...]:
        seen: set[int] = set()
        ordered: list[int] = []
        for psm in (self.config.psm, *self.config.fallback_psm):
            if psm in seen:
                continue
            seen.add(psm)
            ordered.append(psm)
        return tuple(ordered)

    def _result_from_data(self, data: dict[str, list[str]], psm: int) -> OCRResult:
        lines: list[str] = []
        scores: list[float] = []
        word_count = len(data.get("text", []))
        for index in range(word_count):
            text = data["text"][index].strip()
            confidence = self._parse_confidence(data["conf"][index])
            if not text:
                continue
            lines.append(text)
            if confidence >= 0.0:
                scores.append(confidence)

        raw_text = " ".join(lines).strip()
        return OCRResult(
            raw_text=raw_text,
            confidence=sum(scores) / len(scores) if scores else 0.0,
            lines=lines,
            psm=psm,
            error=None if raw_text else "no text detected by OCR",
        )

    @staticmethod
    def _is_better_result(candidate: OCRResult, current: OCRResult) -> bool:
        if bool(candidate.raw_text) != bool(current.raw_text):
            return bool(candidate.raw_text)
        if candidate.confidence != current.confidence:
            return candidate.confidence > current.confidence
        return len(candidate.raw_text) > len(current.raw_text)

    @staticmethod
    def _parse_confidence(value: str) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return -1.0
        if confidence < 0:
            return -1.0
        return confidence / 100.0
