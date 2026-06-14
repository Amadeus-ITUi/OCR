from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from robocon_ocr.config import OCRConfig
from robocon_ocr.image_recognition.factory import create_recognizer
from robocon_ocr.image_recognition.lightweight_recognizer import LightweightMathRecognizer


def test_create_recognizer_builds_lightweight_backend():
    recognizer = create_recognizer(OCRConfig(backend="lightweight"))

    assert isinstance(recognizer, LightweightMathRecognizer)


def test_lightweight_recognizer_maps_explicit_gpu_to_gpu_zero():
    recognizer = LightweightMathRecognizer(OCRConfig(backend="lightweight", device="gpu"))

    assert recognizer._resolve_device() == "gpu:0"


def test_lightweight_recognizer_accepts_auto_device():
    recognizer = LightweightMathRecognizer(OCRConfig(backend="lightweight", device="auto"))

    assert recognizer._resolve_device() is None


def test_lightweight_recognizer_disables_hpi_by_default():
    recognizer = LightweightMathRecognizer(OCRConfig(backend="lightweight"))

    assert recognizer._build_engine_kwargs()["enable_hpi"] is False


def test_lightweight_recognizer_normalizes_and_tags_backend():
    recognizer = LightweightMathRecognizer(OCRConfig(backend="lightweight"))
    recognizer._engine = SimpleNamespace(
        predict=lambda **_kwargs: [{"res": {"rec_text": "5/2=", "rec_score": 0.93}}],
    )

    result = recognizer.recognize(Image.new("RGB", (48, 24), "white"))

    assert result.raw_text == "5÷2="
    assert result.confidence == 0.93
    assert result.backend == "lightweight"


def test_lightweight_recognizer_rejects_non_arithmetic_output():
    recognizer = LightweightMathRecognizer(OCRConfig(backend="lightweight"))
    recognizer._engine = SimpleNamespace(
        predict=lambda **_kwargs: [{"res": {"rec_text": "abc", "rec_score": 0.88}}],
    )

    result = recognizer.recognize(Image.new("RGB", (48, 24), "white"))

    assert result.error == "unsupported symbol outside arithmetic charset"
    assert result.backend == "lightweight"
