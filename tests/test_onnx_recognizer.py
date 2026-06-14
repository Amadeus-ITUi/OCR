import numpy as np
import pytest
from PIL import Image

from robocon_ocr.config import OCRConfig
from robocon_ocr.image_recognition.factory import create_recognizer
from robocon_ocr.image_recognition.onnx_recognizer import OnnxMathRecognizer


@pytest.fixture
def onnx_config():
    return OCRConfig(
        backend="onnx",
        onnx_model_path="models/PP-OCRv5_server_rec.onnx",
        onnx_dict_path="models/dict.txt",
        strict_charset=True,
    )


@pytest.fixture
def recognizer(onnx_config):
    return OnnxMathRecognizer(onnx_config)


class TestOnnxRecognizerInit:
    def test_factory_creates_onnx_backend(self, onnx_config):
        rec = create_recognizer(onnx_config)
        assert isinstance(rec, OnnxMathRecognizer)
        assert rec.backend_name == "onnx"
        assert rec.supports_fallback_variants is False

    def test_warmup_loads_session_and_dict(self, recognizer):
        recognizer.warmup()
        assert recognizer._session is not None
        assert recognizer._char_dict is not None
        assert len(recognizer.char_dict) == 18383

    def test_char_dict_first_is_blank(self, recognizer):
        recognizer.warmup()
        assert recognizer.char_dict[0] == "　"


class TestCTCGreedyDecode:
    def test_single_char_decode(self, recognizer):
        recognizer.warmup()
        idx_2 = recognizer.char_dict.index("2")  # 16164
        logits = np.zeros((1, 3, 18385), dtype=np.float32)
        logits[0, :, idx_2 + 1] = 1.0  # class_id = dict_idx + 1
        text, conf = recognizer._ctc_greedy_decode(logits)
        assert text == "2"
        assert conf > 0.9

    def test_blank_filtration(self, recognizer):
        recognizer.warmup()
        idx_5 = recognizer.char_dict.index("5")
        logits = np.zeros((1, 5, 18385), dtype=np.float32)
        # blank, 5, blank, 5, blank — blanks separate distinct characters
        logits[0, 0, 0] = 1.0
        logits[0, 1, idx_5 + 1] = 1.0
        logits[0, 2, 0] = 1.0
        logits[0, 3, idx_5 + 1] = 1.0
        logits[0, 4, 0] = 1.0
        text, _ = recognizer._ctc_greedy_decode(logits)
        assert text == "55"  # blank separates, so two distinct '5' chars

    def test_duplicate_merge(self, recognizer):
        recognizer.warmup()
        idx_a = recognizer.char_dict.index("+")
        logits = np.zeros((1, 4, 18385), dtype=np.float32)
        logits[0, 0, idx_a + 1] = 1.0
        logits[0, 1, idx_a + 1] = 1.0
        logits[0, 2, idx_a + 1] = 1.0
        logits[0, 3, idx_a + 1] = 1.0
        text, _ = recognizer._ctc_greedy_decode(logits)
        assert text == "+"

    def test_multi_char_expression(self, recognizer):
        recognizer.warmup()
        chars = list("2+3=")
        indices = [recognizer.char_dict.index(c) for c in chars]
        logits = np.zeros((1, 8, 18385), dtype=np.float32)
        pos = 0
        for idx in indices:
            logits[0, pos, idx + 1] = 1.0
            pos += 1
            logits[0, pos, 0] = 1.0  # blank separator
            pos += 1
        text, _ = recognizer._ctc_greedy_decode(logits)
        assert text == "2+3="

    def test_empty_logits_returns_empty(self, recognizer):
        recognizer.warmup()
        logits = np.zeros((1, 10, 18385), dtype=np.float32)
        logits[0, :, 0] = 1.0  # all blanks
        text, conf = recognizer._ctc_greedy_decode(logits)
        assert text == ""
        assert conf == 0.0

    def test_out_of_range_index_ignored(self, recognizer):
        recognizer.warmup()
        logits = np.zeros((1, 3, 18385), dtype=np.float32)
        logits[0, 0, 18384] = 1.0  # last class, maps to dict[18383] which is out of range
        text, _ = recognizer._ctc_greedy_decode(logits)
        assert text == ""


class TestRecognizeEndToEnd:
    def test_recognize_simple_expression(self, recognizer):
        img = Image.new("RGB", (800, 120), "white")
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=48)
        except OSError:
            font = ImageFont.load_default()
        draw.text((20, 20), "5+2=", fill="black", font=font)

        result = recognizer.recognize(img)
        assert result.backend == "onnx"
        assert result.error is None
        assert len(result.raw_text) > 0

    def test_recognize_empty_image(self, recognizer):
        img = Image.new("RGB", (200, 60), "white")
        result = recognizer.recognize(img)
        assert result.backend == "onnx"
        assert result.error == "no text detected by OCR"
        assert result.confidence == 0.0

    def test_recognize_with_charset_validation(self, onnx_config):
        onnx_config.strict_charset = True
        rec = OnnxMathRecognizer(onnx_config)
        # White image - no text, should error
        img = Image.new("RGB", (200, 60), "white")
        result = rec.recognize(img)
        assert "no text detected" in (result.error or "")


class TestPreprocessShape:
    def test_preprocess_output_shape(self, recognizer):
        recognizer.warmup()
        import cv2
        img = Image.new("RGB", (400, 100), "white")
        arr = np.array(img.convert("RGB"))[:, :, ::-1].copy().astype(np.float32)
        h, w = arr.shape[:2]
        ratio = 48.0 / h
        new_w = int(w * ratio)
        arr = cv2.resize(arr, (new_w, 48), interpolation=cv2.INTER_LINEAR)
        arr = (arr - 127.5) / 127.5
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)
        assert arr.shape == (1, 3, 48, 192)
        assert -1.1 < arr.min() < 1.1
        assert -1.1 < arr.max() < 1.1


def test_config_defaults():
    config = OCRConfig()
    assert config.onnx_model_path == "models/PP-OCRv5_server_rec.onnx"
    assert config.onnx_dict_path == "models/dict.txt"


def test_factory_rejects_unknown_backend():
    config = OCRConfig(backend="unknown_backend")
    with pytest.raises(ValueError, match="unsupported OCR backend"):
        create_recognizer(config)
