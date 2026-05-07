from robocon_ocr.config import OCRConfig
from robocon_ocr.image_recognition.pix2tex_recognizer import Pix2TexMathRecognizer


def test_resolve_no_cuda_uses_cpu_when_explicit_cpu():
    recognizer = Pix2TexMathRecognizer(OCRConfig(device="cpu"))

    assert recognizer._resolve_no_cuda(cuda_available=True) is True


def test_resolve_no_cuda_uses_gpu_when_explicit_cuda():
    recognizer = Pix2TexMathRecognizer(OCRConfig(device="cuda"))

    assert recognizer._resolve_no_cuda(cuda_available=False) is False


def test_resolve_no_cuda_prefers_gpu_in_auto_mode_when_available():
    recognizer = Pix2TexMathRecognizer(OCRConfig(device="auto"))

    assert recognizer._resolve_no_cuda(cuda_available=True) is False


def test_resolve_no_cuda_falls_back_to_cpu_in_auto_mode_when_gpu_unavailable():
    recognizer = Pix2TexMathRecognizer(OCRConfig(device="auto"))

    assert recognizer._resolve_no_cuda(cuda_available=False) is True
