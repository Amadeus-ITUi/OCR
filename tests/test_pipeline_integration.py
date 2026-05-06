from PIL import Image, ImageDraw

from robocon_ocr.config import PipelineConfig
from robocon_ocr.image_recognition.pix2tex_recognizer import OCRResult
from robocon_ocr.pipeline import _recognize_with_fallback_variants, run_pipeline
from robocon_ocr.image_recognition.preprocess import prepare_image_for_ocr


def test_problem_0001_is_not_empty_expression(tmp_path):
    class StubRecognizer:
        def __init__(self, config) -> None:
            self.config = config

        def warmup(self) -> None:
            return None

        def recognize(self, image):
            return OCRResult(raw_text="5+2=", confidence=1.0, lines=["5+2="])

    dataset_dir = tmp_path
    image = Image.new("RGB", (1280, 720), (30, 40, 50))
    draw = ImageDraw.Draw(image)
    draw.polygon([(220, 140), (1060, 140), (1060, 612), (220, 612)], fill="white")
    draw.text((460, 320), "5+2=", fill="black")
    image.save(dataset_dir / "problem_0001.png")

    import robocon_ocr.pipeline as pipeline_module

    original = pipeline_module.Pix2TexMathRecognizer
    pipeline_module.Pix2TexMathRecognizer = StubRecognizer
    try:
        records = run_pipeline(PipelineConfig(dataset_dir=dataset_dir))
    finally:
        pipeline_module.Pix2TexMathRecognizer = original

    target = next(record for record in records if record.image_name == "problem_0001.png")

    assert target.ocr.raw_text
    assert target.ocr.error is None
    assert target.parsed.expression == "5+2"
    assert target.parsed.answer == 7


def test_recognize_with_fallback_variants_uses_prepared_on_failure(tmp_path):
    image = Image.new("RGB", (1280, 720), (30, 40, 50))
    draw = ImageDraw.Draw(image)
    draw.polygon([(220, 140), (1060, 140), (1060, 612), (220, 612)], fill="white")
    draw.text((460, 320), "5+2=", fill="black")
    preprocess = prepare_image_for_ocr(image, PipelineConfig(dataset_dir=tmp_path).preprocess)

    class StubRecognizer:
        def recognize(self, candidate_image):
            if candidate_image.mode == "RGB":
                return OCRResult(
                    raw_text="\\left(\\begin{array}{c}{{bad}}\\end{array}\\right)",
                    confidence=0.0,
                    lines=[],
                    error="unsupported symbol outside arithmetic charset",
                )
            return OCRResult(raw_text="5+2=", confidence=1.0, lines=["5+2="])

    candidates = _recognize_with_fallback_variants(StubRecognizer(), preprocess)

    assert candidates[0].error == "unsupported symbol outside arithmetic charset"
    assert any(candidate.raw_text == "5+2=" for candidate in candidates[1:])
