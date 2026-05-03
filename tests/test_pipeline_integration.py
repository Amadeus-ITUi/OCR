from PIL import Image, ImageDraw

from robocon_ocr.config import PipelineConfig
from robocon_ocr.image_recognition.pix2tex_recognizer import OCRResult
from robocon_ocr.pipeline import run_pipeline


def test_problem_0001_is_not_empty_expression(tmp_path):
    class StubRecognizer:
        def __init__(self, config) -> None:
            self.config = config

        def warmup(self) -> None:
            return None

        def recognize_candidates(self, image):
            return [OCRResult(raw_text="5+2=", confidence=1.0, lines=["5+2="])]

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
