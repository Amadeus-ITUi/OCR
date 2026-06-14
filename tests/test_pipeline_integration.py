from PIL import Image, ImageDraw

from robocon_ocr.config import PipelineConfig
from robocon_ocr.image_recognition.base import OCRResult
from robocon_ocr.pipeline import run_image_pipeline, run_pipeline
from robocon_ocr.staged_pipeline import run_dataset_pipeline_image


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

    original = pipeline_module.create_recognizer
    pipeline_module.create_recognizer = lambda _config: StubRecognizer(_config)
    try:
        records = run_pipeline(PipelineConfig(dataset_dir=dataset_dir))
    finally:
        pipeline_module.create_recognizer = original

    target = next(record for record in records if record.image_name == "problem_0001.png")

    assert target.ocr.raw_text
    assert target.ocr.error is None
    assert target.parsed.expression == "5+2"
    assert target.parsed.answer == 7


def test_run_dataset_pipeline_image_stop_after_board_detection_skips_ocr(tmp_path):
    image = Image.new("RGB", (1280, 720), (30, 40, 50))
    draw = ImageDraw.Draw(image)
    draw.polygon([(220, 140), (1060, 140), (1060, 612), (220, 612)], fill="white")
    draw.text((460, 320), "5+2=", fill="black")

    class FailIfCalledRecognizer:
        def recognize(self, image):
            raise AssertionError("OCR should not run when stop_after_stage=board_detection")

    context = run_dataset_pipeline_image(
        image=image,
        image_name="problem_0001.png",
        config=PipelineConfig(dataset_dir=tmp_path, stop_after_stage="board_detection"),
        recognizer=FailIfCalledRecognizer(),
    )

    assert context.board_detection is not None
    assert context.board_detection.roi_found is True
    assert context.rectification is None
    assert context.enhancement is None
    assert context.ocr_stage is None
    assert context.parsed is None


def test_run_dataset_pipeline_image_extracts_expression_region(tmp_path):
    image = Image.new("RGB", (1280, 720), (30, 40, 50))
    draw = ImageDraw.Draw(image)
    draw.polygon([(220, 140), (1060, 140), (1060, 612), (220, 612)], fill="white")
    draw.text((460, 320), "5+2=", fill="black")

    context = run_dataset_pipeline_image(
        image=image,
        image_name="problem_0001.png",
        config=PipelineConfig(dataset_dir=tmp_path, stop_after_stage="expression_region"),
    )

    assert context.rectification is not None
    assert context.expression_region is not None
    assert context.expression_region.region_found is True
    assert context.expression_region.cropped_region is not None
    assert context.expression_region.cropped_region.width < context.rectification.rectified.width
    assert context.expression_region.cropped_region.height < context.rectification.rectified.height
    assert context.enhancement is None
    assert context.ocr_stage is None
    assert context.parsed is None


def test_run_image_pipeline_stop_after_ocr_still_exposes_parsed_result(tmp_path):
    class StubRecognizer:
        supports_fallback_variants = False

        def __init__(self, config) -> None:
            self.config = config

        def warmup(self) -> None:
            return None

        def recognize(self, image):
            return OCRResult(raw_text="2×(9+19)×20÷5=", confidence=1.0, lines=["2×(9+19)×20÷5="], backend="lightweight")

    image = Image.new("RGB", (1280, 720), (30, 40, 50))
    draw = ImageDraw.Draw(image)
    draw.polygon([(220, 140), (1060, 140), (1060, 612), (220, 612)], fill="white")
    draw.text((380, 320), "2x(9+19)x20/5=", fill="black")

    import robocon_ocr.pipeline as pipeline_module

    original = pipeline_module.create_recognizer
    pipeline_module.create_recognizer = lambda _config: StubRecognizer(_config)
    try:
        record = run_image_pipeline(
            image=image,
            image_name="camera_0.png",
            config=PipelineConfig(dataset_dir=tmp_path, stop_after_stage="ocr"),
        )
    finally:
        pipeline_module.create_recognizer = original

    assert record.ocr.raw_text == "2×(9+19)×20÷5="
    assert record.parsed.expression == "2×(9+19)×20÷5"
    assert record.parsed.answer == 224
