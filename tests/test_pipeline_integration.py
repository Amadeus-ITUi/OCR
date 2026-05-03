import shutil
from PIL import Image, ImageDraw

from robocon_ocr.config import PipelineConfig
from robocon_ocr.pipeline import run_pipeline


def test_problem_0001_is_not_empty_expression(tmp_path):
    if shutil.which("tesseract") is None:
        return

    dataset_dir = tmp_path
    image = Image.new("RGB", (1280, 720), (30, 40, 50))
    draw = ImageDraw.Draw(image)
    draw.polygon([(220, 140), (1060, 140), (1060, 612), (220, 612)], fill="white")
    draw.text((460, 320), "5+2=", fill="black")
    image.save(dataset_dir / "problem_0001.png")
    records = run_pipeline(PipelineConfig(dataset_dir=dataset_dir))
    target = next(record for record in records if record.image_name == "problem_0001.png")

    assert target.ocr.raw_text
    assert target.ocr.error is None
    assert target.parsed.expression == "5+2"
    assert target.parsed.answer == 7
