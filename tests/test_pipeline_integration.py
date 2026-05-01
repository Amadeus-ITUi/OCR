import shutil
from pathlib import Path

from robocon_ocr.config import PipelineConfig
from robocon_ocr.pipeline import run_pipeline


def test_problem_0001_is_not_empty_expression():
    if shutil.which("tesseract") is None:
        return

    dataset_dir = Path(__file__).resolve().parents[1] / "dataset" / "num_10_com_1"
    records = run_pipeline(PipelineConfig(dataset_dir=dataset_dir))
    target = next(record for record in records if record.image_name == "problem_0001.png")

    assert target.ocr.raw_text
    assert target.ocr.error is None
    assert target.parsed.expression == "5+2"
    assert target.parsed.answer == 7
