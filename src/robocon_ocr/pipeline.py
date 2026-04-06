from __future__ import annotations

from robocon_ocr.config import PipelineConfig
from robocon_ocr.image_recognition.dataset_source import list_images, load_labels
from robocon_ocr.image_recognition.paddle_recognizer import PaddleMathRecognizer
from robocon_ocr.image_recognition.preprocess import prepare_for_ocr, save_debug_images
from robocon_ocr.result.expression import parse_expression
from robocon_ocr.result.reporter import PipelineRecord


def run_pipeline(config: PipelineConfig) -> list[PipelineRecord]:
    image_paths = list_images(config.dataset_dir)
    labels = load_labels(config.label_file) if config.label_file else {}
    recognizer = PaddleMathRecognizer(config.ocr)
    records: list[PipelineRecord] = []

    for image_path in image_paths:
        cropped, prepared = prepare_for_ocr(image_path, config.preprocess)
        if config.debug_dir is not None:
            save_debug_images(image_path.name, cropped, prepared, config.debug_dir)

        ocr_result = recognizer.recognize(prepared)
        parsed = parse_expression(ocr_result.raw_text)
        records.append(
            PipelineRecord(
                image_name=image_path.name,
                ocr=ocr_result,
                parsed=parsed,
                label=labels.get(image_path.name),
            )
        )

    return records

