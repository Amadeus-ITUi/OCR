from __future__ import annotations

from PIL import Image

from robocon_ocr.config import PipelineConfig
from robocon_ocr.image_recognition.dataset_source import list_images, load_labels
from robocon_ocr.image_recognition.tesseract_recognizer import TesseractMathRecognizer
from robocon_ocr.image_recognition.preprocess import (
    prepare_for_ocr,
    prepare_image_for_ocr,
    save_debug_images,
)
from robocon_ocr.result.expression import ParsedExpression, parse_expression
from robocon_ocr.result.reporter import PipelineRecord


def _select_best_result(candidates) -> tuple:
    best_choice = None

    for candidate in candidates:
        parsed = parse_expression(candidate.raw_text)
        choice = (
            candidate,
            parsed,
            (
                parsed.is_valid,
                candidate.confidence,
                len(parsed.expression),
                -(parsed.error is not None),
            ),
        )
        if best_choice is None or choice[2] > best_choice[2]:
            best_choice = choice

    if best_choice is None:
        return None, ParsedExpression("", "", None, False, "empty expression")
    return best_choice[0], best_choice[1]


def run_pipeline(config: PipelineConfig) -> list[PipelineRecord]:
    image_paths = list_images(config.dataset_dir)
    labels = load_labels(config.label_file) if config.label_file else {}
    recognizer = TesseractMathRecognizer(config.ocr)
    records: list[PipelineRecord] = []

    for image_path in image_paths:
        cropped, prepared = prepare_for_ocr(image_path, config.preprocess)
        if config.debug_dir is not None:
            save_debug_images(image_path.name, cropped, prepared, config.debug_dir)

        ocr_candidates = recognizer.recognize_candidates(prepared)
        ocr_result, parsed = _select_best_result(ocr_candidates)
        records.append(
            PipelineRecord(
                image_name=image_path.name,
                ocr=ocr_result,
                parsed=parsed,
                label=labels.get(image_path.name),
            )
        )

    return records


def run_image_pipeline(
    image: Image.Image,
    image_name: str,
    config: PipelineConfig,
) -> PipelineRecord:
    recognizer = TesseractMathRecognizer(config.ocr)
    cropped, prepared = prepare_image_for_ocr(image.convert("RGB"), config.preprocess)
    if config.debug_dir is not None:
        save_debug_images(image_name, cropped, prepared, config.debug_dir)

    ocr_candidates = recognizer.recognize_candidates(prepared)
    ocr_result, parsed = _select_best_result(ocr_candidates)
    return PipelineRecord(
        image_name=image_name,
        ocr=ocr_result,
        parsed=parsed,
        label=None,
    )
