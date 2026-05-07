from __future__ import annotations

from PIL import Image
from PIL import ImageOps

from robocon_ocr.config import PipelineConfig
from robocon_ocr.image_recognition.base import MathTextRecognizer, OCRResult
from robocon_ocr.image_recognition.dataset_source import list_images, load_labels
from robocon_ocr.image_recognition.factory import create_recognizer
from robocon_ocr.image_recognition.preprocess import (
    PreprocessResult,
    prepare_for_ocr,
    prepare_image_for_ocr,
    save_debug_images,
)
from robocon_ocr.result.expression import ParsedExpression, parse_expression
from robocon_ocr.result.reporter import PipelineRecord
from robocon_ocr.staged_pipeline import context_to_record
from robocon_ocr.staged_pipeline import run_dataset_pipeline_image
from robocon_ocr.staged_pipeline import save_stage_debug_images


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


def _empty_record_result() -> tuple[OCRResult, ParsedExpression]:
    return (
        OCRResult(
            raw_text="",
            confidence=0.0,
            lines=[],
            psm=None,
            error="roi not found",
        ),
        ParsedExpression("", "", None, False, "roi not found"),
    )


def _record_from_preprocess(
    image_name: str,
    preprocess_result: PreprocessResult,
    recognizer: MathTextRecognizer,
    label,
) -> PipelineRecord:
    if not preprocess_result.roi_found:
        ocr_result, parsed = _empty_record_result()
    else:
        ocr_candidates = _recognize_with_fallback_variants(recognizer, preprocess_result)
        ocr_result, parsed = _select_best_result(ocr_candidates)

    return PipelineRecord(
        image_name=image_name,
        ocr=ocr_result,
        parsed=parsed,
        label=label,
        roi_found=preprocess_result.roi_found,
        roi_quad=preprocess_result.roi_quad,
    )


def _recognize_with_fallback_variants(
    recognizer: MathTextRecognizer,
    preprocess_result: PreprocessResult,
) -> list[OCRResult]:
    primary = recognizer.recognize(preprocess_result.rectified)
    candidates = [primary]
    if not getattr(recognizer, "supports_fallback_variants", True):
        return candidates
    primary_parsed = parse_expression(primary.raw_text)
    short_expression = primary_parsed.is_valid and len(primary_parsed.expression) <= 6
    if primary.error is None and primary_parsed.is_valid and not short_expression:
        return candidates

    fallback_images = [
        ("gray", preprocess_result.rectified.convert("L"), 1.03 if short_expression else 1.0),
        ("prepared", preprocess_result.prepared, 1.02 if short_expression else 1.0),
        (
            "invert_prepared",
            ImageOps.invert(preprocess_result.prepared.convert("L")),
            1.01 if short_expression else 1.0,
        ),
    ]
    seen_raw = {primary.raw_text}
    if short_expression:
        primary.confidence = min(primary.confidence, 1.0)
    for _name, image, confidence_boost in fallback_images:
        candidate = recognizer.recognize(image)
        if candidate.raw_text in seen_raw and candidate.error == primary.error:
            continue
        candidate.confidence *= confidence_boost
        candidates.append(candidate)
        seen_raw.add(candidate.raw_text)
    return candidates


def _save_legacy_debug_outputs(
    image_name: str,
    image: Image.Image,
    config: PipelineConfig,
) -> None:
    if config.debug_dir is None:
        return
    preprocess_result = prepare_image_for_ocr(image.convert("RGB"), config.preprocess)
    save_debug_images(
        image_name,
        preprocess_result.cropped,
        preprocess_result.prepared,
        config.debug_dir,
        rectified=preprocess_result.rectified,
    )


def run_pipeline(config: PipelineConfig) -> list[PipelineRecord]:
    image_paths = list_images(config.dataset_dir)
    labels = load_labels(config.label_file) if config.label_file else {}
    recognizer = create_recognizer(config.ocr)
    records: list[PipelineRecord] = []
    warmed_up = False

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        context = run_dataset_pipeline_image(
            image=image,
            image_name=image_path.name,
            config=config,
            recognizer=recognizer,
        )
        context.label = labels.get(image_path.name)
        if context.board_detection is not None and context.board_detection.roi_found and config.ocr.warmup and not warmed_up:
            recognizer.warmup()
            warmed_up = True
        if config.debug_dir is not None:
            _save_legacy_debug_outputs(image_path.name, image, config)
            if config.debug_save_stages:
                save_stage_debug_images(context, config.debug_dir)
        records.append(context_to_record(context))

    return records


def run_image_pipeline(
    image: Image.Image,
    image_name: str,
    config: PipelineConfig,
) -> PipelineRecord:
    recognizer = create_recognizer(config.ocr)
    if config.ocr.warmup:
        recognizer.warmup()
    context = run_dataset_pipeline_image(
        image=image,
        image_name=image_name,
        config=config,
        recognizer=recognizer,
    )
    if config.debug_dir is not None:
        _save_legacy_debug_outputs(image_name, image, config)
        if config.debug_save_stages:
            save_stage_debug_images(context, config.debug_dir)
    return context_to_record(context)
