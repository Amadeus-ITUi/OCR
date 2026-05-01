from __future__ import annotations

import argparse
from pathlib import Path

from robocon_ocr.config import PipelineConfig
from robocon_ocr.pipeline import run_pipeline
from robocon_ocr.result.reporter import summarize


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline OCR pipeline on dataset images.")
    parser.add_argument("dataset_dir", type=Path, help="Dataset image directory.")
    parser.add_argument(
        "--label-file",
        type=Path,
        help="Tab-separated label file. Defaults to <dataset_dir>/problems_and_answers.txt if present.",
    )
    parser.add_argument("--debug-dir", type=Path, help="Directory for cropped/preprocessed debug images.")
    return parser


def resolve_label_file(dataset_dir: Path, label_file: Path | None) -> Path | None:
    if label_file is not None:
        return label_file.expanduser()

    candidate = dataset_dir / "problems_and_answers.txt"
    if candidate.is_file():
        return candidate
    return None


def build_config(args: argparse.Namespace) -> PipelineConfig:
    dataset_dir = args.dataset_dir.expanduser()
    return PipelineConfig(
        dataset_dir=dataset_dir,
        label_file=resolve_label_file(dataset_dir, args.label_file),
        debug_dir=args.debug_dir.expanduser() if args.debug_dir else None,
    )


def print_records(records) -> None:
    for record in records:
        print(f"[{record.image_name}]")
        print(f"  raw_text: {record.ocr.raw_text}")
        print(f"  normalized: {record.parsed.normalized_text}")
        print(f"  expression: {record.parsed.expression}")
        print(f"  answer: {record.parsed.answer}")
        print(f"  confidence: {record.ocr.confidence:.4f}")
        print(f"  psm: {record.ocr.psm}")
        print(f"  valid: {record.parsed.is_valid}")
        if record.label is not None:
            print(f"  gt_expression: {record.label.expression}")
            print(f"  gt_answer: {record.label.answer}")
            print(f"  expression_match: {record.expression_match}")
            print(f"  answer_match: {record.answer_match}")
        if record.ocr.error:
            print(f"  ocr_error: {record.ocr.error}")
        if record.parsed.error:
            print(f"  error: {record.parsed.error}")


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    config = build_config(args)
    records = run_pipeline(config)

    print_records(records)

    summary = summarize(records)
    print("[summary]")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return 0
