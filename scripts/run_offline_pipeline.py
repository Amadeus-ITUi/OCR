#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from robocon_ocr.config import PipelineConfig
from robocon_ocr.pipeline import run_pipeline
from robocon_ocr.result.reporter import summarize


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline OCR pipeline on dataset images.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Dataset image directory.")
    parser.add_argument("--label-file", type=Path, help="Tab-separated label file.")
    parser.add_argument("--debug-dir", type=Path, help="Directory for cropped/preprocessed debug images.")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    config = PipelineConfig(
        dataset_dir=args.dataset_dir,
        label_file=args.label_file,
        debug_dir=args.debug_dir,
    )
    records = run_pipeline(config)

    for record in records:
        print(f"[{record.image_name}]")
        print(f"  raw_text: {record.ocr.raw_text}")
        print(f"  normalized: {record.parsed.normalized_text}")
        print(f"  expression: {record.parsed.expression}")
        print(f"  answer: {record.parsed.answer}")
        print(f"  confidence: {record.ocr.confidence:.4f}")
        print(f"  valid: {record.parsed.is_valid}")
        if record.label is not None:
            print(f"  gt_expression: {record.label.expression}")
            print(f"  gt_answer: {record.label.answer}")
            print(f"  expression_match: {record.expression_match}")
            print(f"  answer_match: {record.answer_match}")
        if record.parsed.error:
            print(f"  error: {record.parsed.error}")

    summary = summarize(records)
    print("[summary]")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

