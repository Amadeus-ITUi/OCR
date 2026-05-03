from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import sys
import time
from time import strftime

import numpy as np

from robocon_ocr.config import CameraConfig, PipelineConfig
from robocon_ocr.image_recognition.preprocess import prepare_image_for_ocr, save_debug_images
from robocon_ocr.image_recognition.tesseract_recognizer import TesseractMathRecognizer
from robocon_ocr.pipeline import _select_best_result, run_pipeline
from robocon_ocr.result.reporter import summarize
from robocon_ocr.vision_capture.usb_camera import USBCameraCapture


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Robocon OCR on datasets or a USB camera frame.")
    subparsers = parser.add_subparsers(dest="command")

    dataset_parser = subparsers.add_parser("dataset", help="Run OCR on a dataset directory.")
    dataset_parser.add_argument("dataset_dir", type=Path, help="Dataset image directory.")
    dataset_parser.add_argument(
        "--label-file",
        type=Path,
        help="Tab-separated label file. Defaults to <dataset_dir>/problems_and_answers.txt if present.",
    )
    dataset_parser.add_argument(
        "--debug-dir",
        type=Path,
        help="Directory for cropped/preprocessed debug images.",
    )

    camera_parser = subparsers.add_parser("camera", help="Run realtime OCR on a USB camera stream.")
    camera_parser.add_argument("--device-index", type=int, default=2, help="USB camera device index, default: 2")
    camera_parser.add_argument("--width", type=int, default=1280, help="Capture width, default: 1280")
    camera_parser.add_argument("--height", type=int, default=720, help="Capture height, default: 720")
    camera_parser.add_argument("--fps", type=float, default=30.0, help="Requested capture FPS, default: 30")
    camera_parser.add_argument(
        "--pixel-format",
        default="MJPG",
        help="Requested fourcc pixel format such as MJPG or YUYV, default: MJPG",
    )
    camera_parser.add_argument(
        "--warmup-frames",
        type=int,
        default=5,
        help="Number of frames to discard before OCR, default: 5",
    )
    camera_parser.add_argument(
        "--capture-timeout-ms",
        type=int,
        default=3000,
        help="Camera capture timeout in milliseconds, default: 3000",
    )
    camera_parser.add_argument(
        "--interval-ms",
        type=int,
        default=50,
        help="Delay between OCR attempts in milliseconds, default: 50",
    )
    camera_parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional maximum number of OCR iterations before exit.",
    )
    camera_parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print every OCR result instead of only changes.",
    )
    camera_parser.add_argument(
        "--show-window",
        action="store_true",
        help="Show cv2 debug windows with the live frame and OCR details.",
    )
    camera_parser.add_argument(
        "--window-scale",
        type=float,
        default=0.75,
        help="Scale factor for cv2 debug windows, default: 0.75",
    )
    camera_parser.add_argument(
        "--save-frame",
        type=Path,
        help="Optional path to save the captured RGB frame before OCR.",
    )
    camera_parser.add_argument(
        "--debug-dir",
        type=Path,
        help="Directory for cropped/preprocessed debug images.",
    )
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


def build_camera_config(args: argparse.Namespace) -> CameraConfig:
    return CameraConfig(
        device_index=args.device_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        pixel_format=args.pixel_format,
        warmup_frames=args.warmup_frames,
        capture_timeout_ms=args.capture_timeout_ms,
        interval_ms=args.interval_ms,
        max_frames=args.max_frames,
        emit_only_changes=not args.print_all,
        save_frame=args.save_frame.expanduser() if args.save_frame else None,
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


def _camera_signature(record) -> tuple[str, int | None, bool, str | None]:
    return (
        record.parsed.expression,
        record.parsed.answer,
        record.parsed.is_valid,
        record.parsed.error,
    )


def print_camera_record(record, frame_index: int) -> None:
    timestamp = strftime("%H:%M:%S")
    print(f"[camera frame={frame_index} time={timestamp}]")
    print_records([record])
    print()


def _wrap_text(text: str, width: int = 48) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    remaining = text
    while len(remaining) > width:
        lines.append(remaining[:width])
        remaining = remaining[width:]
    lines.append(remaining)
    return lines


def _build_camera_overlay(
    frame_rgb,
    record,
    frame_index: int,
    ocr_ms: float,
    window_scale: float,
):
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("未安装 OpenCV GUI 版本，无法显示调试窗口。") from exc

    frame_bgr = cv2.cvtColor(np.asarray(frame_rgb), cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]
    panel_width = max(440, int(w * 0.32))
    canvas = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
    canvas[:, :w] = frame_bgr
    canvas[:, w:] = (28, 28, 28)

    text_x = w + 18
    y = 36
    line_gap = 28
    small_gap = 22
    font = cv2.FONT_HERSHEY_SIMPLEX

    def put(line: str, color=(235, 235, 235), scale=0.6, thickness=1, gap=line_gap):
        nonlocal y
        cv2.putText(canvas, line, (text_x, y), font, scale, color, thickness, cv2.LINE_AA)
        y += gap

    put(f"Frame: {frame_index}", color=(120, 220, 255), scale=0.72, thickness=2)
    put(f"OCR: {ocr_ms:.1f} ms", color=(120, 220, 255), gap=small_gap)
    put(f"Confidence: {record.ocr.confidence:.4f}", gap=small_gap)
    put(f"PSM: {record.ocr.psm}", gap=small_gap)
    put(f"Valid: {record.parsed.is_valid}", color=(120, 255, 160) if record.parsed.is_valid else (80, 160, 255))
    put("Raw:", color=(180, 180, 255), gap=small_gap)
    for line in _wrap_text(record.ocr.raw_text or "<empty>"):
        put(line, scale=0.54, gap=small_gap)
    put("Expression:", color=(180, 180, 255), gap=small_gap)
    for line in _wrap_text(record.parsed.expression or "<empty>"):
        put(line, scale=0.54, gap=small_gap)
    put(f"Answer: {record.parsed.answer}", gap=small_gap)
    if record.parsed.error:
        put("Error:", color=(80, 160, 255), gap=small_gap)
        for line in _wrap_text(record.parsed.error):
            put(line, color=(80, 160, 255), scale=0.54, gap=small_gap)
    if record.ocr.error:
        put("OCR Error:", color=(80, 160, 255), gap=small_gap)
        for line in _wrap_text(record.ocr.error):
            put(line, color=(80, 160, 255), scale=0.54, gap=small_gap)

    if window_scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=window_scale, fy=window_scale, interpolation=cv2.INTER_AREA)
    return canvas


def _show_debug_windows(original, cropped, prepared, record, frame_index: int, ocr_ms: float, window_scale: float) -> bool:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("未安装 OpenCV GUI 版本，无法显示调试窗口。") from exc

    try:
        overlay = _build_camera_overlay(original, record, frame_index, ocr_ms, window_scale)
        cropped_bgr = cv2.cvtColor(np.asarray(cropped), cv2.COLOR_RGB2BGR)
        prepared_bgr = cv2.cvtColor(np.asarray(prepared.convert("RGB")), cv2.COLOR_RGB2BGR)
        if window_scale != 1.0:
            cropped_bgr = cv2.resize(cropped_bgr, None, fx=window_scale, fy=window_scale, interpolation=cv2.INTER_AREA)
            prepared_bgr = cv2.resize(prepared_bgr, None, fx=window_scale, fy=window_scale, interpolation=cv2.INTER_AREA)

        cv2.imshow("robocon_ocr_live", overlay)
        cv2.imshow("robocon_ocr_cropped", cropped_bgr)
        cv2.imshow("robocon_ocr_prepared", prepared_bgr)
        key = cv2.waitKey(1) & 0xFF
        return key not in {27, ord("q"), ord("Q")}
    except cv2.error as exc:
        raise RuntimeError(
            "当前 OpenCV 构建不支持窗口显示。请安装带 GUI 的 `opencv-python`，不要使用 headless 版本。"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if args_list and args_list[0] not in {"dataset", "camera", "-h", "--help"}:
        args_list = ["dataset", *args_list]

    args = build_argparser().parse_args(args_list)

    if args.command in {None, "dataset"}:
        config = build_config(args)
        records = run_pipeline(config)
        print_records(records)
        summary = summarize(records)
        print("[summary]")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    else:
        camera = USBCameraCapture(build_camera_config(args))
        pipeline_config = PipelineConfig(
            dataset_dir=Path("."),
            debug_dir=args.debug_dir.expanduser() if args.debug_dir else None,
        )
        recognizer = TesseractMathRecognizer(pipeline_config.ocr)
        last_signature = None
        emitted = 0
        try:
            for frame_index, image in camera.stream_frames():
                started_at = time.perf_counter()
                cropped, prepared = prepare_image_for_ocr(image.convert("RGB"), pipeline_config.preprocess)
                if pipeline_config.debug_dir is not None:
                    save_debug_images(f"camera_{args.device_index}_{frame_index:06d}.png", cropped, prepared, pipeline_config.debug_dir)
                ocr_candidates = recognizer.recognize_candidates(prepared)
                ocr_result, parsed = _select_best_result(ocr_candidates)
                record = SimpleNamespace(
                    image_name=f"camera_{args.device_index}_{frame_index:06d}.png",
                    ocr=ocr_result,
                    parsed=parsed,
                    label=None,
                )
                ocr_ms = (time.perf_counter() - started_at) * 1000.0
                signature = _camera_signature(record)
                if args.print_all or signature != last_signature:
                    print_camera_record(record, frame_index)
                    last_signature = signature
                    emitted += 1
                if args.show_window:
                    keep_running = _show_debug_windows(
                        image,
                        cropped,
                        prepared,
                        record,
                        frame_index,
                        ocr_ms,
                        args.window_scale,
                    )
                    if not keep_running:
                        break
        except KeyboardInterrupt:
            print("\n[camera] stopped by user")
        finally:
            if args.show_window:
                try:
                    import cv2
                    cv2.destroyAllWindows()
                except Exception:
                    pass
        print("[summary]")
        print(f"  frames_emitted: {emitted}")

    return 0
