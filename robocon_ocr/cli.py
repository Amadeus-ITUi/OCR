from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys
import threading
import time
from time import strftime

import numpy as np
from PIL import Image

from robocon_ocr.camera_tuning import DEFAULT_CAMERA_TUNING
from robocon_ocr.config import CameraConfig, PipelineConfig
from robocon_ocr.image_recognition.preprocess import PreprocessResult, ROIDebugInfo, prepare_image_for_ocr, save_debug_images
from robocon_ocr.image_recognition.tesseract_recognizer import OCRResult, TesseractMathRecognizer
from robocon_ocr.pipeline import _select_best_result, run_pipeline
from robocon_ocr.result.expression import ParsedExpression
from robocon_ocr.result.reporter import PipelineRecord, summarize
from robocon_ocr.vision_capture.usb_camera import USBCameraCapture


@dataclass(slots=True)
class LatestFrameBuffer:
    frame_index: int = -1
    frame_bgr: np.ndarray | None = None
    captured_at: float = 0.0
    stopped: bool = False
    error: Exception | None = None
    condition: threading.Condition = field(default_factory=threading.Condition)

    def publish(self, frame_index: int, frame_bgr: np.ndarray) -> None:
        with self.condition:
            self.frame_index = frame_index
            self.frame_bgr = frame_bgr.copy()
            self.captured_at = time.monotonic()
            self.condition.notify_all()

    def stop(self, error: Exception | None = None) -> None:
        with self.condition:
            self.stopped = True
            self.error = error
            self.condition.notify_all()

    def wait_for_next(self, last_processed_index: int) -> tuple[int, np.ndarray, float] | None:
        with self.condition:
            while True:
                if self.error is not None:
                    raise RuntimeError("摄像头采集线程异常退出") from self.error
                if self.frame_bgr is not None and self.frame_index > last_processed_index:
                    return self.frame_index, self.frame_bgr.copy(), self.captured_at
                if self.stopped:
                    return None
                self.condition.wait(timeout=0.1)


@dataclass(slots=True)
class CameraStats:
    frames_captured: int = 0
    frames_processed: int = 0
    frames_skipped_no_roi: int = 0
    frames_dropped_stale: int = 0
    frames_emitted: int = 0


@dataclass(slots=True)
class DisplayState:
    original: Image.Image
    preprocess: PreprocessResult
    record: PipelineRecord
    frame_index: int
    ocr_ms: float


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
    camera_parser.add_argument("--device-index", type=int, help="USB camera device index. Default comes from camera_tuning.py.")
    camera_parser.add_argument("--width", type=int, help="Capture width. Default comes from camera_tuning.py.")
    camera_parser.add_argument("--height", type=int, help="Capture height. Default comes from camera_tuning.py.")
    camera_parser.add_argument("--fps", type=float, help="Requested capture FPS. Default comes from camera_tuning.py.")
    camera_parser.add_argument(
        "--pixel-format",
        help="Requested fourcc pixel format such as MJPG. Default comes from camera_tuning.py.",
    )
    camera_parser.add_argument(
        "--warmup-frames",
        type=int,
        help="Number of frames to discard before OCR. Default comes from camera_tuning.py.",
    )
    camera_parser.add_argument(
        "--capture-timeout-ms",
        type=int,
        help="Camera capture timeout in milliseconds. Default comes from camera_tuning.py.",
    )
    camera_parser.add_argument(
        "--interval-ms",
        type=int,
        help="Legacy delay between OCR attempts in milliseconds. Default comes from camera_tuning.py.",
    )
    camera_parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional maximum number of captured frames before exit.",
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
    defaults = DEFAULT_CAMERA_TUNING
    return CameraConfig(
        device_index=args.device_index if args.device_index is not None else defaults.device_index,
        width=args.width if args.width is not None else defaults.width,
        height=args.height if args.height is not None else defaults.height,
        fps=args.fps if args.fps is not None else defaults.fps,
        pixel_format=args.pixel_format if args.pixel_format is not None else defaults.pixel_format,
        warmup_frames=args.warmup_frames if args.warmup_frames is not None else defaults.warmup_frames,
        capture_timeout_ms=(
            args.capture_timeout_ms if args.capture_timeout_ms is not None else defaults.capture_timeout_ms
        ),
        interval_ms=args.interval_ms if args.interval_ms is not None else defaults.interval_ms,
        max_frames=args.max_frames,
        emit_only_changes=not args.print_all,
        save_frame=args.save_frame.expanduser() if args.save_frame else None,
        async_latest_frame=defaults.async_latest_frame,
        auto_exposure=defaults.auto_exposure,
        exposure_dynamic_framerate=defaults.exposure_dynamic_framerate,
        exposure_time_absolute=defaults.exposure_time_absolute,
        gain=defaults.gain,
        brightness=defaults.brightness,
        contrast=defaults.contrast,
        saturation=defaults.saturation,
        sharpness=defaults.sharpness,
        gamma=defaults.gamma,
        white_balance_automatic=defaults.white_balance_automatic,
        white_balance_temperature=defaults.white_balance_temperature,
        focus_automatic_continuous=defaults.focus_automatic_continuous,
        focus_absolute=defaults.focus_absolute,
        backlight_compensation=defaults.backlight_compensation,
        power_line_frequency=defaults.power_line_frequency,
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
        print(f"  roi_found: {record.roi_found}")
        if record.roi_quad is not None:
            print(f"  roi_quad: {record.roi_quad}")
        if record.label is not None:
            print(f"  gt_expression: {record.label.expression}")
            print(f"  gt_answer: {record.label.answer}")
            print(f"  expression_match: {record.expression_match}")
            print(f"  answer_match: {record.answer_match}")
        if record.ocr.error:
            print(f"  ocr_error: {record.ocr.error}")
        if record.parsed.error:
            print(f"  error: {record.parsed.error}")


def _camera_signature(record: PipelineRecord) -> tuple[str, int | None, bool, str | None]:
    return (
        record.parsed.expression,
        record.parsed.answer,
        record.parsed.is_valid,
        record.parsed.error,
    )


def print_camera_record(record: PipelineRecord, frame_index: int) -> None:
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


def _draw_roi_quad(frame_bgr: np.ndarray, roi_quad) -> np.ndarray:
    if roi_quad is None:
        return frame_bgr
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("未安装 OpenCV GUI 版本，无法显示调试窗口。") from exc

    canvas = frame_bgr.copy()
    points = np.array(roi_quad, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [points], isClosed=True, color=(0, 255, 255), thickness=3)
    return canvas


def _format_roi_debug_lines(roi_debug: ROIDebugInfo) -> list[str]:
    def fmt(value, digits: int = 3) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, int):
            return str(value)
        return f"{value:.{digits}f}"

    candidate_label = roi_debug.best_candidate_type or "none"
    reason = roi_debug.failure_reason or "passed"
    tolerance_label = (
        f"RectTol: {fmt(roi_debug.rectangle_ratio_tolerance)}"
        if candidate_label == "rectangle"
        else f"QuadTol: {fmt(roi_debug.quadrilateral_ratio_tolerance)}"
    )
    return [
        f"ROI: {roi_debug.roi_found}",
        f"Reason: {reason}",
        f"Type: {candidate_label}",
        f"Area: {fmt(roi_debug.best_candidate_area_ratio)} / {fmt(roi_debug.min_area_ratio_threshold)}",
        f"Edge: {fmt(roi_debug.best_candidate_edge_strength, 1)} / {fmt(roi_debug.edge_threshold, 1)}",
        f"Ratio: {fmt(roi_debug.best_candidate_ratio)}",
        f"Err: {fmt(roi_debug.best_candidate_ratio_error)}",
        tolerance_label,
        f"WhiteThr: {fmt(roi_debug.white_threshold, 0)}",
        f"Cand: {roi_debug.candidate_count}",
    ]


def _draw_roi_debug_overlay(frame_bgr: np.ndarray, roi_debug: ROIDebugInfo) -> np.ndarray:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("未安装 OpenCV GUI 版本，无法显示调试窗口。") from exc

    canvas = frame_bgr.copy()
    lines = _format_roi_debug_lines(roi_debug)
    if not lines:
        return canvas

    overlay = canvas.copy()
    panel_width = min(max(320, int(canvas.shape[1] * 0.34)), canvas.shape[1] - 20)
    panel_height = min(32 + (len(lines) * 24), canvas.shape[0] - 20)
    x0 = max(10, canvas.shape[1] - panel_width - 10)
    y0 = 44
    x1 = x0 + panel_width
    y1 = y0 + panel_height
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (10, 10, 10), thickness=-1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    y = y0 + 24
    cv2.putText(canvas, "ROI Debug", (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 220, 255), 2, cv2.LINE_AA)
    y += 24
    for line in lines:
        color = (235, 235, 235)
        if line.startswith("Reason:") and roi_debug.failure_reason is not None:
            color = (80, 160, 255)
        cv2.putText(canvas, line, (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, color, 1, cv2.LINE_AA)
        y += 22
    return canvas


def _put_panel_title(image_bgr: np.ndarray, title: str) -> np.ndarray:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("未安装 OpenCV GUI 版本，无法显示调试窗口。") from exc

    canvas = image_bgr.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (20, 20, 20), thickness=-1)
    cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
    return canvas


def _fit_panel(image_bgr: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("未安装 OpenCV GUI 版本，无法显示调试窗口。") from exc

    target_w, target_h = target_size
    resized = cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized


def _build_text_panel(
    record: PipelineRecord,
    frame_index: int,
    ocr_ms: float,
    panel_size: tuple[int, int],
) -> np.ndarray:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("未安装 OpenCV GUI 版本，无法显示调试窗口。") from exc

    panel_w, panel_h = panel_size
    canvas = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    canvas[:, :] = (28, 28, 28)

    text_x = 18
    y = 42
    line_gap = 28
    small_gap = 22
    font = cv2.FONT_HERSHEY_SIMPLEX

    def put(line: str, color=(235, 235, 235), scale=0.6, thickness=1, gap=line_gap):
        nonlocal y
        if y < panel_h - 10:
            cv2.putText(canvas, line, (text_x, y), font, scale, color, thickness, cv2.LINE_AA)
        y += gap

    put("OCR Results / Debug", color=(120, 220, 255), scale=0.8, thickness=2, gap=34)
    put(f"Frame: {frame_index}", color=(120, 220, 255), scale=0.72, thickness=2)
    put(f"OCR: {ocr_ms:.1f} ms", color=(120, 220, 255), gap=small_gap)
    put(f"ROI: {record.roi_found}", color=(120, 255, 160) if record.roi_found else (80, 160, 255), gap=small_gap)
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
    return canvas


def _show_debug_windows(display: DisplayState, window_scale: float) -> bool:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("未安装 OpenCV GUI 版本，无法显示调试窗口。") from exc

    try:
        color_bgr = cv2.cvtColor(np.asarray(display.original), cv2.COLOR_RGB2BGR)
        color_bgr = _draw_roi_quad(color_bgr, display.record.roi_quad)
        color_bgr = _draw_roi_debug_overlay(color_bgr, display.preprocess.roi_debug)

        gray = cv2.cvtColor(np.asarray(display.original), cv2.COLOR_RGB2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray_bgr = _draw_roi_quad(gray_bgr, display.record.roi_quad)

        prepared_bgr = cv2.cvtColor(np.asarray(display.preprocess.prepared), cv2.COLOR_GRAY2BGR)
        text_panel = _build_text_panel(
            display.record,
            display.frame_index,
            display.ocr_ms,
            panel_size=(color_bgr.shape[1], color_bgr.shape[0]),
        )

        panel_w = color_bgr.shape[1]
        panel_h = color_bgr.shape[0]
        color_panel = _put_panel_title(_fit_panel(color_bgr, (panel_w, panel_h)), "Color Original")
        gray_panel = _put_panel_title(_fit_panel(gray_bgr, (panel_w, panel_h)), "Gray Original")
        binary_panel = _put_panel_title(_fit_panel(prepared_bgr, (panel_w, panel_h)), "OCR Input")
        info_panel = _put_panel_title(_fit_panel(text_panel, (panel_w, panel_h)), "OCR / Debug")

        top_row = np.hstack([color_panel, gray_panel])
        bottom_row = np.hstack([binary_panel, info_panel])
        dashboard = np.vstack([top_row, bottom_row])
        if window_scale != 1.0:
            dashboard = cv2.resize(dashboard, None, fx=window_scale, fy=window_scale, interpolation=cv2.INTER_AREA)

        cv2.imshow("robocon_ocr_dashboard", dashboard)
        key = cv2.waitKey(1) & 0xFF
        return key not in {27, ord("q"), ord("Q")}
    except cv2.error as exc:
        raise RuntimeError(
            "当前 OpenCV 构建不支持窗口显示。请安装带 GUI 的 `opencv-python`，不要使用 headless 版本。"
        ) from exc


def _save_camera_frame(image_rgb: Image.Image, save_path: Path | None) -> None:
    if save_path is None:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image_rgb.save(save_path)


def _empty_camera_record(image_name: str, preprocess: PreprocessResult) -> PipelineRecord:
    return PipelineRecord(
        image_name=image_name,
        ocr=OCRResult(raw_text="", confidence=0.0, lines=[], psm=None, error="roi not found"),
        parsed=ParsedExpression("", "", None, False, "roi not found"),
        label=None,
        roi_found=preprocess.roi_found,
        roi_quad=preprocess.roi_quad,
    )


def _process_camera_frame(
    frame_bgr: np.ndarray,
    frame_index: int,
    args: argparse.Namespace,
    pipeline_config: PipelineConfig,
    recognizer: TesseractMathRecognizer,
) -> tuple[PipelineRecord, PreprocessResult, Image.Image, float]:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("未安装 OpenCV。请先执行 `pip install -r requirements.txt`。") from exc

    image_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    started_at = time.perf_counter()
    preprocess_result = prepare_image_for_ocr(image_rgb, pipeline_config.preprocess)
    if pipeline_config.debug_dir is not None:
        save_debug_images(
            f"camera_{args.device_index}_{frame_index:06d}.png",
            preprocess_result.cropped,
            preprocess_result.prepared,
            pipeline_config.debug_dir,
            rectified=preprocess_result.rectified,
        )
    _save_camera_frame(image_rgb, args.save_frame.expanduser() if args.save_frame else None)

    if not preprocess_result.roi_found:
        record = _empty_camera_record(f"camera_{args.device_index}_{frame_index:06d}.png", preprocess_result)
        return record, preprocess_result, image_rgb, (time.perf_counter() - started_at) * 1000.0

    ocr_candidates = recognizer.recognize_candidates(preprocess_result.prepared)
    ocr_result, parsed = _select_best_result(ocr_candidates)
    record = PipelineRecord(
        image_name=f"camera_{args.device_index}_{frame_index:06d}.png",
        ocr=ocr_result,
        parsed=parsed,
        label=None,
        roi_found=True,
        roi_quad=preprocess_result.roi_quad,
    )
    return record, preprocess_result, image_rgb, (time.perf_counter() - started_at) * 1000.0


def _run_async_camera(args: argparse.Namespace) -> int:
    camera = USBCameraCapture(build_camera_config(args))
    pipeline_config = PipelineConfig(
        dataset_dir=Path("."),
        debug_dir=args.debug_dir.expanduser() if args.debug_dir else None,
    )
    recognizer = TesseractMathRecognizer(pipeline_config.ocr)
    buffer = LatestFrameBuffer()
    stats = CameraStats()
    stop_event = threading.Event()
    display_lock = threading.Lock()
    display_state: DisplayState | None = None
    last_signature = None
    last_processed_index = -1
    processing_error: Exception | None = None

    def capture_worker() -> None:
        try:
            for frame_index, frame_bgr in camera.stream_raw_frames():
                if stop_event.is_set():
                    break
                stats.frames_captured += 1
                buffer.publish(frame_index, frame_bgr)
        except Exception as exc:
            buffer.stop(exc)
        else:
            buffer.stop()

    def ocr_worker() -> None:
        nonlocal display_state, last_signature, last_processed_index, processing_error
        try:
            while not stop_event.is_set():
                item = buffer.wait_for_next(last_processed_index)
                if item is None:
                    break
                frame_index, frame_bgr, _captured_at = item
                if last_processed_index >= 0 and frame_index > last_processed_index + 1:
                    stats.frames_dropped_stale += frame_index - last_processed_index - 1
                record, preprocess_result, image_rgb, ocr_ms = _process_camera_frame(
                    frame_bgr=frame_bgr,
                    frame_index=frame_index,
                    args=args,
                    pipeline_config=pipeline_config,
                    recognizer=recognizer,
                )
                last_processed_index = frame_index
                if not preprocess_result.roi_found:
                    stats.frames_skipped_no_roi += 1
                else:
                    stats.frames_processed += 1
                    signature = _camera_signature(record)
                    if args.print_all or signature != last_signature:
                        print_camera_record(record, frame_index)
                        last_signature = signature
                        stats.frames_emitted += 1
                with display_lock:
                    display_state = DisplayState(
                        original=image_rgb,
                        preprocess=preprocess_result,
                        record=record,
                        frame_index=frame_index,
                        ocr_ms=ocr_ms,
                    )
        except Exception as exc:
            processing_error = exc
            stop_event.set()
            buffer.stop(exc)

    capture_thread = threading.Thread(target=capture_worker, name="camera-capture", daemon=True)
    ocr_thread = threading.Thread(target=ocr_worker, name="camera-ocr", daemon=True)
    capture_thread.start()
    ocr_thread.start()

    try:
        if args.show_window:
            while capture_thread.is_alive() or ocr_thread.is_alive():
                with display_lock:
                    current_display = display_state
                if current_display is not None:
                    keep_running = _show_debug_windows(current_display, args.window_scale)
                    if not keep_running:
                        stop_event.set()
                        buffer.stop()
                        break
                else:
                    time.sleep(0.01)
        capture_thread.join()
        ocr_thread.join()
    except KeyboardInterrupt:
        stop_event.set()
        buffer.stop()
        capture_thread.join()
        ocr_thread.join()
        print("\n[camera] stopped by user")
    finally:
        if args.show_window:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass

    if processing_error is not None:
        raise processing_error

    print("[summary]")
    print(f"  frames_captured: {stats.frames_captured}")
    print(f"  frames_processed: {stats.frames_processed}")
    print(f"  frames_skipped_no_roi: {stats.frames_skipped_no_roi}")
    print(f"  frames_dropped_stale: {stats.frames_dropped_stale}")
    print(f"  frames_emitted: {stats.frames_emitted}")
    return 0


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
        return 0

    return _run_async_camera(args)
