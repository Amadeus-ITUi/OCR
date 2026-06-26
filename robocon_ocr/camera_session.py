from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from robocon_ocr.camera_tuning import DEFAULT_CAMERA_TUNING
from robocon_ocr.config import CameraConfig, OCRConfig, PipelineConfig
from robocon_ocr.image_recognition.factory import create_recognizer
from robocon_ocr.recognition_filter import RecognitionFilter
from robocon_ocr.recognition_output import RecognitionOutput
from robocon_ocr.staged_pipeline import context_to_record, run_camera_pipeline_frame
from robocon_ocr.vision_capture.usb_camera import USBCameraCapture


# ---------------------------------------------------------------------------
# Shared frame buffer (also used by cli.py)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LatestFrameBuffer:
    """Thread-safe single-slot buffer for the most recent camera frame."""

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


# ---------------------------------------------------------------------------
# Camera session
# ---------------------------------------------------------------------------


class CameraRecognitionSession:
    """High-level session for real-time camera OCR recognition.

    Parent projects use this as the single entry point. Three backends are
    supported: ``"onnx"`` (default), ``"lightweight"``, ``"api"``.

    **Callback style** (recommended)::

        def on_result(output: RecognitionOutput) -> None:
            print(f"answer={output.answer}  mod4={output.answer_mod_4}")

        session = CameraRecognitionSession(backend="onnx", on_result=on_result)
        session.start()  # blocks until stop() or KeyboardInterrupt

    **Semaphore / polling style**::

        session = CameraRecognitionSession(backend="onnx")
        session.start_async()
        while True:
            if session.result_ready.wait(timeout=1.0):
                output = session.latest_output
                print(f"answer={output.answer}  mod4={output.answer_mod_4}")
                session.result_ready.clear()
    """

    def __init__(
        self,
        backend: str = "onnx",
        on_result: Callable[[RecognitionOutput], None] | None = None,
        filter_consensus: int = 3,
        camera_config: CameraConfig | None = None,
        **camera_overrides,
    ) -> None:
        """
        Args:
            backend: One of ``"onnx"``, ``"lightweight"``, ``"api"``.
            on_result: Optional callback invoked for each confirmed result.
            filter_consensus: Consecutive-match threshold for local models
                (ignored for ``api`` backend).
            camera_config: Optional full ``CameraConfig``; defaults to
                ``DEFAULT_CAMERA_TUNING`` merged with any ``camera_overrides``.
            **camera_overrides: Individual camera-config overrides, e.g.
                ``device_index=2``, ``width=1280``, ``height=720``.
        """
        if backend not in {"onnx", "lightweight", "api"}:
            raise ValueError(f"Unsupported backend: {backend}")

        self._backend = backend
        ocr_config = OCRConfig(backend=backend, warmup=True)
        self._pipeline_config = PipelineConfig(dataset_dir=Path("."), ocr=ocr_config)

        # Camera config: defaults + overrides
        if camera_config is None:
            camera_config = CameraConfig()
        for attr_name in dir(DEFAULT_CAMERA_TUNING):
            if attr_name.startswith("_"):
                continue
            default_val = getattr(DEFAULT_CAMERA_TUNING, attr_name)
            if getattr(camera_config, attr_name) == getattr(CameraConfig(), attr_name):
                setattr(camera_config, attr_name, default_val)
        for key, value in camera_overrides.items():
            if hasattr(camera_config, key):
                setattr(camera_config, key, value)
        self._camera_config = camera_config

        self._on_result = on_result
        self._filter = RecognitionFilter(
            backend=backend,
            consensus=filter_consensus,
            on_emit=self._on_emit,
        )

        # Public result access
        self.result_ready = threading.Event()
        self.latest_output: RecognitionOutput | None = None
        self._output_lock = threading.Lock()

        # Internal state
        self._recognizer = create_recognizer(ocr_config)
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the camera session (blocking). Runs until ``stop()`` is
        called from another thread or ``KeyboardInterrupt``."""
        if self._running:
            return
        self._run()

    def start_async(self) -> None:
        """Start the camera session in a background daemon thread."""
        if self._running:
            return
        self._thread = threading.Thread(target=self._run, name="camera-session", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the session to stop and wait for the background thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_emit(self, output: RecognitionOutput) -> None:
        with self._output_lock:
            self.latest_output = output
            self.result_ready.set()
        if self._on_result:
            self._on_result(output)

    def _run(self) -> None:
        self._running = True
        camera = USBCameraCapture(self._camera_config)
        self._recognizer.warmup()
        buffer = LatestFrameBuffer()

        last_processed_index = -1

        def capture_worker() -> None:
            try:
                for frame_index, frame_bgr in camera.stream_raw_frames():
                    if self._stop_event.is_set():
                        break
                    buffer.publish(frame_index, frame_bgr)
            except Exception as exc:
                buffer.stop(exc)
            else:
                buffer.stop()

        def ocr_worker() -> None:
            nonlocal last_processed_index
            try:
                while not self._stop_event.is_set():
                    item = buffer.wait_for_next(last_processed_index)
                    if item is None:
                        break
                    frame_index, frame_bgr, _captured_at = item
                    image_name = f"camera_{self._camera_config.device_index}_{frame_index:06d}.png"
                    image_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    context = run_camera_pipeline_frame(
                        frame=image_rgb,
                        image_name=image_name,
                        config=self._pipeline_config,
                        recognizer=self._recognizer,
                    )
                    last_processed_index = frame_index
                    record = context_to_record(context)
                    self._process_record(record)
            except Exception:
                self._stop_event.set()
                buffer.stop()

        capture_thread = threading.Thread(target=capture_worker, name="cam-cap", daemon=True)
        ocr_thread = threading.Thread(target=ocr_worker, name="cam-ocr", daemon=True)
        capture_thread.start()
        ocr_thread.start()

        try:
            capture_thread.join()
            ocr_thread.join()
        except KeyboardInterrupt:
            self._stop_event.set()
            buffer.stop()
            capture_thread.join()
            ocr_thread.join()
        finally:
            self._running = False

    def _process_record(self, record) -> None:
        parsed = record.parsed
        if parsed is None:
            return

        output = RecognitionOutput(
            expression=parsed.expression or record.ocr.raw_text,
            answer=parsed.answer,
            answer_mod_4=parsed.answer % 4 if parsed.answer is not None else None,
            is_valid=parsed.is_valid and parsed.answer is not None,
            confidence=record.ocr.confidence,
            backend=record.ocr.backend,
            error=parsed.error or record.ocr.error,
        )
        self._filter.feed(output)
