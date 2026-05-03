from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from robocon_ocr.config import CameraConfig


def _build_fourcc(cv2_module, pixel_format: str) -> int:
    normalized = pixel_format.strip().upper()
    if len(normalized) != 4:
        raise ValueError("pixel_format must be a four-character code such as MJPG or YUYV")
    return cv2_module.VideoWriter_fourcc(*normalized)


class USBCameraCapture:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config

    def _import_cv2(self):
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "未安装 OpenCV。请先执行 `pip install -r requirements.txt`。"
            ) from exc
        return cv2

    def _open_capture(self):
        cv2 = self._import_cv2()
        capture = cv2.VideoCapture(self.config.device_index)
        if not capture.isOpened():
            raise RuntimeError(f"无法打开 USB 摄像头设备 /dev/video{self.config.device_index}")

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.config.width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.height))
        capture.set(cv2.CAP_PROP_FPS, float(self.config.fps))
        capture.set(cv2.CAP_PROP_FOURCC, float(_build_fourcc(cv2, self.config.pixel_format)))
        return cv2, capture

    def capture_frame(self) -> Image.Image:
        cv2, capture = self._open_capture()
        try:
            return self._read_stable_frame(cv2, capture, discard_frames=max(1, self.config.warmup_frames))
        finally:
            capture.release()

    def stream_frames(self) -> Iterator[tuple[int, Image.Image]]:
        cv2, capture = self._open_capture()
        try:
            frame_index = 0
            warmup = max(1, self.config.warmup_frames)
            yield frame_index, self._read_stable_frame(cv2, capture, discard_frames=warmup)
            frame_index += 1

            while self.config.max_frames is None or frame_index < self.config.max_frames:
                if self.config.interval_ms > 0:
                    time.sleep(self.config.interval_ms / 1000.0)
                yield frame_index, self._read_stable_frame(cv2, capture, discard_frames=1)
                frame_index += 1
        finally:
            capture.release()

    def _read_stable_frame(self, cv2_module, capture, discard_frames: int) -> Image.Image:
        deadline = time.monotonic() + (self.config.capture_timeout_ms / 1000.0)
        frame = None
        frames_to_read = max(1, discard_frames)

        while time.monotonic() < deadline:
            ok, raw_frame = capture.read()
            if not ok:
                continue
            frame = raw_frame
            frames_to_read -= 1
            if frames_to_read <= 0:
                break

        if frame is None or frames_to_read > 0:
            raise RuntimeError("在超时时间内未能从 USB 摄像头读取到稳定画面")

        rgb_frame = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        if self.config.save_frame is not None:
            save_path = Path(self.config.save_frame).expanduser()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(save_path)
        return image
