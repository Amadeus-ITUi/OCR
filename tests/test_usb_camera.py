from __future__ import annotations

import subprocess
from types import SimpleNamespace

from robocon_ocr.camera_tuning import DEFAULT_CAMERA_TUNING
from robocon_ocr.vision_capture.usb_camera import USBCameraCapture


def test_open_capture_sets_stream_properties_before_camera_controls(monkeypatch):
    capture_wrapper = USBCameraCapture(DEFAULT_CAMERA_TUNING)
    calls: list[tuple[str, object, object]] = []

    class FakeCapture:
        def isOpened(self) -> bool:
            calls.append(("isOpened", None, None))
            return True

        def set(self, prop: int, value: float) -> bool:
            calls.append(("set", prop, value))
            return True

    fake_cv2 = SimpleNamespace(
        CAP_PROP_FRAME_WIDTH=1,
        CAP_PROP_FRAME_HEIGHT=2,
        CAP_PROP_FPS=3,
        CAP_PROP_FOURCC=4,
        VideoCapture=lambda _index: FakeCapture(),
        VideoWriter_fourcc=lambda *_args: 1196444237,
    )

    monkeypatch.setattr(capture_wrapper, "_import_cv2", lambda: fake_cv2)
    monkeypatch.setattr(capture_wrapper, "_apply_camera_controls", lambda: calls.append(("controls", None, None)))

    capture_wrapper._open_capture()

    assert calls[:5] == [
        ("isOpened", None, None),
        ("set", 1, float(DEFAULT_CAMERA_TUNING.width)),
        ("set", 2, float(DEFAULT_CAMERA_TUNING.height)),
        ("set", 3, float(DEFAULT_CAMERA_TUNING.fps)),
        ("set", 4, float(1196444237)),
    ]
    assert calls[5] == ("controls", None, None)


def test_camera_controls_are_applied_in_fixed_order():
    capture = USBCameraCapture(DEFAULT_CAMERA_TUNING)

    controls = capture._camera_controls_in_order()

    assert [name for name, _value in controls] == [
        "white_balance_automatic",
        "white_balance_temperature",
        "focus_automatic_continuous",
        "focus_absolute",
        "auto_exposure",
        "exposure_dynamic_framerate",
        "exposure_time_absolute",
        "gain",
        "brightness",
        "contrast",
        "saturation",
        "sharpness",
        "gamma",
        "backlight_compensation",
        "power_line_frequency",
    ]


def test_apply_camera_controls_warns_but_continues_on_failure(monkeypatch, capsys):
    capture = USBCameraCapture(DEFAULT_CAMERA_TUNING)
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr("robocon_ocr.vision_capture.usb_camera.platform.system", lambda: "Linux")
    monkeypatch.setattr("robocon_ocr.vision_capture.usb_camera.shutil.which", lambda _name: "/usr/bin/v4l2-ctl")

    def fake_run(control_name: str, value: int):
        calls.append((control_name, value))
        if control_name == "focus_absolute":
            return subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout="",
                stderr="unknown control 'focus_absolute'",
            )
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(capture, "_run_v4l2_set_ctrl", fake_run)

    capture._apply_camera_controls()

    assert calls[0][0] == "white_balance_automatic"
    assert calls[1][0] == "white_balance_temperature"
    assert calls[2][0] == "focus_automatic_continuous"
    assert calls[3][0] == "focus_absolute"
    assert calls[-1][0] == "power_line_frequency"
    assert len(calls) == len(capture._camera_controls_in_order())
    captured = capsys.readouterr()
    assert "设备不支持控制项 `focus_absolute`" in captured.err


def test_apply_camera_controls_warns_when_v4l2_ctl_is_missing(monkeypatch, capsys):
    capture = USBCameraCapture(DEFAULT_CAMERA_TUNING)
    monkeypatch.setattr("robocon_ocr.vision_capture.usb_camera.platform.system", lambda: "Linux")
    monkeypatch.setattr("robocon_ocr.vision_capture.usb_camera.shutil.which", lambda _name: None)

    capture._apply_camera_controls()

    captured = capsys.readouterr()
    assert "未找到 v4l2-ctl" in captured.err
