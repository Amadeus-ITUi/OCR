from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from robocon_ocr.config import PreprocessConfig


@dataclass(slots=True)
class ROIDebugInfo:
    roi_found: bool
    failure_reason: str | None
    best_candidate_type: str
    best_candidate_area_ratio: float | None
    best_candidate_edge_strength: float | None
    best_candidate_ratio: float | None
    best_candidate_ratio_error: float | None
    min_area_ratio_threshold: float
    edge_threshold: float
    rectangle_ratio_tolerance: float
    quadrilateral_ratio_tolerance: float
    white_threshold: float
    candidate_count: int


@dataclass(slots=True)
class PreprocessResult:
    roi_found: bool
    roi_quad: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None
    cropped: Image.Image
    rectified: Image.Image
    board_binary: Image.Image
    prepared: Image.Image
    roi_debug: ROIDebugInfo


@dataclass(slots=True)
class _CandidateMetrics:
    candidate_type: str
    area_ratio: float
    edge_strength: float
    ratio: float | None
    ratio_error: float | None
    score: float
    quad: np.ndarray | None = None


def _import_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("未安装 OpenCV。请先执行 `pip install -r requirements.txt`。") from exc
    return cv2


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("roi quad must contain exactly four points")

    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def _clip_quad(points: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = np.asarray(points, dtype=np.float32).copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    return clipped


def _expand_quad(points: np.ndarray, padding: int, width: int, height: int) -> np.ndarray:
    if padding <= 0:
        return _clip_quad(points, width, height)

    center = points.mean(axis=0)
    vectors = points - center
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_lengths = np.where(lengths == 0, 1.0, lengths)
    expanded = points + (vectors / safe_lengths) * float(padding)
    return _clip_quad(expanded, width, height)


def _quad_aspect_ratio(points: np.ndarray) -> float:
    ordered = _order_quad(points)
    top = np.linalg.norm(ordered[1] - ordered[0])
    bottom = np.linalg.norm(ordered[2] - ordered[3])
    left = np.linalg.norm(ordered[3] - ordered[0])
    right = np.linalg.norm(ordered[2] - ordered[1])
    width = max((top + bottom) * 0.5, 1.0)
    height = max((left + right) * 0.5, 1.0)
    return width / height


def _ratio_error(ratio: float, target: float) -> float:
    return abs(ratio - target) / max(target, 1e-6)


def _score_candidate(
    area: float,
    edge_strength: float,
    ratio_error: float,
    frame_area: float,
) -> float:
    area_score = area / max(frame_area, 1.0)
    edge_score = edge_strength / 255.0
    ratio_score = max(0.0, 1.0 - ratio_error)
    return (area_score * 2.0) + edge_score + ratio_score


def _build_fallback_result(original: Image.Image, config: PreprocessConfig) -> PreprocessResult:
    gray = original.convert("L")
    arr = np.asarray(gray)
    threshold = int(arr.mean()) if arr.size else 0
    binary = np.where(arr > threshold, 255, 0).astype(np.uint8)
    board_binary = Image.fromarray(binary, mode="L")
    roi_debug = ROIDebugInfo(
        roi_found=False,
        failure_reason="no contour candidate",
        best_candidate_type="none",
        best_candidate_area_ratio=None,
        best_candidate_edge_strength=None,
        best_candidate_ratio=None,
        best_candidate_ratio_error=None,
        min_area_ratio_threshold=config.min_roi_area_ratio,
        edge_threshold=float(config.edge_threshold),
        rectangle_ratio_tolerance=config.rectangle_ratio_tolerance,
        quadrilateral_ratio_tolerance=config.quadrilateral_ratio_tolerance,
        white_threshold=float(config.white_threshold),
        candidate_count=0,
    )
    return PreprocessResult(
        roi_found=False,
        roi_quad=None,
        cropped=original,
        rectified=original,
        board_binary=board_binary,
        prepared=board_binary,
        roi_debug=roi_debug,
    )


def _build_board_binary(gray: np.ndarray, gradient_u8: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    cv2 = _import_cv2()
    _, bright_mask = cv2.threshold(gray, config.white_threshold, 255, cv2.THRESH_BINARY)
    _, edge_mask = cv2.threshold(gradient_u8, config.edge_threshold, 255, cv2.THRESH_BINARY)
    merged = cv2.bitwise_or(bright_mask, edge_mask)
    kernel = np.ones((5, 5), np.uint8)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel, iterations=2)
    merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, kernel, iterations=1)
    return merged


def _diagnostic_score(area_ratio: float, min_area_ratio: float, edge_strength: float, edge_threshold: float, ratio_error: float | None, tolerance: float | None) -> float:
    area_part = min(area_ratio / max(min_area_ratio, 1e-6), 1.5)
    edge_part = min(edge_strength / max(edge_threshold, 1e-6), 1.5)
    if ratio_error is None or tolerance is None:
        ratio_part = 0.0
    else:
        ratio_part = max(0.0, 1.0 - (ratio_error / max(tolerance, 1e-6)))
    return area_part + edge_part + ratio_part


def _build_roi_debug(
    config: PreprocessConfig,
    candidate_count: int,
    best_candidate: _CandidateMetrics | None,
    roi_found: bool,
) -> ROIDebugInfo:
    failure_reason: str | None = None
    if not roi_found:
        if candidate_count == 0 or best_candidate is None:
            failure_reason = "no contour candidate"
    if best_candidate is None:
        if failure_reason is None and not roi_found:
            failure_reason = "no contour candidate"
        return ROIDebugInfo(
            roi_found=roi_found,
            failure_reason=failure_reason,
            best_candidate_type="none",
            best_candidate_area_ratio=None,
            best_candidate_edge_strength=None,
            best_candidate_ratio=None,
            best_candidate_ratio_error=None,
            min_area_ratio_threshold=config.min_roi_area_ratio,
            edge_threshold=float(config.edge_threshold),
            rectangle_ratio_tolerance=config.rectangle_ratio_tolerance,
            quadrilateral_ratio_tolerance=config.quadrilateral_ratio_tolerance,
            white_threshold=float(config.white_threshold),
            candidate_count=candidate_count,
        )

    if not roi_found:
        if best_candidate.area_ratio < config.min_roi_area_ratio:
            failure_reason = "area below threshold"
        elif best_candidate.edge_strength < float(config.edge_threshold):
            failure_reason = "edge too weak"
        elif best_candidate.candidate_type == "rectangle" and (best_candidate.ratio_error is None or best_candidate.ratio_error > config.rectangle_ratio_tolerance):
            failure_reason = "rectangle ratio mismatch"
        elif best_candidate.candidate_type == "quadrilateral" and (best_candidate.ratio_error is None or best_candidate.ratio_error > config.quadrilateral_ratio_tolerance):
            failure_reason = "quadrilateral ratio mismatch"
        else:
            failure_reason = "no valid roi after filtering"

    return ROIDebugInfo(
        roi_found=roi_found,
        failure_reason=failure_reason,
        best_candidate_type=best_candidate.candidate_type,
        best_candidate_area_ratio=best_candidate.area_ratio,
        best_candidate_edge_strength=best_candidate.edge_strength,
        best_candidate_ratio=best_candidate.ratio,
        best_candidate_ratio_error=best_candidate.ratio_error,
        min_area_ratio_threshold=config.min_roi_area_ratio,
        edge_threshold=float(config.edge_threshold),
        rectangle_ratio_tolerance=config.rectangle_ratio_tolerance,
        quadrilateral_ratio_tolerance=config.quadrilateral_ratio_tolerance,
        white_threshold=float(config.white_threshold),
        candidate_count=candidate_count,
    )


def _find_best_quad(image: Image.Image, config: PreprocessConfig) -> tuple[np.ndarray | None, Image.Image, ROIDebugInfo]:
    cv2 = _import_cv2()
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    gradient_u8 = np.clip(gradient, 0, 255).astype(np.uint8)
    merged = _build_board_binary(blurred, gradient_u8, config)
    board_binary = Image.fromarray(merged, mode="L")
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = image.width * image.height * config.min_roi_area_ratio
    frame_area = float(image.width * image.height)
    best_score = float("-inf")
    best_quad: np.ndarray | None = None
    best_valid_candidate: _CandidateMetrics | None = None
    best_diagnostic_candidate: _CandidateMetrics | None = None

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        contour_mask = np.zeros_like(gray)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=2)
        edge_strength = float(cv2.mean(gradient_u8, mask=contour_mask)[0])
        area_ratio = area / max(frame_area, 1.0)

        rect = cv2.minAreaRect(contour)
        (rect_w, rect_h) = rect[1]
        rect_ratio = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1.0)
        rect_error = _ratio_error(rect_ratio, config.target_aspect_ratio)
        rect_diag = _CandidateMetrics(
            candidate_type="rectangle",
            area_ratio=area_ratio,
            edge_strength=edge_strength,
            ratio=rect_ratio,
            ratio_error=rect_error,
            score=_diagnostic_score(
                area_ratio,
                config.min_roi_area_ratio,
                edge_strength,
                float(config.edge_threshold),
                rect_error,
                config.rectangle_ratio_tolerance,
            ),
            quad=cv2.boxPoints(rect),
        )
        if best_diagnostic_candidate is None or rect_diag.score > best_diagnostic_candidate.score:
            best_diagnostic_candidate = rect_diag

        if area >= min_area and edge_strength >= float(config.edge_threshold) and rect_error <= config.rectangle_ratio_tolerance:
            quad = cv2.boxPoints(rect)
            score = _score_candidate(area, edge_strength, rect_error, frame_area)
            if score > best_score:
                best_score = score
                best_quad = quad
                best_valid_candidate = _CandidateMetrics(
                    candidate_type="rectangle",
                    area_ratio=area_ratio,
                    edge_strength=edge_strength,
                    ratio=rect_ratio,
                    ratio_error=rect_error,
                    score=score,
                    quad=quad,
                )

        if len(approx) != 4:
            continue
        quad = approx.reshape(4, 2).astype(np.float32)
        quad_ratio = _quad_aspect_ratio(quad)
        quad_error = _ratio_error(quad_ratio, config.target_aspect_ratio)
        quad_diag = _CandidateMetrics(
            candidate_type="quadrilateral",
            area_ratio=area_ratio,
            edge_strength=edge_strength,
            ratio=quad_ratio,
            ratio_error=quad_error,
            score=_diagnostic_score(
                area_ratio,
                config.min_roi_area_ratio,
                edge_strength,
                float(config.edge_threshold),
                quad_error,
                config.quadrilateral_ratio_tolerance,
            ),
            quad=quad,
        )
        if best_diagnostic_candidate is None or quad_diag.score > best_diagnostic_candidate.score:
            best_diagnostic_candidate = quad_diag

        if area < min_area or edge_strength < float(config.edge_threshold) or quad_error > config.quadrilateral_ratio_tolerance:
            continue

        score = _score_candidate(area, edge_strength, quad_error, frame_area)
        if score > best_score:
            best_score = score
            best_quad = quad
            best_valid_candidate = _CandidateMetrics(
                candidate_type="quadrilateral",
                area_ratio=area_ratio,
                edge_strength=edge_strength,
                ratio=quad_ratio,
                ratio_error=quad_error,
                score=score,
                quad=quad,
            )

    if best_quad is None:
        return None, board_binary, _build_roi_debug(
            config=config,
            candidate_count=len(contours),
            best_candidate=best_diagnostic_candidate,
            roi_found=False,
        )

    ordered = _order_quad(best_quad)
    expanded = _expand_quad(ordered, config.roi_padding, image.width, image.height)
    return _order_quad(expanded), board_binary, _build_roi_debug(
        config=config,
        candidate_count=len(contours),
        best_candidate=best_valid_candidate or best_diagnostic_candidate,
        roi_found=True,
    )


def _warp_quad(image: Image.Image, quad: np.ndarray, config: PreprocessConfig) -> Image.Image:
    cv2 = _import_cv2()
    rgb = np.asarray(image.convert("RGB"))
    destination = np.array(
        [
            [0, 0],
            [config.perspective_width - 1, 0],
            [config.perspective_width - 1, config.perspective_height - 1],
            [0, config.perspective_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    warped = cv2.warpPerspective(
        rgb,
        matrix,
        (config.perspective_width, config.perspective_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(warped)


def _prepare_rectified_for_ocr(image: Image.Image, config: PreprocessConfig) -> Image.Image:
    cv2 = _import_cv2()
    gray = image.convert("L")
    if config.scale_factor != 1.0:
        gray = gray.resize(
            (
                max(1, int(gray.width * config.scale_factor)),
                max(1, int(gray.height * config.scale_factor)),
            ),
            Image.Resampling.LANCZOS,
        )
    arr = np.asarray(gray)
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return Image.fromarray(binary.astype(np.uint8), mode="L")


def _bounding_crop(image: Image.Image, quad: np.ndarray) -> Image.Image:
    xs = quad[:, 0]
    ys = quad[:, 1]
    x0 = max(0, int(np.floor(xs.min())))
    y0 = max(0, int(np.floor(ys.min())))
    x1 = min(image.width, int(np.ceil(xs.max())) + 1)
    y1 = min(image.height, int(np.ceil(ys.max())) + 1)
    return image.crop((x0, y0, x1, y1))


def detect_roi(image: Image.Image, config: PreprocessConfig) -> tuple[bool, np.ndarray | None]:
    quad, _board_binary, _roi_debug = _find_best_quad(image, config)
    if quad is None:
        return False, None
    return True, quad


def crop_foreground_text(image: Image.Image, config: PreprocessConfig) -> Image.Image:
    roi_found, quad = detect_roi(image, config)
    if not roi_found or quad is None:
        return image
    return _bounding_crop(image, quad)


def prepare_image_for_ocr(
    original: Image.Image,
    config: PreprocessConfig,
) -> PreprocessResult:
    rgb = original.convert("RGB")
    quad, board_binary, roi_debug = _find_best_quad(rgb, config)
    roi_found = quad is not None
    if not roi_found or quad is None:
        fallback = _build_fallback_result(rgb, config)
        fallback.board_binary = board_binary
        fallback.roi_debug = roi_debug
        return fallback

    cropped = _bounding_crop(rgb, quad)
    rectified = _warp_quad(rgb, quad, config)
    prepared = _prepare_rectified_for_ocr(rectified, config)
    roi_quad = tuple((int(point[0]), int(point[1])) for point in quad)
    return PreprocessResult(
        roi_found=True,
        roi_quad=roi_quad,
        cropped=cropped,
        rectified=rectified,
        board_binary=board_binary,
        prepared=prepared,
        roi_debug=roi_debug,
    )


def prepare_for_ocr(image_path: Path, config: PreprocessConfig) -> PreprocessResult:
    original = Image.open(image_path).convert("RGB")
    return prepare_image_for_ocr(original, config)


def save_debug_images(
    image_name: str,
    cropped: Image.Image,
    prepared: Image.Image,
    debug_dir: Path,
    rectified: Image.Image | None = None,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem
    cropped.save(debug_dir / f"{stem}_cropped.png")
    if rectified is not None:
        rectified.save(debug_dir / f"{stem}_rectified.png")
    prepared.save(debug_dir / f"{stem}_prepared.png")
