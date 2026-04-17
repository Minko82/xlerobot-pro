"""
Shared camera + YOLO detection module.

Provides a lazy-singleton RealSense camera (depth enabled) and YOLO_WORLD
inference helpers.  Importable from both the Jupyter notebook and the
record_with_yolo.py script.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

_ARPA_VISION_ROOT = "/home/xle/arpa_vision"
_ARPA_VISION_WEIGHTS = (
    "/home/xle/arpa_vision/arpa_vision/scripts/yolov8x-worldv2_best.pt"
)
_EV_BATTERY_QUERIES = [
    "Bolt", "BusBar", "InteriorScrew", "Nut", "OrangeCover", "Screw", "Screw Hole",
]

# ---------------------------------------------------------------------------
# Camera singleton
# ---------------------------------------------------------------------------

_camera = None
_camera_error: str = ""


def reset_camera() -> None:
    """Disconnect and clear the camera singleton (forces re-init on next use)."""
    global _camera, _camera_error
    if _camera is not None:
        try:
            _camera.disconnect()
        except Exception:
            pass
        _camera = None
    _camera_error = ""


def init_camera() -> None:
    global _camera, _camera_error
    if _camera is not None:
        return
    use_realsense = os.getenv("USE_REALSENSE", "true").lower() in ("1", "true", "yes")
    if use_realsense:
        _init_realsense()
    else:
        _init_opencv()


def _init_realsense() -> None:
    global _camera, _camera_error
    try:
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        serial = os.getenv("REALSENSE_SERIAL", "")
        if not serial:
            cameras = RealSenseCamera.find_cameras()
            if not cameras:
                raise RuntimeError("No RealSense cameras found.")
            serial = str(cameras[0]["id"])

        config = RealSenseCameraConfig(
            serial_number_or_name=serial,
            fps=30,
            width=640,
            height=480,
            use_depth=True,
        )
        cam = RealSenseCamera(config)
        cam.connect(warmup=True)
        _camera = cam
        logger.info("RealSense camera opened with depth enabled.")
    except Exception as exc:
        _camera_error = f"RealSense init failed: {exc}"
        logger.warning(_camera_error + " — falling back to OpenCV.")
        _init_opencv()


def _init_opencv() -> None:
    global _camera, _camera_error
    index = int(os.getenv("OPENCV_CAMERA_INDEX", "0"))
    try:
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        config = OpenCVCameraConfig(index_or_path=index, fps=30, width=640, height=480)
        cam = OpenCVCamera(config)
        cam.connect(warmup=True)
        _camera = cam
        logger.info(f"OpenCV camera (lerobot) opened at index {index}.")
        return
    except Exception as exc:
        logger.warning(f"lerobot OpenCVCamera failed ({exc}) — trying raw cv2.")

    import cv2 as _cv2

    class _CV2Camera:
        use_depth = False

        def __init__(self, idx: int):
            self._cap = _cv2.VideoCapture(idx)
            if not self._cap.isOpened():
                raise RuntimeError(f"cv2.VideoCapture({idx}) failed.")
            for _ in range(5):
                self._cap.read()

        def read(self) -> np.ndarray:
            ret, frame_bgr = self._cap.read()
            if not ret or frame_bgr is None:
                raise RuntimeError("cv2 frame read failed.")
            return _cv2.cvtColor(frame_bgr, _cv2.COLOR_BGR2RGB)

        def disconnect(self):
            self._cap.release()

    try:
        _camera = _CV2Camera(index)
        logger.info(f"Raw cv2 camera opened at index {index}.")
    except Exception as exc:
        _camera_error = f"OpenCV camera init failed: {exc}"
        logger.error(_camera_error)


# ---------------------------------------------------------------------------
# Frame capture
# ---------------------------------------------------------------------------

def capture_frame() -> tuple[np.ndarray | None, np.ndarray | None, str, str]:
    """
    Capture one RGB frame and (if available) a depth map.

    Returns:
        (frame_rgb, depth_map, frame_b64, error)
        depth_map is (H, W) uint16 in millimetres, or None if unavailable.
    """
    init_camera()
    if _camera is None:
        return None, None, "", _camera_error or "Camera not available."
    try:
        frame_rgb = _camera.read()

        depth_map = None
        if getattr(_camera, "use_depth", False) and hasattr(_camera, "frame_lock"):
            with _camera.frame_lock:
                raw = _camera.latest_depth_frame
                if raw is not None:
                    depth_map = raw.copy()

        frame_b64 = _frame_to_base64(frame_rgb)
        return frame_rgb, depth_map, frame_b64, ""
    except Exception as exc:
        error = f"Frame capture failed: {exc}"
        logger.error(error)
        return None, None, "", error


def _frame_to_base64(frame_rgb: np.ndarray) -> str:
    img = PILImage.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# YOLO singleton
# ---------------------------------------------------------------------------

_yolo_model = None
_yolo_error: str = ""


def get_yolo_model():
    global _yolo_model, _yolo_error
    if _yolo_model is not None:
        return _yolo_model, ""
    if _yolo_error:
        return None, _yolo_error

    try:
        import torch
        if _ARPA_VISION_ROOT not in sys.path:
            sys.path.insert(0, _ARPA_VISION_ROOT)
        from arpa_vision.scripts.BoundingBoxDetectors import YOLO_WORLD

        weight_path = os.getenv("YOLO_MODEL", _ARPA_VISION_WEIGHTS)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        try:
            _yolo_model = YOLO_WORLD(weight_file_path=weight_path)
        except Exception as cuda_exc:
            logger.warning(f"CUDA load failed ({cuda_exc}); falling back to CPU.")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            _orig = torch.cuda.is_available
            torch.cuda.is_available = lambda: False
            try:
                _yolo_model = YOLO_WORLD(weight_file_path=weight_path)
            finally:
                torch.cuda.is_available = _orig

        return _yolo_model, ""
    except Exception as exc:
        _yolo_error = f"YOLO_WORLD load failed: {exc}"
        logger.error(_yolo_error)
        return None, _yolo_error


def _build_queries(target_object: str) -> list[str]:
    if not target_object:
        return _EV_BATTERY_QUERIES
    requested = {lbl.strip().lower() for lbl in target_object.split(",") if lbl.strip()}
    matched = [q for q in _EV_BATTERY_QUERIES if q.lower() in requested]
    return matched if matched else [lbl.strip() for lbl in target_object.split(",") if lbl.strip()]


def _depth_at_center(
    depth_map: np.ndarray, cx: float, cy: float, patch_r: int = 5
) -> float | None:
    """Median depth (metres) in a patch around (cx, cy). None if no valid returns."""
    y0 = max(0, int(cy) - patch_r)
    y1 = min(depth_map.shape[0], int(cy) + patch_r + 1)
    x0 = max(0, int(cx) - patch_r)
    x1 = min(depth_map.shape[1], int(cx) + patch_r + 1)
    patch = depth_map[y0:y1, x0:x1].astype(np.float32)
    valid = patch[patch > 0]
    return round(float(np.median(valid)) / 1000.0, 4) if len(valid) > 0 else None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def run_yolo_detection(
    frame_rgb: np.ndarray,
    target_object: str,
    confidence_threshold: float = 0.20,
    depth_map: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    """
    Run YOLO_WORLD on frame_rgb.

    Returns (detections, annotated_frame_b64, error).
    Each detection has: id, label, confidence, bbox_px, x, y, z.
      x/y: normalised image coords (-0.5 … +0.5, 0,0 = centre).
      z:   metres from camera, or None if depth unavailable.
    """
    model, err = get_yolo_model()
    if model is None:
        return [], "", err

    try:
        queries = _build_queries(target_object)
        candidates = model.predict(frame_rgb, queries=queries, debug=False)

        detections: list[dict[str, Any]] = []
        h, w = frame_rgb.shape[:2]
        annotated = frame_rgb.copy()
        det_idx = 0

        for query, preds in candidates.items():
            for box, prob in zip(preds["boxes"], preds["probs"]):
                if prob < confidence_threshold:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                z_m = _depth_at_center(depth_map, cx, cy) if depth_map is not None else None
                label_id = query.lower().replace(" ", "_")
                detections.append({
                    "id": f"{label_id}_{det_idx}",
                    "label": query,
                    "confidence": round(prob, 3),
                    "bbox_px": [x1, y1, x2, y2],
                    "x": round(cx / w - 0.5, 4),
                    "y": round(cy / h - 0.5, 4),
                    "z": z_m,
                })
                z_label = f"{z_m:.3f}m" if z_m is not None else "z=?"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated, f"{query} {prob:.2f} {z_label}", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )
                det_idx += 1

        return detections, _frame_to_base64(annotated), ""
    except Exception as exc:
        error = f"YOLO inference failed: {exc}"
        logger.error(error)
        return [], "", error


# ---------------------------------------------------------------------------
# High-level observe helper (used by notebook & record script)
# ---------------------------------------------------------------------------

MAX_SCREWS = 5  # maximum detections stored in the fixed-size array


def observe_screws(
    target_object: str = "screw",
    confidence_threshold: float = 0.25,
) -> dict[str, Any]:
    """
    Capture a frame, run YOLO, return detections plus a fixed-size xyz array.

    The 'screw_xyz' field is a float32 array of shape (MAX_SCREWS * 3,):
        [x0, y0, z0, x1, y1, z1, ..., xN, yN, zN, 0, 0, 0, ...]
    Zero-padding fills any unfilled slots.  The policy can learn to ignore
    zero-padded entries (typically the first few will always be populated).
    """
    frame_rgb, depth_map, frame_b64, cap_err = capture_frame()
    if frame_rgb is None:
        return {
            "error": cap_err,
            "detected": [],
            "count": 0,
            "screw_xyz": np.zeros(MAX_SCREWS * 3, dtype=np.float32),
            "frame_b64": "",
            "depth_status": "unavailable",
        }

    detections, annotated_b64, yolo_err = run_yolo_detection(
        frame_rgb, target_object, confidence_threshold, depth_map=depth_map
    )

    # Build fixed-size xyz array
    xyz = np.zeros(MAX_SCREWS * 3, dtype=np.float32)
    for i, det in enumerate(detections[:MAX_SCREWS]):
        xyz[i * 3 + 0] = det["x"]
        xyz[i * 3 + 1] = det["y"]
        xyz[i * 3 + 2] = det["z"] if det["z"] is not None else 0.0

    return {
        "detected": detections,
        "count": len(detections),
        "screw_xyz": xyz,
        "frame_b64": annotated_b64 or frame_b64,
        "depth_status": "enabled" if depth_map is not None else "unavailable",
        "error": yolo_err or cap_err,
    }
