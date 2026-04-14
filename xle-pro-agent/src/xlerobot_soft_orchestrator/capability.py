"""Robot capability registry.

Three capabilities are exposed to the agent:

  get_observation   — capture a camera frame and (eventually) locate screws
  run_pick          — stub: execute the fine-tuned VLA policy to pick one screw
  run_visual_qa     — stub: post-pick wrist-camera grasp check

get_observation is now wired to real hardware:
  - Default: OpenCV camera (index 0, your dev-machine webcam or any USB cam)
  - Optional: Intel RealSense D435/D435i (set USE_REALSENSE=true in .env)

The camera image is returned as a base64-encoded PNG under the key "frame_b64"
so that the Streamlit UI can render it inline with st.image().

YOLO detection is stubbed — when your model is ready, drop it into
_run_detection() and the rest of the pipeline stays the same.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from langchain_core.tools import tool
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Camera initialisation
# Lazy singleton — camera is opened once on first call to get_observation,
# not at import time, so the agent can still be imported without hardware.
# ---------------------------------------------------------------------------

_camera = None          # lerobot Camera instance
_camera_error: str = "" # set if init failed, surfaced in capability result


def _init_camera() -> None:
    """Open the camera once and cache it.  Tries RealSense first if configured."""
    global _camera, _camera_error

    if _camera is not None:
        return  # already open

    use_realsense = os.getenv("USE_REALSENSE", "false").lower() in ("1", "true", "yes")

    if use_realsense:
        _init_realsense()
    else:
        _init_opencv()


def _init_realsense() -> None:
    """Attempt to open an Intel RealSense camera."""
    global _camera, _camera_error
    try:
        from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

        serial = os.getenv("REALSENSE_SERIAL", "")  # leave blank to use first found
        config = RealSenseCameraConfig(
            serial_number_or_name=serial if serial else _find_first_realsense_serial(),
            fps=30,
            width=640,
            height=480,
        )
        cam = RealSenseCamera(config)
        cam.connect(warmup=True)
        _camera = cam
        logger.info("RealSense camera opened.")
    except Exception as exc:
        _camera_error = f"RealSense init failed: {exc}"
        logger.warning(_camera_error)
        logger.info("Falling back to OpenCV camera.")
        _init_opencv()


def _find_first_realsense_serial() -> str:
    """Return the serial number of the first attached RealSense device."""
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
    cameras = RealSenseCamera.find_cameras()
    if not cameras:
        raise RuntimeError("No RealSense cameras found.")
    return str(cameras[0]["id"])


def _init_opencv() -> None:
    """Open an OpenCV camera at the configured index."""
    global _camera, _camera_error
    try:
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        index = int(os.getenv("OPENCV_CAMERA_INDEX", "0"))
        config = OpenCVCameraConfig(
            index_or_path=index,
            fps=30,
            width=640,
            height=480,
        )
        cam = OpenCVCamera(config)
        cam.connect(warmup=True)
        _camera = cam
        logger.info(f"OpenCV camera opened at index {index}.")
    except Exception as exc:
        _camera_error = f"OpenCV camera init failed: {exc}"
        logger.error(_camera_error)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame_to_base64(frame_rgb: np.ndarray) -> str:
    """Encode an (H, W, 3) uint8 RGB numpy array as a base64 PNG string."""
    img = Image.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _run_detection(frame_rgb: np.ndarray, target_object: str) -> list[dict[str, Any]]:
    """
    Object detection placeholder.

    Replace the body of this function with your YOLO World inference when ready.
    The expected return format is:
        [{"label": str, "id": str, "x": float, "y": float, "z": float,
          "confidence": float, "bbox_px": [x1, y1, x2, y2]}, ...]

    For now we return plausible-looking stub detections so the agent can
    exercise its reasoning loop end-to-end.
    """
    # --- STUB ---
    # When you have the model:
    #   from inference import get_model
    #   model = get_model("yolo-world-l", api_key=os.getenv("ROBOFLOW_API_KEY"))
    #   results = model.infer(frame_rgb, text=target_object, confidence=0.3)[0]
    #   return [ ... parse results.predictions ... ]
    h, w = frame_rgb.shape[:2]
    stub_objects = [
        {"label": target_object, "id": f"{target_object}_0",
         "x": 0.12, "y": -0.05, "z": 0.30, "confidence": 0.93,
         "bbox_px": [w//4, h//4, w//2, h//2]},
        {"label": target_object, "id": f"{target_object}_1",
         "x": 0.22, "y":  0.03, "z": 0.29, "confidence": 0.88,
         "bbox_px": [w//2, h//3, 3*w//4, 2*h//3]},
    ]
    return stub_objects


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Capability:
    id: str
    description: str
    doc: str
    fn: Callable[..., dict[str, Any]]
    required_args: list[str] = field(default_factory=list)
    optional_args: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Capability implementations
# ---------------------------------------------------------------------------

def _get_observation(target_object: str = "screw", **kwargs: Any) -> dict[str, Any]:
    """
    Capture a real camera frame, run detection (stubbed), and return results.

    The returned dict includes:
      frame_b64   — base64-encoded PNG of the captured frame (rendered by the UI)
      detected    — list of detected objects with 3-D positions
      camera_info — which camera backend was used
    """
    # Ensure camera is open
    _init_camera()

    frame_b64: str = ""
    camera_info: str = ""
    detected: list[dict[str, Any]] = []
    error: str = ""

    if _camera is None:
        # Camera could not be opened — return the error so the agent knows
        error = _camera_error or "Camera not available."
        return {
            "capability": "get_observation",
            "target": target_object,
            "detected": [],
            "count": 0,
            "frame_b64": "",
            "camera_info": "none",
            "error": error,
            "note": "Camera unavailable. Check USB connection or set USE_REALSENSE=true.",
        }

    try:
        frame_rgb = _camera.read()          # (H, W, 3) uint8 RGB
        frame_b64 = _frame_to_base64(frame_rgb)
        camera_info = type(_camera).__name__

        # Run detection (stub until YOLO is wired in)
        detected = _run_detection(frame_rgb, target_object)

    except Exception as exc:
        error = f"Frame capture failed: {exc}"
        logger.error(error)

    return {
        "capability": "get_observation",
        "target": target_object,
        "detected": detected,
        "count": len(detected),
        "frame_b64": frame_b64,   # base64 PNG — rendered by ui.py
        "camera_info": camera_info,
        "error": error,
        "note": (
            "Detection results are STUB placeholders. "
            "Wire _run_detection() to YOLO World when model is ready."
        ),
    }


def _run_pick(screw_id: str, x: float, y: float, z: float, **kwargs: Any) -> dict[str, Any]:
    """Stub: execute the fine-tuned VLA pick policy for one screw."""
    success = random.random() > 0.25
    return {
        "capability": "run_pick",
        "screw_id": screw_id,
        "target_position": {"x": x, "y": y, "z": z},
        "status": "SUCCESS" if success else "FAILURE",
        "failure_reason": None if success else "policy did not achieve stable grasp",
        "note": "STUB — real call will run VLA policy inference loop (ACT default)",
    }


def _run_visual_qa(screw_id: str, **kwargs: Any) -> dict[str, Any]:
    """Stub: post-pick wrist-camera grasp confirmation."""
    grasp_confirmed = random.random() > 0.2
    return {
        "capability": "run_visual_qa",
        "screw_id": screw_id,
        "grasp_confirmed": grasp_confirmed,
        "confidence": round(random.uniform(0.70, 0.99) if grasp_confirmed else random.uniform(0.10, 0.45), 2),
        "note": "STUB — real call will run post-pick wrist-camera classifier",
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Capability] = {
    "get_observation": Capability(
        id="get_observation",
        description=(
            "Capture a live camera frame and locate objects (screws, nuts, etc.) in the scene. "
            "Returns a base64 image of the frame and a list of detected objects with 3-D positions."
        ),
        doc="""
Capability: get_observation
----------------------------
Purpose:
  Capture the current camera frame and run object detection. Returns detected
  objects with 3-D positions in the robot-base frame, plus the raw camera image.

Required args:  (none)
Optional args:
  target_object (str, default "screw") — label filter for detection.

Returns:
  detected:     list of { id, label, x, y, z, confidence, bbox_px }
  count:        number of detected objects
  frame_b64:    base64-encoded PNG of the captured frame (displayed in UI)
  camera_info:  which camera backend was used (OpenCVCamera / RealSenseCamera)
  error:        non-empty string if something went wrong

Notes:
  - Detection is currently STUBBED. Positions are plausible but not real.
  - The image IS real — it comes from your physical camera.
  - Wire _run_detection() in capability.py to YOLO World when the model is ready.

Workflow:  START → get_observation → run_pick → run_visual_qa → (loop or done)
""",
        fn=_get_observation,
        optional_args=["target_object"],
    ),

    "run_pick": Capability(
        id="run_pick",
        description="Execute the fine-tuned VLA pick policy to grasp one screw. (STUB)",
        doc="""
Capability: run_pick
---------------------
Purpose:
  Run the fine-tuned VLA (ACT) policy to pick the specified screw.

Required args:
  screw_id (str)    — the "id" field from get_observation output
  x, y, z  (float) — screw position from get_observation (metres, base frame)

Returns:
  status:         "SUCCESS" or "FAILURE"
  failure_reason: string if FAILURE, else null

After this call you MUST call run_visual_qa to verify the grasp.

NOTE: Currently a stub. Real implementation will run the LeRobot ACT policy loop.
""",
        fn=_run_pick,
        required_args=["screw_id", "x", "y", "z"],
    ),

    "run_visual_qa": Capability(
        id="run_visual_qa",
        description="Post-pick wrist-camera check: verify the screw was successfully grasped. (STUB)",
        doc="""
Capability: run_visual_qa
--------------------------
Purpose:
  Capture a wrist-camera frame and determine whether the gripper holds the screw.

Required args:
  screw_id (str) — the screw ID that was just picked

Returns:
  grasp_confirmed (bool)
  confidence      (float) [0, 1]

NOTE: Currently a stub.
""",
        fn=_run_visual_qa,
        required_args=["screw_id"],
    ),
}


# ---------------------------------------------------------------------------
# Agent-facing interface
# ---------------------------------------------------------------------------

def list_capabilities() -> dict[str, Any]:
    """Return a compact index of all available capabilities."""
    return {
        "capabilities": [
            {
                "id": cap.id,
                "description": cap.description,
                "required_args": cap.required_args,
                "optional_args": cap.optional_args,
            }
            for cap in REGISTRY.values()
        ]
    }


def read_capability(capability_id: str) -> dict[str, Any]:
    """Return the full contract/doc for one capability."""
    cap = REGISTRY.get(capability_id)
    if cap is None:
        return {"error": f"Unknown capability: {capability_id!r}. Available: {list(REGISTRY.keys())}"}
    return {
        "id": cap.id,
        "description": cap.description,
        "doc": cap.doc.strip(),
        "required_args": cap.required_args,
        "optional_args": cap.optional_args,
    }


def run_capability(capability_id: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate args and execute a capability by id."""
    cap = REGISTRY.get(capability_id)
    if cap is None:
        return {"error": f"Unknown capability: {capability_id!r}. Available: {list(REGISTRY.keys())}"}
    args = args or {}
    missing = [k for k in cap.required_args if k not in args]
    if missing:
        return {
            "error": f"Missing required args for {capability_id!r}: {missing}",
            "hint": f"Call read_capability('{capability_id}') to see the full contract.",
        }
    try:
        return cap.fn(**args)
    except Exception as exc:
        return {"error": str(exc), "capability_id": capability_id}


# ---------------------------------------------------------------------------
# LangChain tool wrappers
# ---------------------------------------------------------------------------

@tool
def list_capabilities_tool(**kwargs: Any) -> dict[str, Any]:
    """List all available robot capabilities with their required and optional args."""
    return list_capabilities()


@tool
def read_capability_tool(capability_id: str, **kwargs: Any) -> dict[str, Any]:
    """Read the full documentation and argument contract for one capability before using it."""
    return read_capability(capability_id)


@tool
def run_capability_tool(
    capability_id: str,
    args: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute a robot capability with validated arguments.

    Args:
        capability_id: ID from list_capabilities (e.g. "get_observation")
        args: Capability-specific arguments dict.
    """
    merged: dict[str, Any] = {}
    if isinstance(args, dict):
        merged.update(args)
    if isinstance(params, dict):
        merged.update(params)
    skip_keys = {"v__args", "type"}
    wrapper_keys = {"kwargs", "parameters", "payload", "input"}
    for key, value in kwargs.items():
        if key in skip_keys:
            continue
        if key in wrapper_keys and isinstance(value, dict):
            merged.update(value)
        else:
            merged[key] = value

    return run_capability(capability_id, merged or None)


TOOLS = [
    list_capabilities_tool,
    read_capability_tool,
    run_capability_tool,
]