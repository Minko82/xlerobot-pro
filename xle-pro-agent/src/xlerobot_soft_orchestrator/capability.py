"""Robot capability registry.

Capabilities exposed to the agent:

  observe_with_yolo  — capture a frame, run YOLO open-vocabulary detection,
                       return bounding boxes + pixel-space positions.
                       Use this when the task requires locating or picking objects.

  observe_with_vlm   — capture a frame, send it to Gemini Vision, return a
                       natural-language description of the scene.
                       Use this for general "what do I see?" inquiries.

  run_pick           — stub: execute the fine-tuned VLA policy to pick one object
  run_visual_qa      — stub: post-pick wrist-camera grasp check

Camera backends (set in .env):
  USE_REALSENSE=true          → Intel RealSense D435
  REALSENSE_SERIAL=<serial>   → specific device (optional)
  OPENCV_CAMERA_INDEX=<n>     → fallback OpenCV camera index (default 0)

YOLO backend (set in .env):
  YOLO_MODEL=yoloe-26s-seg.pt → model weights (default yoloe-26s-seg.pt)
                                 swap for your colleague's custom weights when ready

VLM backend:
  Uses Gemini via the same Vertex AI credentials already configured in config.py.
  VERTEX_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS must be set.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import cv2
from dataclasses import dataclass, field
from typing import Any, Callable
import random

import numpy as np
from langchain_core.tools import tool
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera — lazy singleton
# ---------------------------------------------------------------------------

_camera = None
_camera_error: str = ""


def _init_camera() -> None:
    global _camera, _camera_error
    if _camera is not None:
        return
    use_realsense = os.getenv("USE_REALSENSE", "false").lower() in ("1", "true", "yes")
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
        logger.warning(_camera_error + " — falling back to OpenCV.")
        _init_opencv()


def _find_first_realsense_serial() -> str:
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
    cameras = RealSenseCamera.find_cameras()
    if not cameras:
        raise RuntimeError("No RealSense cameras found.")
    return str(cameras[0]["id"])


def _init_opencv() -> None:
    global _camera, _camera_error
    index = int(os.getenv("OPENCV_CAMERA_INDEX", "0"))

    # Prefer lerobot wrapper; fall back to raw cv2
    try:
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        config = OpenCVCameraConfig(index_or_path=index, fps=30, width=640, height=480)
        cam = OpenCVCamera(config)
        cam.connect(warmup=True)
        _camera = cam
        logger.info(f"OpenCV camera (lerobot) opened at index {index}.")
        return
    except ImportError:
        logger.info("lerobot not found — trying raw cv2.")
    except Exception as exc:
        logger.warning(f"lerobot OpenCVCamera failed ({exc}) — trying raw cv2.")

    try:
        import cv2

        class _CV2Camera:
            def __init__(self, idx: int):
                self._cap = cv2.VideoCapture(idx)
                if not self._cap.isOpened():
                    raise RuntimeError(f"cv2.VideoCapture({idx}) failed.")
                for _ in range(5):
                    self._cap.read()

            def read(self) -> np.ndarray:
                ret, frame_bgr = self._cap.read()
                if not ret or frame_bgr is None:
                    raise RuntimeError("cv2 frame read failed.")
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            def disconnect(self):
                self._cap.release()

        _camera = _CV2Camera(index)
        logger.info(f"OpenCV camera (raw cv2) opened at index {index}.")
    except Exception as exc:
        _camera_error = f"OpenCV camera init failed: {exc}"
        logger.error(_camera_error)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _capture_frame() -> tuple[np.ndarray | None, str, str]:
    """
    Capture one RGB frame from the camera.
    Returns (frame_rgb, frame_b64, error).
    frame_rgb is None on failure.
    """
    _init_camera()
    if _camera is None:
        return None, "", _camera_error or "Camera not available."
    try:
        frame_rgb = _camera.read()          # (H, W, 3) uint8 RGB
        frame_b64 = _frame_to_base64(frame_rgb)
        return frame_rgb, frame_b64, ""
    except Exception as exc:
        error = f"Frame capture failed: {exc}"
        logger.error(error)
        return None, "", error


def _frame_to_base64(frame_rgb: np.ndarray) -> str:
    """Encode (H, W, 3) uint8 RGB numpy array as a base64 PNG string."""
    img = PILImage.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _frame_to_base64_vlm(frame_rgb: np.ndarray, max_dim: int = 400, jpeg_quality: int = 85) -> str:
    """Resize and JPEG-encode a frame for VLM submission.

    Keeps longest edge ≤ max_dim and encodes as JPEG to minimise token cost.
    PNG is ~3-5x larger and buys nothing for scene-description tasks.
    """
    h, w = frame_rgb.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)   # never upscale
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = PILImage.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
        img = img.resize((new_w, new_h), PILImage.LANCZOS)
    else:
        img = PILImage.fromarray(frame_rgb.astype(np.uint8), mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _camera_unavailable_result(capability: str, extra: dict | None = None) -> dict[str, Any]:
    base = {
        "capability": capability,
        "frame_b64": "",
        "camera_info": "none",
        "error": _camera_error or "Camera not available.",
        "note": "Camera unavailable. Check USB connection or .env settings.",
    }
    if extra:
        base.update(extra)
    return base

# ---------------------------------------------------------------------------
# YOLO — lazy singleton
# ---------------------------------------------------------------------------

_yolo_model = None
_yolo_error: str = ""

def _get_yolo_model():
    """Load and cache the YOLO model, converting to TensorRT if needed."""
    global _yolo_model, _yolo_error
    if _yolo_model is not None:
        return _yolo_model, ""
    if _yolo_error:
        return None, _yolo_error
    
    try:
        from ultralytics import YOLO
        weights = os.getenv("YOLO_MODEL", "yoloe-26s-seg.pt")
        
        # 1. Define the engine path (e.g., yoloe-26s-seg.engine)
        engine_path = weights.replace(".pt", ".engine")
        
        # 2. If the engine doesn't exist, create it (Exporting)
        if not os.path.exists(engine_path):
            logger.info(f"TensorRT engine not found. Exporting {weights} to {engine_path} (FP16)...")
            tmp_model = YOLO(weights)
            # This is the memory-saving magic: half=True (FP16)
            tmp_model.export(format="engine", device=0, half=True, simplify=True)
            del tmp_model # Free PyTorch memory immediately
        
        # 3. Load the optimized engine
        _yolo_model = YOLO(engine_path, task="segment")
        logger.info(f"YOLO TensorRT model loaded: {engine_path}")
        return _yolo_model, ""

    except Exception as exc:
        _yolo_error = f"YOLO TensorRT load/export failed: {exc}"
        logger.error(_yolo_error)
        return None, _yolo_error


def _run_yolo_detection(
    frame_rgb: np.ndarray,
    target_object: str,
    confidence_threshold: float = 0.20,
) -> tuple[list[dict[str, Any]], str, str]:
    """
    Run YOLO inference on frame_rgb.

    For open-vocabulary detection (YOLO-World), the target_object label is used
    as the text prompt.  For standard YOLO models the label is used as a
    post-inference filter — any detection whose class name contains
    target_object (case-insensitive) is kept; pass target_object="" to keep all.

    Returns (detections, annotated_frame_b64, error).
    annotated_frame_b64 is a base64 PNG with YOLO bounding boxes drawn on it.
    """
    model, err = _get_yolo_model()
    if model is None:
        return [], "", err

    try:
        import cv2

        # YOLOE-26 / YOLO-World: set text prompt classes before inference.
        # set_classes only needs calling when the target changes — safe to call every time.
        if hasattr(model, "set_classes"):
            labels = [lbl.strip() for lbl in target_object.split(",") if lbl.strip()]
            model.set_classes(labels if labels else ["object"])
        elif target_object:
            logger.warning(
                f"Model does not support set_classes — '{target_object}' will be "
                "used as a post-inference label filter only. Switch to yoloe-26s-seg.pt."
            )

        results = model.predict(frame_rgb, conf=confidence_threshold, verbose=False)

        detections: list[dict[str, Any]] = []
        annotated_b64 = ""

        for r in results:
            annotated_bgr = r.plot()
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            annotated_b64 = _frame_to_base64(annotated_rgb)

            # YOLOE returns masks (seg model) as well as boxes — use boxes for position
            boxes = r.boxes
            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                label = model.names[cls_id] if model.names else str(cls_id)

                # For non-world models, post-filter by target label
                if target_object and not hasattr(model, "set_classes"):
                    if target_object.lower() not in label.lower():
                        continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                h, w = frame_rgb.shape[:2]

                detections.append({
                    "id": f"{label}_{i}",
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox_px": [x1, y1, x2, y2],
                    "x": round((cx / w - 0.5), 4),
                    "y": round((cy / h - 0.5), 4),
                    "z": None,
                    "note": "x/y are normalised image coords, not metric. Wire depth for real 3-D.",
                })

        return detections, annotated_b64, ""

    except Exception as exc:
        error = f"YOLO inference failed: {exc}"
        logger.error(error)
        return [], "", error


# ---------------------------------------------------------------------------
# VLM (Gemini Vision) helper
# ---------------------------------------------------------------------------

def _run_vlm_description(frame_b64: str, question: str) -> tuple[str, str]:
    """
    Send a base64-encoded PNG to Gemini Vision and return (description, error).

    Uses the same Vertex AI credentials already configured in config.py
    (GOOGLE_APPLICATION_CREDENTIALS + VERTEX_PROJECT_ID).
    """
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part, Image as VertexImage

        project = os.getenv("VERTEX_PROJECT_ID", "")
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        vlm_model = os.getenv("VLM_MODEL", "gemini-2.5-flash-lite")

        if not project:
            return "", "VERTEX_PROJECT_ID not set in .env"

        vertexai.init(project=project, location=location)
        model = GenerativeModel(vlm_model)

        image_bytes = base64.b64decode(frame_b64)
        image_part = Part.from_data(data=image_bytes, mime_type="image/jpeg")
        text_part = question or (
            "Describe this robot workspace image in detail. "
            "List all visible objects, their approximate positions relative to each other, "
            "and anything relevant to a pick-and-place robot task."
        )

        response = model.generate_content([image_part, text_part])
        description = response.text.strip()
        return description, ""

    except ImportError:
        error = "vertexai SDK not installed. Run: pip install google-cloud-aiplatform"
        logger.error(error)
        return "", error
    except Exception as exc:
        error = f"VLM inference failed: {exc}"
        logger.error(error)
        return "", error


# ---------------------------------------------------------------------------
# Capability implementations
# ---------------------------------------------------------------------------

def _observe_with_yolo(
    target_object: str = "",
    confidence_threshold: float = 0.25,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Capture a frame and run YOLO object detection.

    Returns bounding boxes and normalised pixel-space positions for all
    detected objects matching target_object.  The annotated frame (with
    boxes drawn) is returned as frame_b64 so the UI renders it.

    Use this capability when you need to LOCATE or PICK objects.
    """
    frame_rgb, frame_b64, cap_err = _capture_frame()
    if frame_rgb is None:
        return _camera_unavailable_result("observe_with_yolo", {"detected": [], "count": 0})

    camera_info = type(_camera).__name__
    detections, annotated_b64, yolo_err = _run_yolo_detection(
        frame_rgb, target_object, confidence_threshold
    )

    # Prefer annotated frame; fall back to raw frame
    display_b64 = annotated_b64 if annotated_b64 else frame_b64

    return {
        "capability": "observe_with_yolo",
        "target": target_object or "(all objects)",
        "detected": detections,
        "count": len(detections),
        "frame_b64": display_b64,
        "camera_info": camera_info,
        "error": yolo_err or cap_err,
        "note": (
            "Bounding boxes drawn on frame. x/y are normalised image coords — "
            "wire RealSense depth stream for metric 3-D positions. "
            "Swap YOLO_MODEL= in .env for your colleague's custom weights."
        ),
    }


def _observe_with_vlm(
    question: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Capture a frame and ask Gemini Vision to describe the scene.

    Returns a natural-language answer.  Use this for general inquiries
    ("what do I see?") rather than precise object localisation.
    """
    frame_rgb, frame_b64, cap_err = _capture_frame()
    if frame_rgb is None:
        return _camera_unavailable_result("observe_with_vlm", {"description": ""})

    camera_info = type(_camera).__name__
    vlm_frame_b64 = _frame_to_base64_vlm(frame_rgb)
    description, vlm_err = _run_vlm_description(vlm_frame_b64, question)

    return {
        "capability": "observe_with_vlm",
        "question": question,
        "description": description,
        "frame_b64": frame_b64,     # raw frame — no boxes for VLM path
        "camera_info": camera_info,
        "error": vlm_err or cap_err,
        "note": (
            f"VLM used: {os.getenv('VLM_MODEL', 'gemini-2.0-flash-001')}. "
            "For object localisation use observe_with_yolo instead."
        ),
    }


def _run_pick(screw_id: str, x: float, y: float, z: float = 0.0, **kwargs: Any) -> dict[str, Any]:
    """Stub: execute the fine-tuned VLA pick policy for one object."""
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
        "confidence": round(
            random.uniform(0.70, 0.99) if grasp_confirmed else random.uniform(0.10, 0.45), 2
        ),
        "note": "STUB — real call will run post-pick wrist-camera classifier",
    }


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
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Capability] = {
    "observe_with_yolo": Capability(
        id="observe_with_yolo",
        description=(
            "Capture a live camera frame and run YOLO open-vocabulary object detection. "
            "Returns bounding boxes and pixel positions for detected objects. "
            "Use this when you need to LOCATE or PICK specific objects."
        ),
        doc="""
Capability: observe_with_yolo
------------------------------
Purpose:
  Capture the current camera frame and run YOLO inference to locate objects.
  Returns bounding boxes with normalised pixel-space positions.
  The annotated frame (boxes drawn) is returned as frame_b64 for the UI.

  Swap the model weights via YOLO_MODEL= in .env:
    YOLO_MODEL=yoloe-26s-seg.pt    # default, general-purpose
    YOLO_MODEL=/path/to/custom.pt  # your colleague's fine-tuned weights

Required args:  (none)
Optional args:
  target_object        (str,   default "")    — object label to detect, e.g. "screw".
                                                Leave empty to detect all classes.
                                                For YOLO-World, comma-separate multiple labels.
  confidence_threshold (float, default 0.25)  — minimum detection confidence [0, 1].

Returns:
  detected:    list of { id, label, confidence, bbox_px, x, y, z }
               x/y are normalised image coords until depth is wired in.
  count:       number of detections
  frame_b64:   annotated PNG with bounding boxes (displayed in UI)
  camera_info: camera backend used
  error:       non-empty if something went wrong

When to use:
  - "find the screws", "locate the nut", "how many objects are on the table"
  - Any task that will lead to run_pick

Workflow:  START → observe_with_yolo → run_pick → run_visual_qa → done
""",
        fn=_observe_with_yolo,
        optional_args=["target_object", "confidence_threshold"],
    ),

    "observe_with_vlm": Capability(
        id="observe_with_vlm",
        description=(
            "Capture a live camera frame and send it to Gemini Vision for a natural-language "
            "scene description. Use this for general 'what do I see?' inquiries, "
            "NOT for precise object localisation."
        ),
        doc="""
Capability: observe_with_vlm
-----------------------------
Purpose:
  Capture the current camera frame and ask a vision-language model (Gemini)
  to describe or answer a question about the scene in natural language.

  Model is controlled by VLM_MODEL= in .env (default: gemini-2.0-flash-001).
  Uses the same Vertex AI credentials as the rest of the system.

Required args:  (none)
Optional args:
  question (str, default "") — specific question to ask about the scene,
                               e.g. "Are there any loose screws visible?"
                               Leave empty for a full scene description.

Returns:
  description: natural-language answer from the VLM
  frame_b64:   raw camera frame (no bounding boxes)
  camera_info: camera backend used
  error:       non-empty if something went wrong

When to use:
  - "What do you see?", "Describe the workspace"
  - Qualitative questions: "Is the bin full?", "Is the arm clear of obstacles?"
  - Do NOT use for pick tasks — use observe_with_yolo for those.

Workflow:  START → observe_with_vlm → respond to user
""",
        fn=_observe_with_vlm,
        optional_args=["question"],
    ),

    "run_pick": Capability(
        id="run_pick",
        description="Execute the fine-tuned VLA pick policy to grasp one object. (STUB)",
        doc="""
Capability: run_pick
---------------------
Purpose:
  Run the fine-tuned VLA (ACT) policy to pick the specified object.
  Must be preceded by observe_with_yolo to obtain real positions.

Required args:
  screw_id (str)           — the "id" field from observe_with_yolo output
  x, y     (float)         — normalised image coords from observe_with_yolo
  z        (float, opt)    — depth in metres (0.0 until depth is wired in)

Returns:
  status:         "SUCCESS" or "FAILURE"
  failure_reason: string if FAILURE, else null

After this call you MUST call run_visual_qa to verify the grasp.

NOTE: Currently a stub. Real implementation will run the LeRobot ACT policy loop.
""",
        fn=_run_pick,
        required_args=["screw_id", "x", "y"],
        optional_args=["z"],
    ),

    "run_visual_qa": Capability(
        id="run_visual_qa",
        description="Post-pick wrist-camera check: verify the object was successfully grasped. (STUB)",
        doc="""
Capability: run_visual_qa
--------------------------
Purpose:
  Capture a wrist-camera frame and determine whether the gripper holds the object.

Required args:
  screw_id (str) — the object ID that was just picked

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
# Agent-facing interface  (unchanged — agent calls these via tool wrappers)
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
# LangChain tool wrappers  (unchanged)
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
        capability_id: ID from list_capabilities (e.g. "observe_with_yolo")
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