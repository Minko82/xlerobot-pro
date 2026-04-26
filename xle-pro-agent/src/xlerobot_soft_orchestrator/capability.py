"""Robot capability registry.

Capabilities exposed to the agent:

  observe_with_base_camera   — capture base frame, call VLM with a question.
  observe_with_wrist_camera  — capture wrist frame, call VLM with a question.
  observe_with_both_cameras  — capture both frames, call VLM with a question.

  yolo_base_camera           — run YOLO on the base camera frame.
  yolo_wrist_camera          — run YOLO on the wrist camera frame.

  start_vla_policy           — launch the async inference client from the policy
                               registry (policies.yaml).  Requires only policy_id.
  stop_vla_policy            — terminate the running inference client.

  wait                       — pause agent execution for N seconds.

Camera backend:
  Both cameras are served by the lerobot ZMQ camera-server.
  Configure via .env:
    ZMQ_SERVER_HOST  (default: localhost)
    ZMQ_SERVER_PORT  (default: 5555)

VLA inference:
  Configure via .env:
    VLA_INFERENCE_SERVER  (default: 192.168.50.42:8080)
    ROBOT_PORT            (default: /dev/ttyACM0)
    ROBOT_ID              (default: left_arm)

YOLO backend:
  Uses arpa_vision YOLO_WORLD with EV battery weights by default.
  Override weight path via YOLO_MODEL= in .env.

VLM backend:
  Uses Gemini via Vertex AI.
  VERTEX_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS must be set.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import subprocess
import threading
import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from langchain_core.tools import tool
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy Registry
# ---------------------------------------------------------------------------

def _load_policies() -> dict[str, dict[str, str]]:
    policy_path = Path(__file__).parent / "policies.yaml"
    if not policy_path.exists():
        logger.warning("policies.yaml not found at %s. VLA policies unavailable.", policy_path)
        return {}
    with open(policy_path, "r") as f:
        return yaml.safe_load(f) or {}

POLICIES: dict[str, dict[str, str]] = _load_policies()


# ---------------------------------------------------------------------------
# ZMQ Camera — dual lazy singletons
# ---------------------------------------------------------------------------

_camera_base = None
_camera_wrist = None
_camera_base_error: str = ""
_camera_wrist_error: str = ""

ZMQ_HOST: str = os.getenv("ZMQ_SERVER_HOST", "localhost")
ZMQ_PORT: int = int(os.getenv("ZMQ_SERVER_PORT", "5555"))


def _init_zmq_camera(camera_name: str) -> tuple[Any, str]:
    try:
        from lerobot.cameras.zmq.camera_zmq import ZMQCamera
        from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig

        config = ZMQCameraConfig(
            server_address=ZMQ_HOST,
            port=ZMQ_PORT,
            camera_name=camera_name,
            timeout_ms=1000,
            warmup_s=1,
        )
        cam = ZMQCamera(config)
        cam.connect(warmup=True)
        logger.info("ZMQ camera '%s' connected at %s:%s.", camera_name, ZMQ_HOST, ZMQ_PORT)
        return cam, ""
    except Exception as exc:
        err = f"ZMQ camera '{camera_name}' init failed: {exc}"
        logger.warning(err)
        return None, err


def _get_frame(camera_name: str) -> tuple[np.ndarray | None, str]:
    """Return (frame_rgb, error) for the named camera ('base' or 'wrist')."""
    global _camera_base, _camera_base_error, _camera_wrist, _camera_wrist_error

    if camera_name == "base":
        if _camera_base is None:
            _camera_base, _camera_base_error = _init_zmq_camera("base")
        if _camera_base is not None:
            try:
                return _camera_base.read(), ""
            except Exception as exc:
                return None, f"Base camera read failed: {exc}"
        return None, _camera_base_error

    if camera_name == "wrist":
        if _camera_wrist is None:
            _camera_wrist, _camera_wrist_error = _init_zmq_camera("wrist")
        if _camera_wrist is not None:
            try:
                return _camera_wrist.read(), ""
            except Exception as exc:
                return None, f"Wrist camera read failed: {exc}"
        return None, _camera_wrist_error

    return None, f"Unknown camera name: '{camera_name}'. Use 'base' or 'wrist'."


# ---------------------------------------------------------------------------
# Frame encoding helpers
# ---------------------------------------------------------------------------

def _frame_to_base64(frame_rgb: np.ndarray) -> str:
    """Encode (H, W, 3) uint8 RGB array as a base64 PNG string (for UI display)."""
    img = PILImage.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _frame_to_base64_vlm(frame_rgb: np.ndarray, max_dim: int = 400, jpeg_quality: int = 85) -> str:
    """Resize + JPEG-encode a frame for VLM submission (minimises token cost)."""
    h, w = frame_rgb.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = PILImage.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
        img = img.resize((new_w, new_h), PILImage.LANCZOS)
    else:
        img = PILImage.fromarray(frame_rgb.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# VLA Process Management
# ---------------------------------------------------------------------------

_vla_process: subprocess.Popen | None = None

_VLA_SERVER         = os.getenv("VLA_INFERENCE_SERVER", "192.168.50.42:8080")
_ROBOT_PORT         = os.getenv("ROBOT_PORT", "/dev/ttyACM0")
_ROBOT_ID           = os.getenv("ROBOT_ID", "left_arm")
_VLA_START_TIMEOUT  = int(os.getenv("VLA_START_TIMEOUT", "300"))  # seconds

# Fires only after client.start() completes (model loaded on server) and both
# the action-receiver thread AND the control-loop thread have crossed their
# threading.Barrier — meaning the robot is about to send its first observation.
_VLA_READY_SIGNAL = "Action receiving thread starting"


_vla_log_path: Path | None = None   # path to the current policy's log file


def _watch_log_file(
    log_path: Path,
    process: subprocess.Popen,
    ready_event: threading.Event,
    signal_seen: list[bool],
    output_buf: list[str],
) -> None:
    """Tail the log file until the ready signal appears or the process exits.

    Writing subprocess output to a file (instead of a PIPE) is critical: a PIPE
    has a 64 KB kernel buffer and blocks the subprocess write() syscall when full.
    If the orchestrator's log handler is slow (e.g. lock contention with the agent),
    the buffer fills up and the robot client freezes — no more observations sent,
    no more action chunks generated.  A file never blocks the writer.
    """
    with open(log_path, "r", errors="replace") as f:
        while True:
            line = f.readline()
            if not line:
                if process.poll() is not None:
                    # Process exited without emitting the ready signal
                    ready_event.set()
                    return
                time.sleep(0.05)
                continue
            line = line.rstrip()
            if line:
                logger.info("[robot_client] %s", line)
                output_buf.append(line)
            if _VLA_READY_SIGNAL in line:
                signal_seen[0] = True
                ready_event.set()
                return   # stop watching; subprocess continues writing to the file on its own


def _start_vla_policy(policy_id: str, **kwargs: Any) -> dict[str, Any]:
    global _vla_process, _vla_log_path

    if _vla_process is not None and _vla_process.poll() is None:
        return {
            "status": "FAILURE",
            "error": "A VLA policy is already running. Call stop_vla_policy first.",
            "pid": _vla_process.pid,
        }

    policy_cfg = POLICIES.get(policy_id)
    if not policy_cfg:
        return {
            "status": "FAILURE",
            "error": (
                f"Policy '{policy_id}' not found in policies.yaml. "
                f"Available: {list(POLICIES.keys())}"
            ),
        }

    camera_config = json.dumps({
        "base": {
            "type": "zmq", "server_address": ZMQ_HOST, "port": ZMQ_PORT,
            "camera_name": "base", "width": 640, "height": 480, "fps": 30,
        },
        "wrist": {
            "type": "zmq", "server_address": ZMQ_HOST, "port": ZMQ_PORT,
            "camera_name": "wrist", "width": 640, "height": 480, "fps": 30,
        },
    })

    cmd = [
        "python", "-u", "-m", "lerobot.async_inference.robot_client",
        "--robot.type=so101_follower",
        f"--robot.port={_ROBOT_PORT}",
        f"--robot.id={_ROBOT_ID}",
        f"--robot.cameras={camera_config}",
        f"--task={policy_cfg['task']}",
        f"--server_address={_VLA_SERVER}",
        f"--policy_type={policy_cfg['policy_type']}",
        f"--pretrained_name_or_path={policy_cfg['repo_id']}",
        "--policy_device=cuda",
        "--client_device=cpu",
        "--actions_per_chunk=50",
    ]

    # Write output to a file, NOT a PIPE.  A PIPE has a finite kernel buffer
    # (~64 KB on Linux); if the orchestrator is slow reading it, the subprocess
    # blocks inside write() and the entire robot control loop freezes.
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    _vla_log_path = log_dir / f"vla_client_{int(time.time())}.log"
    log_file = open(_vla_log_path, "wb")

    try:
        _vla_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=log_file,
            stderr=log_file,   # both streams to the same file
        )
        log_file.close()  # orchestrator only needs the read side
        # Robot's _follower.py calls input() on startup to confirm calibration.
        # Send ENTER to auto-accept; without this the process hangs forever.
        _vla_process.stdin.write(b"\n")
        _vla_process.stdin.flush()
    except Exception as exc:
        log_file.close()
        return {"status": "FAILURE", "error": str(exc)}

    logger.info(
        "Policy '%s' launching — logs: %s (tail -f %s)",
        policy_id, _vla_log_path, _vla_log_path,
    )

    ready_event = threading.Event()
    signal_seen = [False]
    output_buf: list[str] = []

    threading.Thread(
        target=_watch_log_file,
        args=(_vla_log_path, _vla_process, ready_event, signal_seen, output_buf),
        daemon=True,
    ).start()

    logger.info(
        "Waiting for policy '%s' to load (timeout: %ds)...",
        policy_id, _VLA_START_TIMEOUT,
    )
    ready_event.wait(timeout=_VLA_START_TIMEOUT)

    def _tail(n: int = 30) -> str:
        lines = output_buf[-n:] if output_buf else []
        return "\n".join(lines) if lines else "(no output captured)"

    if not signal_seen[0]:
        rc = _vla_process.poll()
        if rc is not None:
            err = (
                f"Policy '{policy_id}' process exited with code {rc} "
                "before inference started."
            )
        else:
            _vla_process.terminate()
            try:
                _vla_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _vla_process.kill()
                _vla_process.wait()
            err = (
                f"Policy '{policy_id}' did not start within {_VLA_START_TIMEOUT}s. "
                "Model download may have stalled or the gRPC server is unreachable."
            )
        _vla_process = None
        return {
            "status": "FAILURE",
            "error": err,
            "output_tail": _tail(),
            "log_file": str(_vla_log_path),
        }

    if _vla_process.poll() is not None:
        _vla_process = None
        return {
            "status": "FAILURE",
            "error": f"Policy '{policy_id}' exited immediately after starting inference.",
            "output_tail": _tail(),
            "log_file": str(_vla_log_path),
        }

    return {
        "status": "SUCCESS",
        "message": (
            f"Policy '{policy_id}' is running — model loaded, action chunks generating "
            f"(task: \"{policy_cfg['task']}\", type: {policy_cfg['policy_type']})."
        ),
        "pid": _vla_process.pid,
        "log_file": str(_vla_log_path),
    }


def _stop_vla_policy(**kwargs: Any) -> dict[str, Any]:
    global _vla_process
    if _vla_process is None or _vla_process.poll() is not None:
        _vla_process = None
        return {"status": "WARNING", "message": "No VLA policy was running."}

    _vla_process.terminate()
    try:
        _vla_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _vla_process.kill()
        _vla_process.wait()
    _vla_process = None
    return {"status": "SUCCESS", "message": "VLA policy stopped."}


def _wait(seconds: int, **kwargs: Any) -> dict[str, Any]:
    time.sleep(int(seconds))
    return {"status": "SUCCESS", "message": f"Waited {seconds} seconds."}


# ---------------------------------------------------------------------------
# YOLO — lazy singleton (arpa_vision YOLO_WORLD)
# ---------------------------------------------------------------------------

_ARPA_VISION_ROOT    = "/home/xle/arpa_vision"
_ARPA_VISION_WEIGHTS = "/home/xle/arpa_vision/arpa_vision/scripts/yolov8x-worldv2_best.pt"
_EV_BATTERY_QUERIES  = ["Bolt", "BusBar", "InteriorScrew", "Nut", "OrangeCover", "Screw", "Screw Hole"]

_yolo_model = None
_yolo_error: str = ""


def _get_yolo_model():
    global _yolo_model, _yolo_error
    if _yolo_model is not None:
        return _yolo_model, ""
    if _yolo_error:
        return None, _yolo_error
    try:
        import sys
        import torch
        if _ARPA_VISION_ROOT not in sys.path:
            sys.path.insert(0, _ARPA_VISION_ROOT)
        from arpa_vision.scripts.BoundingBoxDetectors import YOLO_WORLD

        weight_path = os.getenv("YOLO_MODEL", _ARPA_VISION_WEIGHTS)

        _yolo_model = YOLO_WORLD(weight_file_path=weight_path)
        logger.info("YOLO_WORLD loaded on CPU: %s", weight_path)

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


def _run_yolo_detection(
    frame_rgb: np.ndarray,
    target_object: str,
    confidence_threshold: float = 0.20,
) -> tuple[list[dict[str, Any]], str, str]:
    model, err = _get_yolo_model()
    if model is None:
        return [], "", err
    try:
        queries = _build_queries(target_object)
        candidates_2d = model.predict(frame_rgb, queries=queries, debug=False)

        detections: list[dict[str, Any]] = []
        h, w = frame_rgb.shape[:2]
        annotated = frame_rgb.copy()
        det_idx = 0

        for query, preds in candidates_2d.items():
            for box, prob in zip(preds["boxes"], preds["probs"]):
                if prob < confidence_threshold:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                label_id = query.lower().replace(" ", "_")
                detections.append({
                    "id": f"{label_id}_{det_idx}",
                    "label": query,
                    "confidence": round(prob, 3),
                    "bbox_px": [x1, y1, x2, y2],
                    "x": round(cx / w - 0.5, 4),
                    "y": round(cy / h - 0.5, 4),
                    "z": None,
                    "note": "x/y are normalised image coords; wire depth for metric 3-D.",
                })
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated, f"{query} {prob:.2f}", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )
                det_idx += 1

        return detections, _frame_to_base64(annotated), ""
    except Exception as exc:
        err = f"YOLO inference failed: {exc}"
        logger.error(err)
        return [], "", err


# ---------------------------------------------------------------------------
# VLM (Gemini Vision) helper
# ---------------------------------------------------------------------------

def _run_vlm_description(frame_b64: str, question: str) -> tuple[str, str]:
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part

        project  = os.getenv("VERTEX_PROJECT_ID", "")
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        vlm_model_name = os.getenv("VLM_MODEL", "gemini-2.5-flash-lite")

        if not project:
            return "", "VERTEX_PROJECT_ID not set in .env"

        vertexai.init(project=project, location=location)
        model = GenerativeModel(vlm_model_name)

        image_bytes = base64.b64decode(frame_b64)
        image_part = Part.from_data(data=image_bytes, mime_type="image/jpeg")
        text_part = question or (
            "Describe this robot workspace image in detail. "
            "List all visible objects, their approximate positions relative to each other, "
            "and anything relevant to a pick-and-place robot task."
        )

        response = model.generate_content([image_part, text_part])
        return response.text.strip(), ""
    except ImportError:
        err = "vertexai SDK not installed. Run: pip install google-cloud-aiplatform"
        logger.error(err)
        return "", err
    except Exception as exc:
        err = f"VLM inference failed: {exc}"
        logger.error(err)
        return "", err


# ---------------------------------------------------------------------------
# Capability implementations
# ---------------------------------------------------------------------------

def _yolo_base_camera(
    target_object: str = "",
    confidence_threshold: float = 0.25,
    **kwargs: Any,
) -> dict[str, Any]:
    frame, err = _get_frame("base")
    if frame is None:
        return {"capability": "yolo_base_camera", "detected": [], "count": 0,
                "frame_b64": "", "error": err}
    detections, annotated_b64, yolo_err = _run_yolo_detection(frame, target_object, confidence_threshold)
    return {
        "capability": "yolo_base_camera",
        "camera": "base",
        "target": target_object or "(all EV battery classes)",
        "detected": detections,
        "count": len(detections),
        "frame_b64": annotated_b64 or _frame_to_base64(frame),
        "error": yolo_err or err,
    }


def _yolo_wrist_camera(
    target_object: str = "",
    confidence_threshold: float = 0.25,
    **kwargs: Any,
) -> dict[str, Any]:
    frame, err = _get_frame("wrist")
    if frame is None:
        return {"capability": "yolo_wrist_camera", "detected": [], "count": 0,
                "frame_b64": "", "error": err}
    detections, annotated_b64, yolo_err = _run_yolo_detection(frame, target_object, confidence_threshold)
    return {
        "capability": "yolo_wrist_camera",
        "camera": "wrist",
        "target": target_object or "(all EV battery classes)",
        "detected": detections,
        "count": len(detections),
        "frame_b64": annotated_b64 or _frame_to_base64(frame),
        "error": yolo_err or err,
    }


# ---------------------------------------------------------------------------
# VLM context preambles — prepended to every question before it reaches the model.
# These give the VLM the world knowledge it cannot infer from the image alone.
# ---------------------------------------------------------------------------

_VLM_PREAMBLE_BASE = (
    "CONTEXT: You are analysing an image from the fixed base (overview) camera "
    "of a robot manipulation workstation. The workspace contains EV battery components: "
    "screws, nuts, bolts, orange plastic covers, bus bars, and small containers/trays "
    "used to collect picked parts. The robot arm may appear at the edge of the frame. "
    "Answer concisely and literally — if you can see the object, say so; "
    "if you cannot, say so.\n\nQUESTION: "
)

_VLM_PREAMBLE_WRIST = (
    "CONTEXT: You are analysing an image from the wrist-mounted camera on a robot arm. "
    "The black ribbed cylindrical structures visible are the robot's parallel-jaw gripper "
    "fingers (end-effector). A successfully grasped object will appear between or pressed "
    "against those gripper fingers. The workspace contains EV battery components: screws, "
    "nuts, bolts, and small plastic containers. "
    "Answer concisely and literally — if you can see the object, say so; "
    "if you cannot, say so.\n\nQUESTION: "
)


def _observe_with_base_camera(question: str = "", **kwargs: Any) -> dict[str, Any]:
    frame, err = _get_frame("base")
    if frame is None:
        return {"capability": "observe_with_base_camera", "question": question,
                "description": "", "frame_b64": "", "error": err}
    vlm_b64 = _frame_to_base64_vlm(frame)
    enriched = _VLM_PREAMBLE_BASE + (question or "Describe the full scene.")
    description, vlm_err = _run_vlm_description(vlm_b64, enriched)
    return {
        "capability": "observe_with_base_camera",
        "camera": "base",
        "question": question,
        "description": description,
        "frame_b64": _frame_to_base64(frame),  # UI only — stripped from agent context
        "error": vlm_err or err,
    }


def _observe_with_wrist_camera(question: str = "", **kwargs: Any) -> dict[str, Any]:
    frame, err = _get_frame("wrist")
    if frame is None:
        return {"capability": "observe_with_wrist_camera", "question": question,
                "description": "", "frame_b64": "", "error": err}
    vlm_b64 = _frame_to_base64_vlm(frame)
    enriched = _VLM_PREAMBLE_WRIST + (question or "Describe what is in or near the gripper.")
    description, vlm_err = _run_vlm_description(vlm_b64, enriched)
    return {
        "capability": "observe_with_wrist_camera",
        "camera": "wrist",
        "question": question,
        "description": description,
        "frame_b64": _frame_to_base64(frame),  # UI only — stripped from agent context
        "error": vlm_err or err,
    }


def _observe_with_both_cameras(question: str = "", **kwargs: Any) -> dict[str, Any]:
    base_frame, base_err = _get_frame("base")
    wrist_frame, wrist_err = _get_frame("wrist")

    descriptions: list[str] = []
    combined_err = " | ".join(filter(None, [base_err, wrist_err]))

    if base_frame is not None:
        enriched = _VLM_PREAMBLE_BASE + (question or "Describe the full scene.")
        desc, vlm_err = _run_vlm_description(_frame_to_base64_vlm(base_frame), enriched)
        if vlm_err:
            combined_err = " | ".join(filter(None, [combined_err, f"Base VLM: {vlm_err}"]))
        else:
            descriptions.append(f"Base camera: {desc}")

    if wrist_frame is not None:
        enriched = _VLM_PREAMBLE_WRIST + (question or "Describe what is in or near the gripper.")
        desc, vlm_err = _run_vlm_description(_frame_to_base64_vlm(wrist_frame), enriched)
        if vlm_err:
            combined_err = " | ".join(filter(None, [combined_err, f"Wrist VLM: {vlm_err}"]))
        else:
            descriptions.append(f"Wrist camera: {desc}")

    return {
        "capability": "observe_with_both_cameras",
        "question": question,
        "description": "\n\n".join(descriptions),
        # Both keys are stripped from agent context — UI only
        "base_frame_b64":  _frame_to_base64(base_frame)  if base_frame  is not None else "",
        "wrist_frame_b64": _frame_to_base64(wrist_frame) if wrist_frame is not None else "",
        "error": combined_err,
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
    "observe_with_base_camera": Capability(
        id="observe_with_base_camera",
        description=(
            "Capture the base (scene-overview) camera frame and ask the VLM a question. "
            "Use for macro workspace understanding: object presence, scene state, obstacle clearance."
        ),
        doc="""
Capability: observe_with_base_camera
--------------------------------------
Purpose:
  Captures the base RealSense/ZMQ camera frame and sends it to Gemini Vision
  along with your question. Returns a natural-language answer.

  The raw frame is returned as frame_b64 for the UI only — it is NOT in the
  agent context window (automatically stripped).

Required args:  (none)
Optional args:
  question (str) — Specific question to guide the VLM response.
                   e.g. "Is a screw visible on the table?"
                   Leave empty for a full scene description.

Returns:
  description: VLM answer
  frame_b64:   raw base camera frame (UI only)
  error:       non-empty if something failed

Use for: "what do you see?", "is the table clear?", "describe the workspace"
Do NOT use for precise object coordinates — use yolo_base_camera instead.
""",
        fn=_observe_with_base_camera,
        optional_args=["question"],
    ),

    "observe_with_wrist_camera": Capability(
        id="observe_with_wrist_camera",
        description=(
            "Capture the wrist (end-effector) camera frame and ask the VLM a question. "
            "Use to inspect grasp quality or confirm what is in the gripper."
        ),
        doc="""
Capability: observe_with_wrist_camera
---------------------------------------
Purpose:
  Captures the wrist ZMQ camera frame and sends it to Gemini Vision.
  Ideal for post-pick grasp verification.

Required args:  (none)
Optional args:
  question (str) — e.g. "Is the screw firmly grasped?", "What is in the gripper?"

Returns:
  description: VLM answer
  frame_b64:   raw wrist camera frame (UI only)
  error:       non-empty if something failed

Use for: grasp confirmation, close-up object inspection.
""",
        fn=_observe_with_wrist_camera,
        optional_args=["question"],
    ),

    "observe_with_both_cameras": Capability(
        id="observe_with_both_cameras",
        description=(
            "Capture both base and wrist frames and ask the VLM a question about both. "
            "Use when full context (scene + end-effector state) is needed."
        ),
        doc="""
Capability: observe_with_both_cameras
---------------------------------------
Purpose:
  Captures both camera frames and runs the VLM twice (once per frame).
  Returns a combined natural-language description.

Required args:  (none)
Optional args:
  question (str) — Applies to both cameras, prefixed with the camera name.

Returns:
  description: combined VLM answer (base then wrist)
  base_frame_b64:  base camera frame (UI only)
  wrist_frame_b64: wrist camera frame (UI only)
  error:       non-empty if something failed

Use for: comprehensive state checks mid-task.
""",
        fn=_observe_with_both_cameras,
        optional_args=["question"],
    ),

    "yolo_base_camera": Capability(
        id="yolo_base_camera",
        description=(
            "Capture the base camera frame and run YOLO open-vocabulary detection. "
            "Returns bounding boxes and pixel positions. Use to LOCATE or COUNT objects."
        ),
        doc="""
Capability: yolo_base_camera
------------------------------
Purpose:
  Captures the base camera frame and runs YOLO_WORLD inference to locate objects.
  Returns bounding boxes with normalised pixel-space positions.

Required args:  (none)
Optional args:
  target_object        (str,   default "")    — object label to detect, e.g. "Screw".
                                                Leave empty to detect all EV battery classes.
                                                Comma-separate for multiple: "Screw,Nut".
  confidence_threshold (float, default 0.25)  — minimum confidence [0, 1].

EV battery classes: Bolt, BusBar, InteriorScrew, Nut, OrangeCover, Screw, Screw Hole

Returns:
  detected:  list of { id, label, confidence, bbox_px, x, y, z }
             x/y are normalised image coords (z is None until depth is wired).
  count:     number of detections
  frame_b64: annotated frame with bounding boxes (UI only)
  error:     non-empty if something failed
""",
        fn=_yolo_base_camera,
        optional_args=["target_object", "confidence_threshold"],
    ),

    "yolo_wrist_camera": Capability(
        id="yolo_wrist_camera",
        description=(
            "Capture the wrist camera frame and run YOLO detection. "
            "Use to verify the grasped object or detect items in close range."
        ),
        doc="""
Capability: yolo_wrist_camera
-------------------------------
Purpose:
  Captures the wrist camera frame and runs YOLO_WORLD inference.
  Useful for close-range object checks or post-pick confirmation.

Required args:  (none)
Optional args:
  target_object        (str,   default "")    — label(s) to detect.
  confidence_threshold (float, default 0.25)  — minimum confidence [0, 1].

Returns: same schema as yolo_base_camera.
""",
        fn=_yolo_wrist_camera,
        optional_args=["target_object", "confidence_threshold"],
    ),

    "start_vla_policy": Capability(
        id="start_vla_policy",
        description=(
            "Launch the async VLA inference client using a predefined policy ID from the registry. "
            f"Available policy IDs: {list(POLICIES.keys())}."
        ),
        doc=(
            "Capability: start_vla_policy\n"
            "------------------------------\n"
            "Purpose:\n"
            "  Looks up the policy_id in policies.yaml and dispatches the lerobot async\n"
            "  inference robot client as a background subprocess.\n\n"
            "  Available policies (from policies.yaml):\n"
            + "\n".join(
                f"    {pid}: task=\"{cfg['task']}\", type={cfg['policy_type']}"
                for pid, cfg in POLICIES.items()
            )
            + "\n\n"
            "Required args:\n"
            "  policy_id (str) — ID from the list above, e.g. 'screw_picking'.\n\n"
            "Returns:\n"
            "  status:  \"SUCCESS\" or \"FAILURE\"\n"
            "  message: confirmation with task description\n"
            "  pid:     OS process ID of the launched client\n\n"
            "IMPORTANT: This call BLOCKS until the model is loaded and action chunks are\n"
            "flowing (or until the 5-minute timeout). On SUCCESS the robot is already moving.\n"
            "Call wait (e.g. 20 seconds) to let the policy execute further, then observe\n"
            "to verify success.  Repeat wait+observe until done, then call stop_vla_policy."
        ),
        fn=_start_vla_policy,
        required_args=["policy_id"],
    ),

    "stop_vla_policy": Capability(
        id="stop_vla_policy",
        description="Terminate the currently running VLA inference client.",
        doc="""
Capability: stop_vla_policy
-----------------------------
Purpose:
  Terminates the background async inference robot client (SIGTERM, then SIGKILL
  if it does not exit within 5 seconds).

Required args:  (none)

Returns:
  status:  "SUCCESS" or "WARNING" (if nothing was running)
  message: confirmation string
""",
        fn=_stop_vla_policy,
    ),

    "wait": Capability(
        id="wait",
        description="Pause agent execution for N seconds to allow physical operations to complete.",
        doc="""
Capability: wait
-----------------
Purpose:
  Block the agent for the given number of seconds.
  Use after start_vla_policy to give the robot time to execute the policy.

Required args:
  seconds (int) — number of seconds to wait, e.g. 20.

Returns:
  status:  "SUCCESS"
  message: confirmation string
""",
        fn=_wait,
        required_args=["seconds"],
    ),
}


# ---------------------------------------------------------------------------
# Agent-facing interface
# ---------------------------------------------------------------------------

def list_capabilities() -> dict[str, Any]:
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
        capability_id: ID from list_capabilities (e.g. "yolo_base_camera")
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
