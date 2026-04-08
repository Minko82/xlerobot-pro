"""Robot capability registry.

list_capabilities  → enumerate what the agent can do
read_capability    → get the full contract/doc for one capability
run_capability     → execute it

Each capability is a dataclass describing:
  - id: stable snake_case identifier
  - description: one-line summary shown in list_capabilities
  - doc: full contract the agent reads before calling
  - fn: the actual implementation (stub → real hardware later)

To add a new robot action, define a Capability and register it in REGISTRY.
The agent discovers capabilities at runtime — nothing is hardcoded in the prompt.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict
from langchain_core.tools import tool


@dataclass
class Capability:
    id: str
    description: str
    doc: str
    fn: Callable[..., dict[str, Any]]
    required_args: list[str] = field(default_factory=list)
    optional_args: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stub implementations — replace with real lerobot calls later
# ---------------------------------------------------------------------------

def _get_observation(target_object: str = "", **kwargs: Any) -> dict[str, Any]:
    """Stub: returns a fake scene observation.

    Real implementation will call:
        robot.get_observation()
    and run a vision model to locate objects in 3-D space.
    """
    objects = [
        {"label": "screw", "x": 0.12, "y": -0.05, "z": 0.30, "confidence": 0.91},
        {"label": "screwdriver", "x": 0.20, "y": 0.10, "z": 0.28, "confidence": 0.87},
        {"label": "battery_pack", "x": -0.15, "y": 0.08, "z": 0.25, "confidence": 0.95},
    ]
    if target_object:
        objects = [o for o in objects if target_object.lower() in o["label"]]
    return {
        "capability": "get_observation",
        "scene": objects,
        "frame_id": f"frame_{random.randint(1000, 9999)}",
        "note": "STUB — real call will invoke robot.get_observation() + vision model",
    }


def _send_action(
    action_type: str = "move",
    target_x: float = 0.0,
    target_y: float = 0.0,
    target_z: float = 0.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Stub: simulates sending a joint/EE command to the arm.

    Real implementation will call:
        robot.send_action(action_dict)
    """
    return {
        "capability": "send_action",
        "action_type": action_type,
        "target": {"x": target_x, "y": target_y, "z": target_z},
        "status": "STUB_OK",
        "note": "STUB — real call will invoke robot.send_action()",
    }


def _plan_grasp(object_label: str = "", x: float = 0.0, y: float = 0.0, z: float = 0.0, **kwargs: Any) -> dict[str, Any]:
    """Derive a grasp pose from a detected object position."""
    return {
        "capability": "plan_grasp",
        "object_label": object_label,
        "approach_pose": {"x": x, "y": y, "z": z + 0.10},
        "grasp_pose": {"x": x, "y": y, "z": z},
        "status": "STUB_OK",
        "note": "STUB — real call will run grasp planner (e.g. AnyGrasp or heuristic)",
    }


def _open_gripper(**kwargs: Any) -> dict[str, Any]:
    return {"capability": "open_gripper", "status": "STUB_OK"}


def _close_gripper(force_limit: float = 20.0, **kwargs: Any) -> dict[str, Any]:
    return {"capability": "close_gripper", "force_limit": force_limit, "status": "STUB_OK"}


def _move_to_home(**kwargs: Any) -> dict[str, Any]:
    return {"capability": "move_to_home", "status": "STUB_OK"}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Capability] = {
    "get_observation": Capability(
        id="get_observation",
        description="Capture a camera frame and locate objects in 3-D space.",
        doc="""
Capability: get_observation
----------------------------
Purpose:
  Capture the current camera frame from the robot's wrist/head camera,
  run object detection, and return a list of detected objects with their
  3-D positions relative to the robot base frame.

Required args:  (none)
Optional args:
  target_object (str) — if provided, filter results to only this label.

Returns:
  scene: list of {label, x, y, z, confidence}
  frame_id: unique identifier for the captured frame

When to use:
  - Before any manipulation task to locate the target object.
  - After a grasp attempt to verify success.
  - Whenever spatial information about the scene is needed.

Do NOT use this capability repeatedly without acting on the results.
""",
        fn=_get_observation,
        optional_args=["target_object"],
    ),
    "send_action": Capability(
        id="send_action",
        description="Send a Cartesian or joint command to the robot arm.",
        doc="""
Capability: send_action
------------------------
Purpose:
  Move the robot arm end-effector to the specified Cartesian position.
  The low-level controller handles joint-space conversion.

Required args:
  action_type (str) — one of: "move", "pick", "place", "push"
  target_x, target_y, target_z (float) — goal position in meters,
    robot-base frame.

Optional args:  (none)

Returns:
  status: "STUB_OK" (stub) or "OK" / "ERROR" (real hardware)

When to use:
  - After plan_grasp to move the arm to the approach/grasp pose.
  - For any arm movement in the workspace.

Safety:
  Always call get_observation first to confirm target position.
  Never send blind actions to unknown positions.
""",
        fn=_send_action,
        required_args=["action_type", "target_x", "target_y", "target_z"],
    ),
    "plan_grasp": Capability(
        id="plan_grasp",
        description="Compute a safe grasp pose for a detected object.",
        doc="""
Capability: plan_grasp
-----------------------
Purpose:
  Given a detected object's 3-D position, compute a feasible approach
  and grasp pose for the robot arm.

Required args:
  object_label (str) — label of the object to grasp (e.g. "screw")
  x, y, z (float)    — object position from get_observation

Optional args:  (none)

Returns:
  approach_pose: {x, y, z}  — pre-grasp hover position
  grasp_pose:   {x, y, z}  — final grasp position

When to use:
  After get_observation confirms object location and before send_action.
  Standard pipeline: get_observation → plan_grasp → send_action(move to approach)
    → send_action(move to grasp) → close_gripper.
""",
        fn=_plan_grasp,
        required_args=["object_label", "x", "y", "z"],
    ),
    "open_gripper": Capability(
        id="open_gripper",
        description="Fully open the robot gripper.",
        doc="""
Capability: open_gripper
-------------------------
Required args:  (none)
When to use:  Before approaching an object to grasp.
""",
        fn=_open_gripper,
    ),
    "close_gripper": Capability(
        id="close_gripper",
        description="Close the gripper to grasp the object.",
        doc="""
Capability: close_gripper
--------------------------
Required args:  (none)
Optional args:
  force_limit (float, default 20.0) — max closing force in Newtons.
When to use:  Once the arm is at the grasp pose.
""",
        fn=_close_gripper,
        optional_args=["force_limit"],
    ),
    "move_to_home": Capability(
        id="move_to_home",
        description="Return the arm to the safe home position.",
        doc="""
Capability: move_to_home
-------------------------
Required args:  (none)
When to use:
  - At the start of a task sequence to establish a known state.
  - After completing a task to park the arm safely.
""",
        fn=_move_to_home,
    ),
}


# ---------------------------------------------------------------------------
# Agent-facing interface
# ---------------------------------------------------------------------------

def list_capabilities() -> dict[str, Any]:
    """Return a compact index of all available capabilities."""
    return {
        "capabilities": [
            {"id": cap.id, "description": cap.description}
            for cap in REGISTRY.values()
        ]
    }

def read_capability(capability_id: str) -> dict[str, Any]:
    """Return the full contract/doc for one capability."""
    cap = REGISTRY.get(capability_id)
    if cap is None:
        return {"error": f"Unknown capability: {capability_id!r}. Call list_capabilities first."}
    return {
        "id": cap.id,
        "description": cap.description,
        "doc": cap.doc.strip(),
        "required_args": cap.required_args,
        "optional_args": cap.optional_args,
    }

def run_capability(capability_id: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute a capability by id with the given args dict."""
    cap = REGISTRY.get(capability_id)
    if cap is None:
        return {"error": f"Unknown capability: {capability_id!r}. Call list_capabilities first."}
    args = args or {}
    missing = [k for k in cap.required_args if k not in args]
    if missing:
        return {
            "error": f"Missing required args for {capability_id!r}: {missing}",
            "hint": f"Read the full contract with read_capability('{capability_id}')",
        }
    try:
        return cap.fn(**args)
    except Exception as exc:
        return {"error": str(exc), "capability_id": capability_id}
    
@tool
def list_capabilities_tool(**kwargs) -> dict[str, Any]:
    """List all available robot capabilities."""
    return list_capabilities()


@tool
def read_capability_tool(capability_id: str, **kwargs) -> dict[str, Any]:
    """Read full documentation for a capability before using it."""
    return read_capability(capability_id)


@tool
def run_capability_tool(
    capability_id: str, 
    args: dict[str, Any] | None = None, 
    params: dict[str, Any] | None = None,
    **kwargs
) -> dict[str, Any]:
    """Execute a capability with validated arguments.
    
    Args:
        capability_id: The capability ID from list_capabilities (e.g. "plan_grasp")
        args: Dict of capability-specific arguments, e.g. {"object_label": "screw", "x": 0.12, "y": -0.05, "z": 0.30}

    This endpoint accepts provider/tool-call variants, including:
    - nested `args`, `kwargs`, or `params`
    - flattened top-level capability fields
    """

    # Providers can vary in tool argument shape (args / kwargs / params / flat).
    # Normalize by recursively unwrapping known wrapper keys into one flat dict.
    wrapper_keys = {"args", "kwargs", "params", "parameters", "payload", "input"}
    merged: dict[str, Any] = {}
    queue: list[dict[str, Any]] = []

    if isinstance(args, dict):
        queue.append(dict(args))
    if isinstance(params, dict):
        queue.append(dict(params))
    if kwargs:
        queue.append(dict(kwargs))

    while queue:
        current = queue.pop(0)
        for key, value in current.items():
            if key in wrapper_keys and isinstance(value, dict):
                queue.append(dict(value))
                continue
            merged[key] = value

    # Cleanup common framework-internal keys when present.
    merged.pop("v__args", None)
    merged.pop("type", None)

    return run_capability(capability_id, merged or None)


TOOLS = [
    list_capabilities_tool,
    read_capability_tool,
    run_capability_tool,
]
