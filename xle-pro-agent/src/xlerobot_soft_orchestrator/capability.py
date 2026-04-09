"""Robot capability registry.

Three capabilities are exposed to the agent:

  get_observation   — identify screws (and other objects) in the scene
  run_pick          — execute the fine-tuned VLA policy to pick one screw
  run_visual_qa     — post-pick camera check: did we actually grab the screw?

Low-level primitives (open/close gripper, send_action, move_to_home, plan_grasp)
are intentionally NOT exposed to the agent.  They are implementation details
owned by run_pick / run_visual_qa and will be called internally when those
capabilities are wired to real hardware.

To add a new capability later (e.g. navigate_to), define a Capability and
register it in REGISTRY — the agent discovers everything at runtime.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Capability:
    id: str
    description: str                       # one-liner shown in list_capabilities
    doc: str                               # full contract the agent reads before calling
    fn: Callable[..., dict[str, Any]]
    required_args: list[str] = field(default_factory=list)
    optional_args: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stub implementations
# Replace each fn with real hardware/policy calls when ready.
# ---------------------------------------------------------------------------

def _get_observation(target_object: str = "screw", **kwargs: Any) -> dict[str, Any]:
    """Stub: returns a fake scene observation with detected screws.

    Real implementation will:
      1. Capture wrist/overhead camera frame.
      2. Run object detection / depth estimation.
      3. Return detected objects with 3-D positions in robot-base frame.
    """
    # Simulate a shrinking screw population across repeated calls so the
    # agent's loop-until-done logic can be exercised in dry runs.
    all_screws = [
        {"label": "screw", "id": "screw_0", "x": 0.12, "y": -0.05, "z": 0.30, "confidence": 0.93},
        {"label": "screw", "id": "screw_1", "x": 0.22, "y":  0.03, "z": 0.29, "confidence": 0.88},
        {"label": "screw", "id": "screw_2", "x": 0.08, "y":  0.11, "z": 0.31, "confidence": 0.91},
    ]
    objects = [o for o in all_screws if target_object.lower() in o["label"]]
    return {
        "capability": "get_observation",
        "target": target_object,
        "detected": objects,
        "count": len(objects),
        "frame_id": f"frame_{random.randint(1000, 9999)}",
        "note": "STUB — real call will invoke camera capture + object detection model",
    }


def _run_pick(screw_id: str, x: float, y: float, z: float, **kwargs: Any) -> dict[str, Any]:
    """Stub: executes the fine-tuned VLA pick policy for one screw.

    Real implementation will:
      1. Build a structured task prompt (e.g. "pick the screw at <pos>").
      2. Run the ACT/VLA inference loop for up to max_steps control steps.
      3. Each step: get observation → predict action → postprocess → send_action.
      4. Move arm to inspection pose when the horizon is reached or the policy
         signals completion.

    Internally handles: approach, open gripper, descend, close gripper, lift.
    None of those primitives are exposed to the agent.
    """
    success = random.random() > 0.25   # 75 % stub success rate
    return {
        "capability": "run_pick",
        "screw_id": screw_id,
        "target_position": {"x": x, "y": y, "z": z},
        "status": "SUCCESS" if success else "FAILURE",
        "failure_reason": None if success else "policy did not achieve stable grasp",
        "note": "STUB — real call will run VLA policy inference loop (ACT default)",
    }


def _run_visual_qa(screw_id: str, **kwargs: Any) -> dict[str, Any]:
    """Stub: post-pick camera check — did the gripper actually capture the screw?

    Real implementation will:
      1. Capture a wrist-camera frame from the inspection pose.
      2. Run a lightweight classifier / detection model.
      3. Return grasp_confirmed=True/False and a confidence score.

    This is intentionally separate from run_pick so the agent can decide
    whether to retry, skip, or escalate on failure.
    """
    grasp_confirmed = random.random() > 0.2   # 80 % stub confirmation rate
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
        description="Capture a camera frame and locate screws (or other objects) in 3-D space.",
        doc="""
Capability: get_observation
----------------------------
Purpose:
  Capture the current camera frame, run object detection, and return a list
  of detected objects with their 3-D positions in the robot-base frame.

Required args:  (none)
Optional args:
  target_object (str, default "screw") — label filter applied to detections.

Returns:
  detected: list of { id, label, x, y, z, confidence }
  count:    number of detected objects matching the filter
  frame_id: identifier for the captured frame (useful for logging)

When to use:
  - At the start of each pick cycle to find the next screw to pick.
  - After completing all picks to confirm the workspace is clear.
  - Do NOT call repeatedly without acting on the results.

Workflow position:  START → get_observation → run_pick → run_visual_qa → (loop or done)
""",
        fn=_get_observation,
        optional_args=["target_object"],
    ),

    "run_pick": Capability(
        id="run_pick",
        description="Execute the fine-tuned VLA pick policy to grasp one screw.",
        doc="""
Capability: run_pick
---------------------
Purpose:
  Run the fine-tuned VLA (ACT) policy to pick the specified screw.
  The policy handles the full pick motion internally; do not call any
  lower-level gripper or movement primitives yourself.

Required args:
  screw_id (str)           — the "id" field from get_observation output
  x, y, z  (float)        — screw position from get_observation (meters, base frame)

Optional args:  (none)

Returns:
  status:         "SUCCESS" or "FAILURE"
  failure_reason: string if FAILURE, else null

After this call you MUST call run_visual_qa to verify the grasp before
proceeding. A "SUCCESS" status from run_pick is optimistic; QA is the
ground truth.

Workflow position:  get_observation → run_pick → run_visual_qa
""",
        fn=_run_pick,
        required_args=["screw_id", "x", "y", "z"],
    ),

    "run_visual_qa": Capability(
        id="run_visual_qa",
        description="Post-pick camera check: verify the screw was successfully grasped.",
        doc="""
Capability: run_visual_qa
--------------------------
Purpose:
  From the current inspection pose, capture a wrist-camera frame and
  determine whether the gripper is actually holding the screw.

Required args:
  screw_id (str) — the screw ID that was just picked (from get_observation)

Optional args:  (none)

Returns:
  grasp_confirmed (bool)  — true if the screw is visibly in the gripper
  confidence      (float) — model confidence in the verdict [0, 1]

Decision logic (agent responsibility):
  - grasp_confirmed=True  → screw removed; call get_observation for next screw
  - grasp_confirmed=False → pick failed; retry run_pick (up to max retries),
                            then move on if still failing to avoid deadlock

Workflow position:  run_pick → run_visual_qa → (get_observation | retry run_pick)
""",
        fn=_run_visual_qa,
        required_args=["screw_id"],
    ),
}


# ---------------------------------------------------------------------------
# Agent-facing interface (called by the LangGraph tool node)
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
        ids = list(REGISTRY.keys())
        return {"error": f"Unknown capability: {capability_id!r}. Available: {ids}"}
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
        ids = list(REGISTRY.keys())
        return {"error": f"Unknown capability: {capability_id!r}. Available: {ids}"}
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
# LangChain tool wrappers (bound to the LangGraph tool node)
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
        capability_id: ID from list_capabilities (e.g. "run_pick")
        args: Capability-specific arguments, e.g.
              {"screw_id": "screw_0", "x": 0.12, "y": -0.05, "z": 0.30}

    Accepted shapes: nested args/params/kwargs dicts are all normalized.
    """
    # The LLM always passes capability args inside the `args` dict parameter.
    # We must flatten that dict directly — it is NOT a wrapper key, it IS the payload.
    # Additional provider-specific wrappers (params, kwargs) are also unwrapped.
    merged: dict[str, Any] = {}

    # Start with the primary `args` parameter — this is the main payload.
    if isinstance(args, dict):
        merged.update(args)

    # Merge `params` if provided (some providers use this instead).
    if isinstance(params, dict):
        merged.update(params)

    # Merge any extra kwargs, but skip known framework-internal keys.
    skip_keys = {"v__args", "type"}
    # If kwargs contains further nested dicts under wrapper keys, unwrap one level.
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
