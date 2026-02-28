"""Camera-to-arm-base frame transform using pinocchio FK with xlerobot.xml (MJCF).

The model is loaded once at import time and reused for every query.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pinocchio as pin

# ---------------------------------------------------------------------------
# Model (loaded once)
# ---------------------------------------------------------------------------

_MJCF_PATH = Path(__file__).resolve().parent.parent / "assets" / "xlerobot.xml"

_model = pin.buildModelFromMJCF(str(_MJCF_PATH))
_data = _model.createData()

# Frame IDs (resolved once)
_BASE_FRAME_ID = _model.getFrameId("Base_2")
# NOTE: The OP_FRAME "head_camera_rgb_optical_frame" has incorrect position
# due to a pinocchio MJCF loader bug (intermediate body offsets are lost for
# sites). The BODY frame "head_camera_rgb_frame" has both the correct position
# AND the correct optical-frame rotation, so we use that instead.
_CAMERA_FRAME_ID = _model.getFrameId("head_camera_rgb_frame")

# Joint indices for the head (resolved once)
_HEAD_PAN_IDX = _model.joints[_model.getJointId("head_pan_joint")].idx_q
_HEAD_TILT_IDX = _model.joints[_model.getJointId("head_tilt_joint")].idx_q


# ---------------------------------------------------------------------------
# Motor <-> URDF convention mapping (head only)
# ---------------------------------------------------------------------------


def _head_motor_to_mjcf(q_deg: np.ndarray) -> np.ndarray:
    """Convert head motor degrees to MJCF joint degrees.

    Pan (index 0): negated (MJCF pan axis is reversed vs old URDF) then
        -5 deg offset to align motor zero with model zero.
    Tilt (index 1): -10 deg offset to account for motor/model zero mismatch.

    These offsets were empirically calibrated and may need per-robot tuning.
    """
    out = q_deg.copy()
    out[0] = -(out[0] - 5.0)
    out[1] = out[1] - 10.0
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def head_fk(q_urdf_rad: np.ndarray) -> np.ndarray:
    """T_baselink_camera given 2 head joint angles in URDF radians."""
    q = pin.neutral(_model)
    q[_HEAD_PAN_IDX] = q_urdf_rad[0]
    q[_HEAD_TILT_IDX] = q_urdf_rad[1]

    pin.forwardKinematics(_model, _data, q)
    pin.updateFramePlacements(_model, _data)

    return _data.oMf[_CAMERA_FRAME_ID].homogeneous


def camera_xyz_to_base_xyz(
    x: float,
    y: float,
    z: float,
    joint_values: Dict[str, float],
) -> Tuple[float, float, float]:
    """Transform (x, y, z) from camera optical frame into the arm Base_2 frame.

    joint_values must include:
      - "head_pan_joint":  head pan in radians  (motor convention, sign is flipped internally)
      - "head_tilt_joint": head tilt in radians (motor convention)
    """
    pan_motor_rad = joint_values.get("head_pan_joint", 0.0)
    tilt_motor_rad = joint_values.get("head_tilt_joint", 0.0)

    # Convert motor-convention values to URDF convention
    motor_deg = np.array([np.rad2deg(pan_motor_rad), np.rad2deg(tilt_motor_rad)])
    urdf_deg = _head_motor_to_mjcf(motor_deg)
    q_head_urdf_rad = np.deg2rad(urdf_deg)

    # Build full configuration with head joints set
    q = pin.neutral(_model)
    q[_HEAD_PAN_IDX] = q_head_urdf_rad[0]
    q[_HEAD_TILT_IDX] = q_head_urdf_rad[1]

    pin.forwardKinematics(_model, _data, q)
    pin.updateFramePlacements(_model, _data)

    oMbase = _data.oMf[_BASE_FRAME_ID]
    oMcam = _data.oMf[_CAMERA_FRAME_ID]

    # T_base_camera: transform points from camera frame to Base_2 frame
    R = oMbase.rotation.T @ oMcam.rotation
    t = oMbase.rotation.T @ (oMcam.translation - oMbase.translation)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    p_cam = np.array([x, y, z, 1.0], dtype=float)
    p_base = (T @ p_cam)[:3]

    return float(p_base[0]), float(p_base[1]), float(p_base[2])
