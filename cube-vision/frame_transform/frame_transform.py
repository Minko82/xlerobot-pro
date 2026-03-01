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
# NOTE: "head_camera_rgb_frame" has an optical-frame euler that makes its Z axis
# align with the tilt rotation axis (Y), so tilt has no effect on its orientation
# in pinocchio. Instead we use "head_camera_link" (which tilts correctly) and
# apply the optical frame rotation manually.
_CAMERA_LINK_FRAME_ID = _model.getFrameId("head_camera_link")

# Rotation from camera_link frame to optical frame.
# camera_link axes at neutral: X = forward (-X world), Y = right (-Y world), Z = up (+Z world)
# Optical convention: Z = forward, X = right, Y = down
# So: optical_Z = link_X, optical_X = link_Y, optical_Y = -link_Z
_R_LINK_TO_OPTICAL = np.array([
    [0,  0, 1],   # optical X (right) = link Y ... column 0 = link coords of optical X
    [1,  0, 0],   # wait, let me be precise
    [0, -1, 0],
])
# Columns of R_link_to_optical are the optical axes expressed in link frame:
#   col 0 (optical X = right):   link Y  = [0, 1, 0]
#   col 1 (optical Y = down):   -link Z  = [0, 0, -1]
#   col 2 (optical Z = forward): link X  = [1, 0, 0]
_R_LINK_TO_OPTICAL = np.array([
    [0,  0, 1],
    [1,  0, 0],
    [0, -1, 0],
])

# Empirical correction between the MJCF camera chain and the arm Base_2 frame
# used by IK in this project. Without this, camera-forward points map mostly to
# Base_2 +X (sideways) instead of Base_2 -Y (forward).
_R_BASE2_CORRECTION = np.array([
    [0.0, 1.0, 0.0],   # x' = y
    [-1.0, 0.0, 0.0],  # y' = -x
    [0.0, 0.0, 1.0],   # z' = z
])

# Joint indices for the head (resolved once)
_HEAD_PAN_IDX = _model.joints[_model.getJointId("head_pan_joint")].idx_q
_HEAD_TILT_IDX = _model.joints[_model.getJointId("head_tilt_joint")].idx_q


# ---------------------------------------------------------------------------
# Motor <-> MJCF convention mapping (head only)
# ---------------------------------------------------------------------------


def _head_motor_to_mjcf(q_deg: np.ndarray) -> np.ndarray:
    """Convert head motor degrees to MJCF joint degrees.

    Pan (index 0): negated (MJCF pan axis sign is opposite motor convention)
        then offset to align motor zero with model zero.
    Tilt (index 1): offset only (positive motor tilt = positive MJCF tilt = down).

    Offsets calibrated with head facing straight forward and level:
        pan motor reads ~1°, tilt motor reads ~14° at MJCF zero.
    """
    out = q_deg.copy()
    out[0] = -(out[0] - 1.0)
    out[1] = out[1] - 14.0
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

    oMf = _data.oMf[_CAMERA_LINK_FRAME_ID]
    # Apply optical frame rotation
    R_optical = oMf.rotation @ _R_LINK_TO_OPTICAL
    T = np.eye(4)
    T[:3, :3] = R_optical
    T[:3, 3] = oMf.translation
    return T


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

    # Convert motor-convention values to MJCF convention
    motor_deg = np.array([np.rad2deg(pan_motor_rad), np.rad2deg(tilt_motor_rad)])
    mjcf_deg = _head_motor_to_mjcf(motor_deg)
    q_head_mjcf_rad = np.deg2rad(mjcf_deg)

    # Build full configuration with head joints set
    q = pin.neutral(_model)
    q[_HEAD_PAN_IDX] = q_head_mjcf_rad[0]
    q[_HEAD_TILT_IDX] = q_head_mjcf_rad[1]

    pin.forwardKinematics(_model, _data, q)
    pin.updateFramePlacements(_model, _data)

    oMbase = _data.oMf[_BASE_FRAME_ID]
    oMcam_link = _data.oMf[_CAMERA_LINK_FRAME_ID]

    # Build optical frame pose: same position as camera_link, rotated to optical convention
    R_cam_optical = oMcam_link.rotation @ _R_LINK_TO_OPTICAL

    # T_base_camera: transform points from camera optical frame to Base_2 frame
    R = oMbase.rotation.T @ R_cam_optical
    t = oMbase.rotation.T @ (oMcam_link.translation - oMbase.translation)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    p_cam = np.array([x, y, z, 1.0], dtype=float)
    p_base = (T @ p_cam)[:3]
    p_base = _R_BASE2_CORRECTION @ p_base

    return float(p_base[0]), float(p_base[1]), float(p_base[2])
