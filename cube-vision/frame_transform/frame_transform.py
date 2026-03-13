"""Camera-to-arm-base transforms using explicit canonical frames.

Canonical robot-frame convention throughout this module:
  X = forward
  Y = right
  Z = up

Input camera detections are still assumed to be in the RealSense optical frame:
  X = right
  Y = down
  Z = forward

The XML now exposes canonical arm-base frames directly, while the physical arm
mounting remains unchanged under nested bodies. The transform path is therefore:

  optical point -> head_camera_link -> world -> Base / Base_2
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pinocchio as pin

# ---------------------------------------------------------------------------
# Model (loaded once)
# ---------------------------------------------------------------------------

_MJCF_PATH = Path(__file__).resolve().parent / "xlerobot" / "xlerobot.xml"

_model = pin.buildModelFromMJCF(str(_MJCF_PATH))
_data = _model.createData()

# Frame IDs (resolved once)
_BASE_FRAME_ID = _model.getFrameId("Base")
_BASE_2_FRAME_ID = _model.getFrameId("Base_2")
_CAMERA_LINK_FRAME_ID = _model.getFrameId("head_camera_link")

# Optical -> canonical camera-link coordinates:
#   forward = optical Z
#   right   = optical X
#   up      = -optical Y
_R_CAMERA_LINK_FROM_OPTICAL = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])

# Joint indices for the head (resolved once)
_HEAD_PAN_IDX = _model.joints[_model.getJointId("head_pan_joint")].idx_q
_HEAD_TILT_IDX = _model.joints[_model.getJointId("head_tilt_joint")].idx_q


# ---------------------------------------------------------------------------
# Motor <-> MJCF convention mapping (head only)
# ---------------------------------------------------------------------------


def _head_motor_to_mjcf(q_deg: np.ndarray) -> np.ndarray:
    """Convert head motor degrees to MJCF joint degrees.

    Pan (index 0): same sign as motor convention.
    Tilt (index 1): offset only (positive motor tilt = positive MJCF tilt = down).

    Offsets calibrated with head facing straight forward and level:
        pan motor reads ~1°, tilt motor reads ~14° at MJCF zero.
    """
    out = q_deg.copy()
    out[1] = out[1] - 14.0
    return out


def _head_joint_configuration(joint_values: Dict[str, float]) -> np.ndarray:
    """Build a model configuration with the head joints set from motor radians."""
    pan_motor_rad = joint_values.get("head_pan_joint", 0.0)
    tilt_motor_rad = joint_values.get("head_tilt_joint", 0.0)

    motor_deg = np.array([np.rad2deg(pan_motor_rad), np.rad2deg(tilt_motor_rad)])
    mjcf_deg = _head_motor_to_mjcf(motor_deg)
    q = pin.neutral(_model)
    q[_HEAD_PAN_IDX] = np.deg2rad(mjcf_deg[0])
    q[_HEAD_TILT_IDX] = np.deg2rad(mjcf_deg[1])
    return q


def _update_head_fk(joint_values: Dict[str, float]) -> None:
    """Run FK for the current head pose."""
    q = _head_joint_configuration(joint_values)
    pin.forwardKinematics(_model, _data, q)
    pin.updateFramePlacements(_model, _data)


def _frame_transform(dst_frame_id: int, src_frame_id: int) -> np.ndarray:
    """Return T_dst_src for two Pinocchio frames."""
    oMdst = _data.oMf[dst_frame_id]
    oMsrc = _data.oMf[src_frame_id]

    T = np.eye(4)
    T[:3, :3] = oMdst.rotation.T @ oMsrc.rotation
    T[:3, 3] = oMdst.rotation.T @ (oMsrc.translation - oMdst.translation)
    return T


def _optical_point_to_camera_link(x: float, y: float, z: float) -> np.ndarray:
    """Convert a RealSense optical-frame point into canonical camera-link axes."""
    p_optical = np.array([x, y, z], dtype=float)
    return _R_CAMERA_LINK_FROM_OPTICAL @ p_optical


def _camera_optical_to_base(
    x: float,
    y: float,
    z: float,
    joint_values: Dict[str, float],
    base_frame_id: int,
) -> Tuple[float, float, float]:
    """Transform a RealSense optical-frame point into a canonical arm-base frame."""
    _update_head_fk(joint_values)
    T_base_camera = _frame_transform(base_frame_id, _CAMERA_LINK_FRAME_ID)
    p_camera = _optical_point_to_camera_link(x, y, z)
    p_base = (T_base_camera @ np.append(p_camera, 1.0))[:3]
    return float(p_base[0]), float(p_base[1]), float(p_base[2])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def head_fk(q_urdf_rad: np.ndarray) -> np.ndarray:
    """Return the world pose of `head_camera_link` in canonical camera axes."""
    q = pin.neutral(_model)
    q[_HEAD_PAN_IDX] = q_urdf_rad[0]
    q[_HEAD_TILT_IDX] = q_urdf_rad[1]

    pin.forwardKinematics(_model, _data, q)
    pin.updateFramePlacements(_model, _data)

    oMf = _data.oMf[_CAMERA_LINK_FRAME_ID]
    T = np.eye(4)
    T[:3, :3] = oMf.rotation
    T[:3, 3] = oMf.translation
    return T


def camera_xyz_to_base_xyz(
    x: float,
    y: float,
    z: float,
    joint_values: Dict[str, float],
) -> Tuple[float, float, float]:
    """Transform a RealSense optical-frame point into the left arm base frame.

    Returned coordinates follow the canonical robot convention:
      X = forward
      Y = right
      Z = up
    """
    return _camera_optical_to_base(x, y, z, joint_values, _BASE_FRAME_ID)


def camera_xyz_to_base2_xyz(
    x: float,
    y: float,
    z: float,
    joint_values: Dict[str, float],
) -> Tuple[float, float, float]:
    """Transform a RealSense optical-frame point into the right arm base frame.

    Returned coordinates follow the canonical robot convention:
      X = forward
      Y = right
      Z = up
    """
    return _camera_optical_to_base(x, y, z, joint_values, _BASE_2_FRAME_ID)
