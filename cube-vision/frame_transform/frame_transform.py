"""Camera-to-arm-base frame transform using numpy FK (no pinocchio dependency).

Kinematic chains are hardcoded from xlerobot_front.urdf.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Rotation / transform helpers
# ---------------------------------------------------------------------------


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    return _rz(y) @ _ry(p) @ _rx(r)


def _tf(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy(*rpy)
    T[:3, 3] = xyz
    return T


def _rot_axis(axis: np.ndarray, q: float) -> np.ndarray:
    """Rodrigues rotation about an arbitrary unit axis."""
    c, s = np.cos(q), np.sin(q)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _revolute(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
    axis: np.ndarray,
    q: float,
) -> np.ndarray:
    T = _tf(xyz, rpy)
    T[:3, :3] = T[:3, :3] @ _rot_axis(axis, q)
    return T


# ---------------------------------------------------------------------------
# Kinematic chains (hardcoded from xlerobot_front.urdf)
# ---------------------------------------------------------------------------

# Arm chain: base_link -> Base_2 -> ... -> Fixed_Jaw_tip_2
_ARM_CHAIN = [
    # arm_base_joint_2 (FIXED)
    ((0.065, 0.133, 0.760), (0.0, 0.0, 1.5708), None),
    # Rotation_2
    ((0.0, -0.0452, 0.0165), (1.5708, 0.0, 0.0), np.array([0.0, -1.0, 0.0])),
    # Pitch_2
    ((0.0, 0.1025, 0.0306), (1.5708, 0.0, 0.0), np.array([-1.0, 0.0, 0.0])),
    # Elbow_2
    ((0.0, 0.11257, 0.028), (-1.5708, 0.0, 0.0), np.array([1.0, 0.0, 0.0])),
    # Wrist_Pitch_2
    ((0.0, 0.0052, 0.1349), (-1.5708, 0.0, 0.0), np.array([1.0, 0.0, 0.0])),
    # Wrist_Roll_2
    ((0.0, -0.0601, 0.0), (0.0, 1.5708, 0.0), np.array([0.0, -1.0, 0.0])),
    # Fixed_Jaw_tip_joint_2 (FIXED)
    ((0.01, -0.097, 0.0), (0.0, 0.0, 0.0), None),
]

# Head chain: base_link -> top_base_link -> ... -> head_camera_depth_optical_frame
_HEAD_CHAIN = [
    # top_base_joint (FIXED)
    ((0.2, 0.0, 0.73), (0.0, 0.0, 0.0), None),
    # head_pan_joint
    ((-0.178, 0.0, 0.0), (0.0, 0.0, 0.0), np.array([0.0, 0.0, 1.0])),
    # head_tilt_joint
    ((0.031, 0.0, 0.43815), (0.0, 0.0, 0.0), np.array([0.0, 1.0, 0.0])),
    # head_camera_joint (FIXED)
    ((0.055, 0.0, 0.0225), (0.0, 0.0, 0.0), None),
    # head_camera_depth_joint (FIXED)
    ((0.0, 0.045, 0.0), (0.0, 0.0, 0.0), None),
    # head_camera_depth_optical_joint (FIXED)
    ((0.0, 0.0, 0.0), (-1.57079632679, 0.0, -1.57079632679), None),
]


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------


def _chain_fk(chain: list, q_rad: np.ndarray) -> np.ndarray:
    """Compute FK through a chain of fixed and revolute joints.

    Entries with axis=None are fixed joints; others are revolute.
    q_rad values are consumed in order for revolute joints only.
    """
    T = np.eye(4)
    qi = 0
    for xyz, rpy, axis in chain:
        if axis is None:
            T = T @ _tf(xyz, rpy)
        else:
            T = T @ _revolute(xyz, rpy, axis, q_rad[qi])
            qi += 1
    return T


def head_fk(q_urdf_rad: np.ndarray) -> np.ndarray:
    """T_baselink_camera given 2 head joint angles in URDF radians."""
    return _chain_fk(_HEAD_CHAIN, q_urdf_rad)


# ---------------------------------------------------------------------------
# Motor <-> URDF convention mapping (head only)
# ---------------------------------------------------------------------------


def _head_motor_to_urdf(q_deg: np.ndarray) -> np.ndarray:
    out = q_deg.copy()
    out[0] = -out[0]
    return out


# ---------------------------------------------------------------------------
# Public API: camera_xyz_to_base_xyz
# ---------------------------------------------------------------------------


def _t_base_camera(q_head_urdf_rad: np.ndarray) -> np.ndarray:
    """4x4 transform from arm Base_2 frame to camera optical frame."""
    T_baselink_base = _tf(*_ARM_CHAIN[0][:2])
    T_baselink_cam = head_fk(q_head_urdf_rad)
    return np.linalg.inv(T_baselink_base) @ T_baselink_cam


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
    urdf_deg = _head_motor_to_urdf(motor_deg)
    q_head_urdf_rad = np.deg2rad(urdf_deg)

    T_base_cam = _t_base_camera(q_head_urdf_rad)

    p_cam = np.array([x, y, z, 1.0], dtype=float)
    p_base = (T_base_cam @ p_cam)[:3]

    return float(p_base[0]), float(p_base[1]), float(p_base[2])
