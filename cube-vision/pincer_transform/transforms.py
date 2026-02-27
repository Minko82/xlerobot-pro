"""Camera-to-base frame transforms.

This module is IK-agnostic: it only deals with rigid-body transforms between
coordinate frames. The IK solver is free to use whatever T_base_camera it
receives without caring how it was built.

Typical usage
-------------
    T = build_t_base_camera(model, data, q_arm, q_head)
    p_base = camera_to_base(p_camera, T)
"""

import numpy as np
import pinocchio as pin

from pincer_transform.constants import (
    ARM_JOINTS,
    BASE_FRAME,
    CAMERA_FRAME,
    HEAD_JOINTS,
    URDF_PATH,
)
from pincer_transform.conventions import arm_motor_to_urdf, head_motor_to_urdf


def build_t_base_camera(
    q_arm: np.ndarray,
    q_head: np.ndarray,
) -> np.ndarray:
    """Return the 4x4 homogeneous transform T_base_camera.

    The transform maps points expressed in the camera frame to the robot base
    frame.  It is computed from the full URDF model at the given joint
    configuration, so head pose matters.

    Parameters
    ----------
    q_arm:
        Arm joint angles in *motor* convention, degrees, shape (5,).
    q_head:
        Head joint angles in *motor* convention, degrees, shape (2,).

    Returns
    -------
    np.ndarray
        4x4 homogeneous transform T such that p_base = T @ [x, y, z, 1].
    """
    model = pin.buildModelFromUrdf(str(URDF_PATH))
    data = model.createData()

    q_full = pin.neutral(model)
    for name, q_deg in zip(ARM_JOINTS, arm_motor_to_urdf(q_arm)):
        jid = model.getJointId(name)
        q_full[model.joints[jid].idx_q] = np.deg2rad(float(q_deg))
    q_head_urdf = head_motor_to_urdf(q_head)
    for name, q_deg in zip(HEAD_JOINTS, q_head_urdf):
        jid = model.getJointId(name)
        idx = model.joints[jid].idx_q
        q_full[idx] = np.deg2rad(float(q_deg))

    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)

    oMbase = data.oMf[model.getFrameId(BASE_FRAME)]
    oMcam = data.oMf[model.getFrameId(CAMERA_FRAME)]

    R = oMbase.rotation.T @ oMcam.rotation
    t = oMbase.rotation.T @ (oMcam.translation - oMbase.translation)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def camera_to_base(p_camera: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform a 3D point from camera frame to base frame.

    Parameters
    ----------
    p_camera:
        Point in camera frame, shape (3,).
    T:
        4x4 homogeneous transform from ``build_t_base_camera``.

    Returns
    -------
    np.ndarray
        Point in base frame, shape (3,).
    """
    p_h = np.array([p_camera[0], p_camera[1], p_camera[2], 1.0], dtype=float)
    return (T @ p_h)[:3]
