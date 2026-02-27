"""Reduced arm-only Pinocchio model construction."""

import numpy as np
import pinocchio as pin

from pincer_transform.constants import ARM_JOINTS, ARM_MOTORS, BASE_FRAME, EE_FRAME, URDF_PATH
from pincer_transform.conventions import arm_motor_to_urdf


def build_arm_model(
    limits: dict[str, tuple[float, float]],
) -> tuple[pin.Model, pin.Data, int, int]:
    """Build a reduced (arm-only) Pinocchio model with calibrated joint limits.

    Parameters
    ----------
    limits:
        Per-motor position limits in motor convention (degrees), as returned by
        ``compute_arm_limits``.

    Returns
    -------
    model, data, base_fid, ee_fid
    """
    full = pin.buildModelFromUrdf(str(URDF_PATH))

    keep = set(ARM_JOINTS)
    lock = [
        jid
        for jid in range(1, full.njoints)
        if full.joints[jid].nq > 0 and full.names[jid] not in keep
    ]
    model = pin.buildReducedModel(full, lock, pin.neutral(full))
    data = model.createData()

    base_fid = model.getFrameId(BASE_FRAME)
    ee_fid = model.getFrameId(EE_FRAME)
    if base_fid >= model.nframes or ee_fid >= model.nframes:
        raise RuntimeError("Missing EE or base frame in reduced IK model.")

    # Apply calibrated limits in URDF convention.
    for motor, joint in zip(ARM_MOTORS, ARM_JOINTS):
        lo_m, hi_m = limits[motor]
        if motor == "left_arm_shoulder_lift":
            lo_u, hi_u = 90.0 - hi_m, 90.0 - lo_m
        elif motor == "left_arm_elbow_flex":
            lo_u, hi_u = lo_m + 90.0, hi_m + 90.0
        else:
            lo_u, hi_u = lo_m, hi_m
        lo_u, hi_u = float(min(lo_u, hi_u)), float(max(lo_u, hi_u))
        jid = model.getJointId(joint)
        idx = model.joints[jid].idx_q
        model.lowerPositionLimit[idx] = np.deg2rad(lo_u)
        model.upperPositionLimit[idx] = np.deg2rad(hi_u)

    return model, data, base_fid, ee_fid


def motor_to_pin_q(q_motor: np.ndarray, model: pin.Model) -> np.ndarray:
    """Map arm motor-convention angles (degrees) into a full Pinocchio q vector."""
    q_pin = pin.neutral(model)
    for name, q_deg in zip(ARM_JOINTS, arm_motor_to_urdf(q_motor)):
        jid = model.getJointId(name)
        q_pin[model.joints[jid].idx_q] = np.deg2rad(float(q_deg))
    return q_pin


def pin_to_motor_q(q_pin: np.ndarray, model: pin.Model) -> np.ndarray:
    """Map a Pinocchio q vector back to arm motor-convention angles (degrees)."""
    from pincer_transform.conventions import arm_urdf_to_motor

    q_urdf = np.zeros(len(ARM_JOINTS), dtype=float)
    for i, name in enumerate(ARM_JOINTS):
        jid = model.getJointId(name)
        q_urdf[i] = np.rad2deg(float(q_pin[model.joints[jid].idx_q]))
    return arm_urdf_to_motor(q_urdf)
