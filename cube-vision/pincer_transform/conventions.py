"""Motor <-> URDF angle convention mappings for the XLerobot arm and head."""

import numpy as np


def arm_motor_to_urdf(q_deg: np.ndarray) -> np.ndarray:
    """Convert arm joint angles from motor convention to URDF convention (degrees)."""
    out = q_deg.copy()
    out[1] = 90.0 - out[1]
    out[2] = out[2] + 90.0
    return out


def arm_urdf_to_motor(q_deg: np.ndarray) -> np.ndarray:
    """Convert arm joint angles from URDF convention to motor convention (degrees)."""
    out = q_deg.copy()
    out[1] = 90.0 - out[1]
    out[2] = out[2] - 90.0
    return out


def head_motor_to_urdf(q_deg: np.ndarray) -> np.ndarray:
    """Convert head joint angles from motor convention to URDF convention (degrees).

    Pan (index 0): negated with +5 deg offset to align motor zero with URDF zero.
    Tilt (index 1): -10 deg offset to account for motor/URDF zero mismatch.

    These offsets were empirically calibrated and may need per-robot tuning.
    """
    out = q_deg.copy()
    out[0] = out[0] - 5.0
    out[1] = out[1] - 10.0
    return out
