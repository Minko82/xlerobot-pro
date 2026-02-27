"""Manually position the gripper on a target, then read q and model EE pos."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pinocchio as pin

from lerobot.robots.xlerobot.xlerobot import XLerobot
from lerobot.robots.xlerobot.config_xlerobot import XLerobotConfig
from pincer_transform.constants import ARM_MOTORS
from pincer_transform.model import build_arm_model, motor_to_pin_q

PORT = "/dev/ttyACM0"


def read_q(bus, names: list[str]) -> np.ndarray:
    """Read present motor positions (degrees) for the given motor names."""
    q = bus.sync_read("Present_Position", names)
    return np.array([q[n] for n in names], dtype=float)


def compute_arm_limits(bus) -> dict[str, tuple[float, float]]:
    """Compute per-motor position limits (degrees) from calibration data."""
    limits: dict[str, tuple[float, float]] = {}
    for m in ARM_MOTORS:
        cal = bus.calibration.get(m)
        if cal is None:
            raise RuntimeError(f"Missing calibration for motor '{m}'.")
        max_res = bus.model_resolution_table[bus.motors[m].model] - 1
        mid = (cal.range_min + cal.range_max) / 2.0
        lo = (cal.range_min - mid) * 360.0 / max_res + 0.5
        hi = (cal.range_max - mid) * 360.0 / max_res - 0.5
        limits[m] = (float(min(lo, hi)), float(max(lo, hi)))
    return limits


def ee_in_base(
    q_motor: np.ndarray,
    model: pin.Model,
    data: pin.Data,
    base_fid: int,
    ee_fid: int,
) -> np.ndarray:
    """Return the EE position in the base frame given motor-convention angles (degrees)."""
    q_pin = motor_to_pin_q(q_motor, model)
    pin.forwardKinematics(model, data, q_pin)
    pin.updateFramePlacements(model, data)

    oMbase = data.oMf[base_fid]
    oMee = data.oMf[ee_fid]

    return oMbase.rotation.T @ (oMee.translation - oMbase.translation)


def main() -> None:
    robot = XLerobot(XLerobotConfig(port1=PORT, use_degrees=True))
    robot.bus1.connect()
    bus = robot.bus1

    bus1_calib = {k: v for k, v in robot.calibration.items() if k in bus.motors}
    if not bus1_calib:
        raise RuntimeError("No calibration found.")
    bus.calibration = bus1_calib
    bus.write_calibration(bus1_calib)

    limits = compute_arm_limits(bus)
    model, data, base_fid, ee_fid = build_arm_model(limits)

    bus.disable_torque()
    print("Torque disabled. Manually move the gripper tip to the target point.")
    input("Press ENTER when the gripper tip is on the target...")

    q = read_q(bus, ARM_MOTORS)
    p_ee = ee_in_base(q, model, data, base_fid, ee_fid)

    print(f"\nq_motor (deg): {q}")
    print(f"Model EE pos (base, m): {p_ee}")
    print(f"Model EE reach (XY, m): {np.linalg.norm(p_ee[:2]):.4f}")
    print(f"Model EE Z (m): {p_ee[2]:.4f}")

    bus.disable_torque()


if __name__ == "__main__":
    main()
