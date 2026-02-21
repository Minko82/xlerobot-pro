import signal
import time

import numpy as np
import pyrealsense2 as rs

from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
from pincer.cameras.d435 import D435, D435Config

PORT = "/dev/ttyACM0"
CAMERA_SERIAL = "310222077874"

ARM_MOTORS = [
    "left_arm_shoulder_pan",
    "left_arm_shoulder_lift",
    "left_arm_elbow_flex",
    "left_arm_wrist_flex",
    "left_arm_wrist_roll",
]
HEAD_MOTORS = ["head_motor_1", "head_motor_2"]


# ---------------------------------------------------------------------------
# Transform helpers
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
# Forward kinematics (hardcoded from xlerobot_front.urdf)
# ---------------------------------------------------------------------------

# Arm chain: base_link -> Base_2 -> ... -> Fixed_Jaw_tip_2
# Each revolute entry: (origin_xyz, origin_rpy, axis)
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


def arm_fk(q_urdf_rad: np.ndarray) -> np.ndarray:
    """T_baselink_ee given 5 arm joint angles in URDF radians."""
    return _chain_fk(_ARM_CHAIN, q_urdf_rad)


def head_fk(q_urdf_rad: np.ndarray) -> np.ndarray:
    """T_baselink_camera given 2 head joint angles in URDF radians."""
    return _chain_fk(_HEAD_CHAIN, q_urdf_rad)


def ee_in_base(q_urdf_rad: np.ndarray) -> np.ndarray:
    """EE position in Base_2 frame."""
    T = arm_fk(q_urdf_rad)
    # arm_fk already starts from base_link through arm_base_joint_2 (Base_2),
    # so T is T_baselink_ee. T_baselink_base is the first fixed joint.
    T_base = _tf(*_ARM_CHAIN[0][:2])
    T_base_ee = np.linalg.inv(T_base) @ T
    return T_base_ee[:3, 3]


def t_base_camera(q_arm_urdf_rad: np.ndarray, q_head_urdf_rad: np.ndarray) -> np.ndarray:
    """4x4 transform from Base_2 to camera optical frame."""
    T_base = arm_fk(np.zeros(5))  # only the fixed joint matters; arm angles irrelevant
    # Actually we just need T_baselink_base which is the first fixed transform.
    T_baselink_base = _tf(*_ARM_CHAIN[0][:2])
    T_baselink_cam = head_fk(q_head_urdf_rad)
    return np.linalg.inv(T_baselink_base) @ T_baselink_cam


# ---------------------------------------------------------------------------
# Motor <-> URDF convention mapping
# ---------------------------------------------------------------------------


def arm_motor_to_urdf(q_deg: np.ndarray) -> np.ndarray:
    out = q_deg.copy()
    out[1] = 90.0 - out[1]
    out[2] = out[2] + 90.0
    return out


def arm_urdf_to_motor(q_deg: np.ndarray) -> np.ndarray:
    out = q_deg.copy()
    out[1] = 90.0 - out[1]
    out[2] = out[2] - 90.0
    return out


def head_motor_to_urdf(q_deg: np.ndarray) -> np.ndarray:
    out = q_deg.copy()
    out[0] = -out[0]  # critical for correct camera left/right mapping
    return out


# ---------------------------------------------------------------------------
# Robot setup helpers
# ---------------------------------------------------------------------------


def configure_arm_motors(bus: FeetechMotorsBus) -> None:
    bus.disable_torque()
    for m in ARM_MOTORS:
        bus.write("Operating_Mode", m, OperatingMode.POSITION.value)
        bus.write("Torque_Limit", m, 800)
        bus.write("Acceleration", m, 80)
        bus.write("P_Coefficient", m, 16)
        bus.write("I_Coefficient", m, 0)
        bus.write("D_Coefficient", m, 32)
    bus.enable_torque()


def compute_arm_limits(bus: FeetechMotorsBus) -> dict[str, tuple[float, float]]:
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


def read_q(bus: FeetechMotorsBus, names: list[str]) -> np.ndarray:
    q = bus.sync_read("Present_Position", names)
    return np.array([q[n] for n in names], dtype=float)


def clip_arm(q: np.ndarray, limits: dict[str, tuple[float, float]]) -> np.ndarray:
    out = q.copy()
    for i, n in enumerate(ARM_MOTORS):
        lo, hi = limits[n]
        out[i] = float(np.clip(out[i], lo, hi))
    return out


def motor_to_urdf_rad(q_motor_deg: np.ndarray) -> np.ndarray:
    return np.deg2rad(arm_motor_to_urdf(q_motor_deg))


def urdf_rad_to_motor(q_urdf_rad: np.ndarray) -> np.ndarray:
    return arm_urdf_to_motor(np.rad2deg(q_urdf_rad))


# ---------------------------------------------------------------------------
# IK solver (damped least squares with numerical Jacobian)
# ---------------------------------------------------------------------------

_IK_DQ = 1e-4  # finite-difference step for Jacobian (rad)
_IK_LAMBDA = 1e-3  # damping factor
_IK_ALPHA = 0.5  # step size


def _numerical_jacobian(q_rad: np.ndarray) -> np.ndarray:
    """3x5 positional Jacobian via finite differences."""
    p0 = ee_in_base(q_rad)
    J = np.zeros((3, len(q_rad)))
    for i in range(len(q_rad)):
        q_pert = q_rad.copy()
        q_pert[i] += _IK_DQ
        J[:, i] = (ee_in_base(q_pert) - p0) / _IK_DQ
    return J


def solve_to_target(
    q_seed_motor: np.ndarray,
    target: np.ndarray,
    limits: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, bool, int, float]:
    """Position-only IK using damped least squares.

    Returns (q_motor_deg, converged, iterations, final_error_m).
    """
    q_motor = q_seed_motor.copy()
    q_rad = motor_to_urdf_rad(q_motor)

    # Compute URDF-space limits for clamping.
    urdf_lo = np.zeros(5)
    urdf_hi = np.zeros(5)
    for i, m in enumerate(ARM_MOTORS):
        lo_m, hi_m = limits[m]
        lo_u, hi_u = arm_motor_to_urdf(np.array([lo_m if j == i else 0.0 for j in range(5)]))[i], \
                     arm_motor_to_urdf(np.array([hi_m if j == i else 0.0 for j in range(5)]))[i]
        urdf_lo[i] = np.deg2rad(min(lo_u, hi_u))
        urdf_hi[i] = np.deg2rad(max(lo_u, hi_u))

    prev_err = float("inf")
    flat_steps = 0

    for step in range(300):
        p_ee = ee_in_base(q_rad)
        e = target - p_ee
        err = float(np.linalg.norm(e))

        if err <= 0.005:
            return urdf_rad_to_motor(q_rad), True, step, err

        J = _numerical_jacobian(q_rad)
        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^{-1} e
        JJT = J @ J.T + (_IK_LAMBDA ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, e)
        q_rad = np.clip(q_rad + _IK_ALPHA * dq, urdf_lo, urdf_hi)

        q_motor_next = clip_arm(urdf_rad_to_motor(q_rad), limits)
        q_rad = motor_to_urdf_rad(q_motor_next)

        if float(np.max(np.abs(q_motor_next - q_motor))) <= 0.02 and (prev_err - err) <= 2e-4:
            flat_steps += 1
        else:
            flat_steps = 0
        q_motor = q_motor_next
        prev_err = err
        if flat_steps >= 25:
            return q_motor, False, step + 1, err

    final_err = float(np.linalg.norm(target - ee_in_base(q_rad)))
    return urdf_rad_to_motor(q_rad), False, 300, final_err


def pan_guess_deg(target: np.ndarray) -> float:
    return float(np.rad2deg(np.arctan2(float(target[0]), max(1e-6, -float(target[1])))))


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup(robot: XLerobot | None, camera: D435 | None) -> None:
    if camera is not None:
        try:
            camera.stop()
        except Exception:
            pass
    if robot is not None:
        try:
            if robot.bus1.is_connected:
                robot.bus1.disconnect(robot.config.disable_torque_on_disconnect)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    robot: XLerobot | None = None
    camera: D435 | None = None
    running = True

    def on_sigint(sig, frame):
        del sig, frame
        nonlocal running
        running = False
        print("\nStopping...")

    signal.signal(signal.SIGINT, on_sigint)

    try:
        # Connect robot and restore calibration.
        robot = XLerobot(XLerobotConfig(port1=PORT, use_degrees=True))
        robot.bus1.connect()
        bus = robot.bus1

        bus1_calib = {k: v for k, v in robot.calibration.items() if k in bus.motors}
        if not bus1_calib:
            raise RuntimeError(
                f"No bus1 calibration at {robot.calibration_fpath}. "
                "Run calibration once before using this script."
            )
        input(f"Calibration found for robot id '{robot.id}'. Press ENTER to restore and continue: ")
        bus.calibration = bus1_calib
        bus.write_calibration(bus1_calib)
        print(f"Bus1 calibration restored. is_calibrated={bus.is_calibrated}\n")

        # Configure arm motors.
        configure_arm_motors(bus)

        limits = compute_arm_limits(bus)

        # Start camera.
        camera = D435(D435Config(serial=CAMERA_SERIAL, width=640, height=480, fps=30, align_to_color=False))
        camera.start()

        # Read joints and compute camera->base transform.
        q_seed = read_q(bus, ARM_MOTORS)
        try:
            q_head = read_q(bus, HEAD_MOTORS)
        except Exception:
            q_head = np.zeros(2, dtype=float)

        T_base_cam = t_base_camera(
            motor_to_urdf_rad(q_seed),
            np.deg2rad(head_motor_to_urdf(q_head)),
        )

        # Capture center-depth target in base frame.
        while True:
            _, depth_image, depth_frame = camera.read()
            h, w = depth_image.shape
            u, v = w // 2, h // 2
            z = depth_frame.get_distance(u, v)
            if z <= 0:
                continue
            x, y, z = rs.rs2_deproject_pixel_to_point(camera.intrinsics, [u, v], z)  # type: ignore[attr-defined]
            p_target = (T_base_cam @ np.array([x, y, z, 1.0], dtype=float))[:3]
            break

        # Initial IK solve.
        q_seed = clip_arm(q_seed, limits)
        q_seed[0] = float(np.clip(pan_guess_deg(p_target), *limits["left_arm_shoulder_pan"]))
        q_goal, ok, iters, ik_err = solve_to_target(q_seed, p_target, limits)
        q_goal = clip_arm(q_goal, limits)

        print(f"IK result: ok={ok}, iters={iters}, planned_err_m={ik_err:.4f}, target_base={p_target}")
        if ik_err > 0.05:
            print(f"[WARN] IK residual is high ({ik_err:.3f} m).")
        print("Running one-shot execution. Press Ctrl+C to stop.")

        # Execution loop.
        period = 1.0 / 10.0
        hold = 0
        i = 0
        while running:
            t0 = time.monotonic()

            q_curr = clip_arm(read_q(bus, ARM_MOTORS), limits)
            p_ee = ee_in_base(motor_to_urdf_rad(q_curr))
            err_vec = p_target - p_ee
            err = float(np.linalg.norm(err_vec))

            if err <= 0.01:
                hold += 1
                if hold >= 3:
                    print(f"Reached target in base frame (ee_err_m={err:.4f}).")
                    break
            else:
                hold = 0

            q_seed_step = q_curr.copy()
            if abs(float(err_vec[0])) >= 0.01:
                q_seed_step[0] = float(np.clip(pan_guess_deg(p_target), *limits["left_arm_shoulder_pan"]))
            q_goal, _, _, _ = solve_to_target(q_seed_step, p_target, limits)

            step = 10.0 if err > 0.12 else 6.0
            q_cmd = clip_arm(np.clip(q_goal, q_curr - step, q_curr + step), limits)
            bus.sync_write("Goal_Position", {n: float(q_cmd[j]) for j, n in enumerate(ARM_MOTORS)})

            if i % 20 == 0:
                print(
                    f"err_norm={err:.4f}, step_deg={step:.1f}, "
                    f"q_pan={q_curr[0]:.2f}, q_lift={q_curr[1]:.2f}, q_elbow={q_curr[2]:.2f}"
                )
            i += 1

            sleep_s = period - (time.monotonic() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    finally:
        cleanup(robot, camera)


if __name__ == "__main__":
    main()
