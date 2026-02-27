"""Calibrate T_base_camera directly from multiple gripper-on-object measurements.

Since the URDF head geometry is inaccurate, this bypasses the head kinematic
chain and solves for the 4x4 rigid transform from camera optical frame to
Base_2 frame using paired (EE position, camera centroid) measurements.

Procedure
---------
1. Run this script.
2. For each sample: place the gripper tip on the object, press ENTER to record,
   then move the arm away and press ENTER to capture the camera view.
   Repeat at 3+ different object positions (move the object each time).
3. The script solves for T_base_camera and saves it for use by control.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import numpy as np
import pinocchio as pin
import json

from lerobot.robots.xlerobot.xlerobot import XLerobot
from lerobot.robots.xlerobot.config_xlerobot import XLerobotConfig
from lerobot.motors.motors_bus import MotorCalibration

from pincer_transform.constants import (
    ARM_JOINTS, ARM_MOTORS, BASE_FRAME, EE_FRAME, URDF_PATH,
    HEAD_MOTORS,
)
from pincer_transform.conventions import arm_motor_to_urdf
from pincer_transform.model import build_arm_model, motor_to_pin_q

from point_cloud import PointCloud
from realsense_capture import capture

PORT = "/dev/ttyACM0"
HEAD_CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "calibration" / "head.json"
CALIBRATION_OUT = Path(__file__).resolve().parent.parent / "calibration" / "t_base_camera.json"


def read_motors(bus, names):
    raw = bus.sync_read("Present_Position", names)
    return np.array([raw[n] for n in names], dtype=float)


def ee_in_base(q_motor, model, data, base_fid, ee_fid):
    q_pin = motor_to_pin_q(q_motor, model)
    pin.forwardKinematics(model, data, q_pin)
    pin.updateFramePlacements(model, data)
    oMbase = data.oMf[base_fid]
    oMee = data.oMf[ee_fid]
    return oMbase.rotation.T @ (oMee.translation - oMbase.translation)


def compute_arm_limits(bus):
    limits = {}
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


def detect_object(capture_dir):
    """Capture and return the centroid of the best object candidate."""
    os.chdir(str(Path(__file__).resolve().parent.parent))
    capture()
    pc = PointCloud(captures_dir=str(capture_dir))
    pc.create_point_cloud_from_rgbd()
    pc.segment_plane()
    objects = pc.dbscan_objects(min_points_per_object=200)
    if not objects:
        raise RuntimeError("No objects detected")

    print(f"\nDetected {len(objects)} object(s):")
    for i, obj in enumerate(objects):
        print(f"  [{i}] centroid={obj['centroid']}, {obj['num_points']} points")

    while True:
        choice = input(f"Which object is the target? [0-{len(objects)-1}]: ").strip()
        try:
            idx = int(choice)
            if 0 <= idx < len(objects):
                break
        except ValueError:
            pass
        print("Invalid choice, try again.")

    centroid = objects[idx]["centroid"]
    print(f"Selected object [{idx}]: centroid={centroid}")
    return centroid


def solve_rigid_transform(pts_camera, pts_base):
    """Solve for T (4x4) such that pts_base ≈ T @ pts_camera (homogeneous).

    Uses SVD-based least-squares for the rotation, then solves for translation.
    Requires >= 3 non-collinear point pairs.
    """
    assert pts_camera.shape == pts_base.shape
    n = pts_camera.shape[0]

    # Centroids
    c_cam = pts_camera.mean(axis=0)
    c_base = pts_base.mean(axis=0)

    # Center the points
    Q = pts_camera - c_cam  # (n, 3)
    P = pts_base - c_base   # (n, 3)

    # Cross-covariance
    H = Q.T @ P  # (3, 3)

    U, S, Vt = np.linalg.svd(H)
    # Ensure proper rotation (det = +1)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T

    t = c_base - R @ c_cam

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def main():
    capture_dir = Path(__file__).resolve().parent.parent / "outputs" / "realsense_capture"

    # --- Connect and prepare ---
    robot = XLerobot(XLerobotConfig(port1=PORT, use_degrees=True))
    robot.bus1.connect()
    bus = robot.bus1

    # Load head calibration
    if not HEAD_CALIBRATION_FILE.exists():
        raise FileNotFoundError(f"Head calibration not found: {HEAD_CALIBRATION_FILE}")
    with open(HEAD_CALIBRATION_FILE) as f:
        head_calib_raw = json.load(f)
    head_calibration = {
        name: MotorCalibration(**vals) for name, vals in head_calib_raw.items()
    }
    bus.calibration = {**bus.calibration, **head_calibration}

    # Build arm model for FK
    bus_calib = {k: v for k, v in robot.calibration.items() if k in bus.motors}
    if bus_calib:
        bus.calibration = {**bus.calibration, **bus_calib}
        bus.write_calibration(bus_calib)
    limits = compute_arm_limits(bus)
    model_arm, data_arm, base_fid, ee_fid = build_arm_model(limits)

    bus.disable_torque()

    # --- Collect calibration samples ---
    pts_camera = []
    pts_base = []
    sample_idx = 0
    MIN_SAMPLES = 3

    print("=" * 60)
    print("CAMERA-TO-BASE CALIBRATION")
    print("=" * 60)
    print(f"Collect at least {MIN_SAMPLES} samples.")
    print("For each sample:")
    print("  1. Place the gripper tip on the object")
    print("  2. Press ENTER to record arm position")
    print("  3. Move the arm out of the way (don't move the object or head)")
    print("  4. Press ENTER to capture the camera view")
    print("  5. Move the object to a new position and repeat")
    print("Type 'done' when finished (after >= 3 samples).\n")

    while True:
        user = input(f"--- Sample {sample_idx + 1}: Place gripper on object, ENTER to record (or 'done'): ").strip()
        if user.lower() == "done":
            if len(pts_camera) < MIN_SAMPLES:
                print(f"Need at least {MIN_SAMPLES} samples, have {len(pts_camera)}. Keep going.")
                continue
            break

        # Read arm motors and compute EE
        q_arm = read_motors(bus, ARM_MOTORS)
        p_ee = ee_in_base(q_arm, model_arm, data_arm, base_fid, ee_fid)
        print(f"  Arm motor (deg): {q_arm}")
        print(f"  EE (Base_2, m):  {p_ee}")

        input("  Now move the arm out of the way. Press ENTER to capture...")

        # Capture and detect
        bus.disconnect()
        centroid = detect_object(capture_dir)

        # Reconnect for next sample
        robot2 = XLerobot(XLerobotConfig(port1=PORT, use_degrees=True))
        robot2.bus1.connect()
        bus = robot2.bus1
        bus.calibration = {**bus.calibration, **head_calibration}
        bus_calib2 = {k: v for k, v in robot2.calibration.items() if k in bus.motors}
        if bus_calib2:
            bus.calibration = {**bus.calibration, **bus_calib2}
            bus.write_calibration(bus_calib2)
        limits = compute_arm_limits(bus)
        model_arm, data_arm, base_fid, ee_fid = build_arm_model(limits)
        bus.disable_torque()

        pts_camera.append(centroid)
        pts_base.append(p_ee)
        sample_idx += 1
        print(f"  Recorded sample {sample_idx}.\n")

    bus.disconnect()

    pts_camera = np.array(pts_camera)
    pts_base = np.array(pts_base)

    print(f"\n{'='*60}")
    print(f"Solving with {len(pts_camera)} samples...")

    # --- Solve for T_base_camera ---
    T = solve_rigid_transform(pts_camera, pts_base)

    # Verify
    print(f"\nT_base_camera (4x4):")
    print(T)

    errors = []
    for i in range(len(pts_camera)):
        p_h = np.array([*pts_camera[i], 1.0])
        p_proj = (T @ p_h)[:3]
        err = np.linalg.norm(p_proj - pts_base[i])
        errors.append(err)
        print(f"\n  Sample {i+1}:")
        print(f"    Camera centroid: {pts_camera[i]}")
        print(f"    Projected Base_2: {p_proj}")
        print(f"    EE ground truth:  {pts_base[i]}")
        print(f"    Error: {err:.4f} m")

    mean_err = np.mean(errors)
    max_err = np.max(errors)
    print(f"\n  Mean error: {mean_err:.4f} m")
    print(f"  Max error:  {max_err:.4f} m")

    # --- Save ---
    calib_data = {
        "T_base_camera": T.tolist(),
        "num_samples": len(pts_camera),
        "mean_error_m": float(mean_err),
        "max_error_m": float(max_err),
    }
    CALIBRATION_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_OUT, "w") as f:
        json.dump(calib_data, f, indent=4)
    print(f"\nSaved T_base_camera to {CALIBRATION_OUT}")

    if mean_err < 0.03:
        print("Calibration looks good!")
    elif mean_err < 0.06:
        print("Calibration is OK. Consider adding more samples for better accuracy.")
    else:
        print("WARNING: High error. Check that object detection picks the right cluster.")


if __name__ == "__main__":
    main()
