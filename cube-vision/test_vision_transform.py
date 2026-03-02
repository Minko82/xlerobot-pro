#!/usr/bin/env python3
"""Vision transform isolation test.

Place an object at a KNOWN, tape-measured position relative to the arm base,
then capture + detect + transform and compare the computed position to the
measured ground truth.

Usage:
    # Interactive — prompts for measured position:
    python test_vision_transform.py

    # Supply ground truth on the command line (meters, Base frame):
    python test_vision_transform.py --truth 0.05 0.25 0.01

    # Use a different color:
    python test_vision_transform.py --color blue --truth 0.0 0.30 0.0

    # Skip capture (reuse last saved frames):
    python test_vision_transform.py --skip-capture --truth 0.0 0.30 0.0
"""

import argparse
import numpy as np
from pathlib import Path

from calibrate import MOTOR_DEFS, BUS_PORT, load_or_run_calibration
from lerobot.motors.feetech import FeetechMotorsBus
from frame_transform.frame_transform import camera_xyz_to_base_xyz
from color_detect import detect_object
from realsense_capture import capture

DEG2RAD = np.pi / 180.0
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def read_head_angles(bus):
    """Read calibrated head pan/tilt in degrees."""
    pos = bus.sync_read("Present_Position", ["head_motor_1", "head_motor_2"])
    pan_deg = float(pos["head_motor_1"])
    tilt_deg = float(pos["head_motor_2"])
    return pan_deg, tilt_deg


def run_test(color: str, ground_truth: np.ndarray | None, skip_capture: bool):
    # ── Hardware setup ──────────────────────────────────────────────────
    bus = FeetechMotorsBus(port=BUS_PORT, motors=MOTOR_DEFS)
    bus.connect()
    load_or_run_calibration(bus)

    head_pan_deg, head_tilt_deg = read_head_angles(bus)
    print(f"\n=== Head motors (calibrated deg) ===")
    print(f"  pan  = {head_pan_deg:.2f}°")
    print(f"  tilt = {head_tilt_deg:.2f}°")

    # ── Capture ─────────────────────────────────────────────────────────
    if not skip_capture:
        print("\n=== Capturing RGBD from RealSense... ===")
        capture()
    else:
        print("\n=== Skipping capture (reusing saved frames) ===")

    # ── Detect ──────────────────────────────────────────────────────────
    print(f"\n=== Detecting '{color}' object... ===")
    centroid_cam = detect_object(color=color)
    print(f"  Camera centroid (optical): [{centroid_cam[0]:.4f}, {centroid_cam[1]:.4f}, {centroid_cam[2]:.4f}]")

    # ── Transform ───────────────────────────────────────────────────────
    joint_values = {
        "head_pan_joint":  head_pan_deg * DEG2RAD,
        "head_tilt_joint": head_tilt_deg * DEG2RAD,
    }
    bx, by, bz = camera_xyz_to_base_xyz(
        centroid_cam[0], centroid_cam[1], centroid_cam[2], joint_values,
    )
    computed = np.array([bx, by, bz])

    print(f"\n=== Transformed to Base frame ===")
    print(f"  Computed: [{bx:.4f}, {by:.4f}, {bz:.4f}] m")

    # ── Compare ─────────────────────────────────────────────────────────
    if ground_truth is None:
        print(
            "\n>>> Enter the tape-measured position in the Base frame (meters)."
            "\n    Base convention: +X = left, -Y = forward, +Z = up"
            "\n    (relative to the arm base mounting point)"
        )
        raw = input("    x y z: ").strip().split()
        if len(raw) != 3:
            print("Expected 3 values. Skipping comparison.")
            bus.disconnect()
            return
        ground_truth = np.array([float(v) for v in raw])

    error = computed - ground_truth
    dist = np.linalg.norm(error)

    print(f"\n{'='*52}")
    print(f"  Ground truth (tape): [{ground_truth[0]:.4f}, {ground_truth[1]:.4f}, {ground_truth[2]:.4f}] m")
    print(f"  Computed (vision):   [{computed[0]:.4f}, {computed[1]:.4f}, {computed[2]:.4f}] m")
    print(f"  Error (per axis):    [{error[0]:.4f}, {error[1]:.4f}, {error[2]:.4f}] m")
    print(f"  Euclidean error:     {dist:.4f} m  ({dist*100:.2f} cm)")
    print(f"{'='*52}")

    if dist < 0.01:
        print("  PASS — error < 1 cm")
    elif dist < 0.03:
        print("  MARGINAL — error 1-3 cm (may need offset tuning)")
    else:
        print("  FAIL — error > 3 cm (investigate transform chain)")

    # ── Save results ────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "vision_transform_test.txt"
    with open(report_path, "w") as f:
        f.write(f"head_pan_deg:  {head_pan_deg:.2f}\n")
        f.write(f"head_tilt_deg: {head_tilt_deg:.2f}\n")
        f.write(f"camera_centroid_optical: {centroid_cam.tolist()}\n")
        f.write(f"computed_base: {computed.tolist()}\n")
        f.write(f"ground_truth_base: {ground_truth.tolist()}\n")
        f.write(f"error_per_axis: {error.tolist()}\n")
        f.write(f"euclidean_error_m: {dist:.6f}\n")
    print(f"\nResults saved to {report_path}")

    bus.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vision transform isolation test — compare detected vs measured position."
    )
    parser.add_argument("--color", default="red", choices=["red", "green", "blue"],
                        help="Object color to detect (default: red)")
    parser.add_argument("--truth", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="Ground truth position in Base frame (meters)")
    parser.add_argument("--skip-capture", action="store_true",
                        help="Skip RealSense capture, reuse saved frames")
    args = parser.parse_args()

    ground_truth = np.array(args.truth) if args.truth else None
    run_test(color=args.color, ground_truth=ground_truth, skip_capture=args.skip_capture)
