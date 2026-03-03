#!/usr/bin/env python3
"""
Motor calibration module for SO-101 robot.

Usage as standalone (always recalibrates):
    python calibrate.py
    python calibrate.py --file calibration/my_calib.json

Usage as module (reuses existing calibration file if present):
    from calibrate import load_or_run_calibration
    load_or_run_calibration(bus)
"""

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
import json
import argparse
from pathlib import Path

CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"
DEFAULT_ARM_CALIBRATION_FILE = CALIBRATION_DIR / "arm_bus.json"
DEFAULT_HEAD_CALIBRATION_FILE = CALIBRATION_DIR / "head_bus.json"

ARM_BUS_PORT = "/dev/ttyACM0"   # bus0: both arms (IDs 1-12)
HEAD_BUS_PORT = "/dev/ttyACM1"  # bus1: head motors (IDs 1-2)

# Motor definitions for arm bus (bus0) — both arms
ARM_MOTOR_DEFS = {
    # Base_2 arm (IDs 1-6)
    "shoulder_pan_2":  Motor(1,  "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift_2": Motor(2,  "sts3215", MotorNormMode.DEGREES),
    "elbow_flex_2":    Motor(3,  "sts3215", MotorNormMode.DEGREES),
    "wrist_flex_2":    Motor(4,  "sts3215", MotorNormMode.DEGREES),
    "wrist_roll_2":    Motor(5,  "sts3215", MotorNormMode.DEGREES),
    "gripper_2":       Motor(6,  "sts3215", MotorNormMode.RANGE_0_100),
    # Base arm (IDs 7-12)
    "shoulder_pan":  Motor(7,  "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(8,  "sts3215", MotorNormMode.DEGREES),
    "elbow_flex":    Motor(9,  "sts3215", MotorNormMode.DEGREES),
    "wrist_flex":    Motor(10, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll":    Motor(11, "sts3215", MotorNormMode.DEGREES),
    "gripper":       Motor(12, "sts3215", MotorNormMode.RANGE_0_100),
}

# Motor definitions for head bus (bus1)
HEAD_MOTOR_DEFS = {
    "head_pan":  Motor(2, "sts3215", MotorNormMode.DEGREES),   # pan (ID 2)
    "head_tilt": Motor(1, "sts3215", MotorNormMode.DEGREES),   # tilt (ID 1)
}

# Legacy aliases for backwards compatibility
BUS_PORT = ARM_BUS_PORT
MOTOR_DEFS = {**ARM_MOTOR_DEFS, **HEAD_MOTOR_DEFS}


def load_calibration(bus: FeetechMotorsBus, filepath: Path) -> dict:
    """Load calibration from JSON and apply it to the bus.

    Returns the raw calibration dict.
    """
    with open(filepath) as f:
        calib_raw = json.load(f)
    bus.calibration = {
        name: MotorCalibration(**vals) for name, vals in calib_raw.items()
    }
    print(f"Loaded calibration from {filepath}")
    return calib_raw


def run_interactive_calibration(bus: FeetechMotorsBus, filepath: Path) -> dict:
    """Run the interactive two-step calibration and save to JSON.

    Steps:
        1. User moves all motors to middle -> set homing offsets
        2. User moves all motors through full range -> record min/max

    Returns the raw calibration dict.
    """
    motor_names = list(bus.motors.keys())
    bus.disable_torque(motor_names)

    input("\n>>> Move ALL motors to the MIDDLE of their range of motion, then press ENTER...")
    homing_offsets = bus.set_half_turn_homings(motor_names)
    print(f"Homing offsets set: {homing_offsets}")

    print("\n>>> Move ALL motors through their FULL range of motion.")
    input("    Move each joint to both extremes. Press ENTER when done...")
    range_mins, range_maxes = bus.record_ranges_of_motion(motor_names)
    print(f"Range mins: {range_mins}")
    print(f"Range maxes: {range_maxes}")

    calib_raw = {}
    for name in motor_names:
        motor = bus.motors[name]
        calib_raw[name] = {
            "id": motor.id,
            "drive_mode": 0,
            "homing_offset": homing_offsets[name],
            "range_min": range_mins[name],
            "range_max": range_maxes[name],
        }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(calib_raw, f, indent=4)
    print(f"Calibration saved to {filepath}")

    bus.calibration = {
        name: MotorCalibration(**vals) for name, vals in calib_raw.items()
    }
    return calib_raw


def load_or_run_calibration(
    bus: FeetechMotorsBus,
    filepath: Path = DEFAULT_ARM_CALIBRATION_FILE,
    force: bool = False,
) -> dict:
    """Load existing calibration or run interactive calibration if missing.

    Args:
        bus: Connected FeetechMotorsBus instance.
        filepath: Path to calibration JSON file.
        force: If True, re-run calibration even if file exists.

    Returns the raw calibration dict.
    """
    if filepath.exists() and not force:
        return load_calibration(bus, filepath)
    else:
        if force:
            print("Force recalibration requested.")
        else:
            print("No calibration file found. Running calibration...")
        return run_interactive_calibration(bus, filepath)


def main():
    parser = argparse.ArgumentParser(description="Calibrate SO-101 motors")
    parser.add_argument(
        "--bus", choices=["arm", "head", "all"], default="all",
        help="Which bus to calibrate (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recalibrate even if calibration file already exists",
    )
    args = parser.parse_args()

    buses = []
    try:
        if args.bus in ("arm", "all"):
            arm_bus = FeetechMotorsBus(port=ARM_BUS_PORT, motors=ARM_MOTOR_DEFS)
            arm_bus.connect()
            buses.append(("arm", arm_bus, DEFAULT_ARM_CALIBRATION_FILE))

        if args.bus in ("head", "all"):
            head_bus = FeetechMotorsBus(port=HEAD_BUS_PORT, motors=HEAD_MOTOR_DEFS)
            head_bus.connect()
            buses.append(("head", head_bus, DEFAULT_HEAD_CALIBRATION_FILE))

        for label, bus, calib_file in buses:
            print(f"\n=== Calibrating {label} bus ===")
            load_or_run_calibration(bus, filepath=calib_file, force=True)

            print(f"\nCalibrated positions ({label}):")
            positions = bus.sync_read("Present_Position", list(bus.motors.keys()))
            for name, val in positions.items():
                print(f"  {name}: {float(val):.2f}")
    finally:
        for _, bus, _ in buses:
            bus.disconnect()


if __name__ == "__main__":
    main()
