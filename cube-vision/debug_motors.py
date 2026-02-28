"""Debug script: measure neutral offsets and joint directions."""
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
import numpy as np
import json
from pathlib import Path

BUS_PORT = "/dev/ttyACM0"
CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration" / "single_bus.json"

norm_mode_body = MotorNormMode.DEGREES

bus = FeetechMotorsBus(
    port=BUS_PORT,
    motors={
        "head_motor_1": Motor(1, "sts3215", norm_mode_body),
        "head_motor_2": Motor(2, "sts3215", norm_mode_body),
        "shoulder_pan":  Motor(7,  "sts3215", norm_mode_body),
        "shoulder_lift": Motor(8,  "sts3215", norm_mode_body),
        "elbow_flex":    Motor(9,  "sts3215", norm_mode_body),
        "wrist_flex":    Motor(10, "sts3215", norm_mode_body),
        "wrist_roll":    Motor(11, "sts3215", norm_mode_body),
        "gripper":       Motor(12, "sts3215", MotorNormMode.RANGE_0_100),
    },
)
bus.connect()

# Load calibration
with open(CALIBRATION_FILE) as f:
    calib_raw = json.load(f)
bus.calibration = {
    name: MotorCalibration(**vals) for name, vals in calib_raw.items()
}

arm_joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

print("=== Joint direction test ===")
print("For each joint, move it in the POSITIVE MJCF direction:")
print("  - shoulder_pan:  rotate the base counter-clockwise (top view)")
print("  - shoulder_lift: tilt the upper arm backward (away from front)")
print("  - elbow_flex:    bend the elbow (fold the forearm up)")
print("  - wrist_flex:    tilt the wrist up")
print("  - wrist_roll:    roll the wrist counter-clockwise")
print()

input(">>> First, put arm in NEUTRAL (straight up). Press ENTER...")
neutral = bus.sync_read("Present_Position", arm_joints)
neutral_vals = {name: float(neutral[name]) for name in arm_joints}
print("Neutral readings:")
for name in arm_joints:
    print(f"  {name:20s} = {neutral_vals[name]:8.2f} deg")

print()
for joint in arm_joints:
    input(f">>> Move ONLY {joint} in the POSITIVE direction, then press ENTER...")
    pos = bus.sync_read("Present_Position", arm_joints)
    delta = float(pos[joint]) - neutral_vals[joint]
    direction = "SAME" if delta > 0 else "REVERSED"
    print(f"  {joint}: moved {delta:+.2f} deg -> direction is {direction}")
    print()

print("\n=== Summary: neutral offsets ===")
for name in arm_joints:
    print(f"  {name:20s} = {neutral_vals[name]:8.2f} deg")

bus.disconnect()
