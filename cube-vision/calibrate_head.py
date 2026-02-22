"""Calibrate head motors (pan + tilt) and save to JSON.

Usage:
    1. Run this script
    2. Move both head motors to the MIDDLE of their range of motion, press ENTER
    3. Move both head motors through their FULL range of motion, press ENTER when done
    4. Calibration is saved to calibration/head.json
"""

import json
from pathlib import Path
from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
from lerobot.motors.feetech import OperatingMode

SERIAL_PORT = "/dev/ttyACM0"
CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"
CALIBRATION_FILE = CALIBRATION_DIR / "head.json"

# Connect
config = XLerobotConfig(port1=SERIAL_PORT, use_degrees=True)
robot = XLerobot(config)
robot.bus1.connect()

head_motors = robot.head_motors
print(f"Head motors found: {head_motors}")

# Disable torque so motors can be moved by hand
robot.bus1.disable_torque(head_motors)
for name in head_motors:
    robot.bus1.write("Operating_Mode", name, OperatingMode.POSITION.value)

# Read current raw positions
raw = robot.bus1.sync_read("Present_Position", head_motors, normalize=False)
print(f"Current raw positions: {raw}")

# Step 1: Set homing offsets (move to middle first)
input("\n>>> Move both head motors to the MIDDLE of their range of motion, then press ENTER...")
homing_offsets = robot.bus1.set_half_turn_homings(head_motors)
print(f"Homing offsets set: {homing_offsets}")

# Verify: after homing, read raw positions (should be near 2047 = half-turn)
calibrated_raw = robot.bus1.sync_read("Present_Position", head_motors, normalize=False)
print(f"Raw positions after homing (should be near 2047): {calibrated_raw}")

# Step 2: Record range of motion
print("\n>>> Move both head motors through their FULL range of motion.")
print("    Move each joint to both extremes. Press ENTER when done...")
range_mins, range_maxes = robot.bus1.record_ranges_of_motion(head_motors)
print(f"Range mins: {range_mins}")
print(f"Range maxes: {range_maxes}")

# Build calibration dict
calibration = {}
for name in head_motors:
    motor = robot.bus1.motors[name]
    calibration[name] = {
        "id": motor.id,
        "drive_mode": 0,
        "homing_offset": homing_offsets[name],
        "range_min": range_mins[name],
        "range_max": range_maxes[name],
    }

# Save
CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
with open(CALIBRATION_FILE, "w") as f:
    json.dump(calibration, f, indent=4)

print(f"\nCalibration saved to {CALIBRATION_FILE}")
print(json.dumps(calibration, indent=4))

robot.bus1.disconnect()
