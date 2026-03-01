#!/usr/bin/env python3
"""Read and print current motor positions (degrees).
Disable torque first so you can manually position the arm."""

from lerobot.motors.feetech import FeetechMotorsBus
from calibrate import MOTOR_DEFS, BUS_PORT, load_or_run_calibration

bus = FeetechMotorsBus(port=BUS_PORT, motors=MOTOR_DEFS)
bus.connect()
load_or_run_calibration(bus)

arm_motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
bus.disable_torque(arm_motors)

input("Move the gripper directly above the cube, then press ENTER...")

positions = bus.sync_read("Present_Position", list(bus.motors.keys()))
print("\nMotor positions (degrees):")
for name, val in positions.items():
    print(f"  {name:20s}: {float(val):.2f}")

bus.disconnect()
