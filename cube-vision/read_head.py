"""Read calibrated head motor positions. Use to find URDF zero offsets."""

import json
from pathlib import Path
from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
from lerobot.motors.motors_bus import MotorCalibration

SERIAL_PORT = "/dev/ttyACM0"
HEAD_CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration" / "head.json"

config = XLerobotConfig(port1=SERIAL_PORT, use_degrees=True)
robot = XLerobot(config)
robot.bus1.connect()

with open(HEAD_CALIBRATION_FILE) as f:
    head_calib_raw = json.load(f)
robot.bus1.calibration = {
    name: MotorCalibration(**vals) for name, vals in head_calib_raw.items()
}

pos = robot.bus1.sync_read("Present_Position", robot.head_motors)
pan = float(pos["head_motor_1"])
tilt = float(pos["head_motor_2"])
print(f"pan={pan:.2f}°, tilt={tilt:.2f}°")
print(f"\nThese values are the URDF zero offsets.")
print(f"Add these to frame_transform._head_motor_to_urdf to correct the transform.")

robot.bus1.disconnect()
