"""Move the arm straight forward to 35cm and hold."""

from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
import numpy as np
import json
from pathlib import Path
from ik_solver import IK_SO101
import time

PORT = "/dev/ttyACM0"
CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration" / "single_bus.json"
norm_mode_body = MotorNormMode.DEGREES

bus = FeetechMotorsBus(
    port=PORT,
    motors={
        "shoulder_pan":  Motor(7,  "sts3215", norm_mode_body),
        "shoulder_lift": Motor(8,  "sts3215", norm_mode_body),
        "elbow_flex":    Motor(9,  "sts3215", norm_mode_body),
        "wrist_flex":    Motor(10, "sts3215", norm_mode_body),
        "wrist_roll":    Motor(11, "sts3215", norm_mode_body),
        "gripper":       Motor(12, "sts3215", MotorNormMode.RANGE_0_100),
    },
)
bus.connect()

if not CALIBRATION_FILE.exists():
    raise FileNotFoundError(f"Calibration not found: {CALIBRATION_FILE}\nRun calibration first.")
with open(CALIBRATION_FILE) as f:
    calib_raw = json.load(f)
bus.calibration = {
    name: MotorCalibration(**vals) for name, vals in calib_raw.items()
    if name in bus.motors
}

arm_motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def apply_limits(bus, motors, torque, acceleration, p, i, d):
    bus.disable_torque(motors)
    for name in motors:
        bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
        bus.write("Torque_Limit", name, torque)
        bus.write("Acceleration", name, acceleration)
        bus.write("P_Coefficient", name, p)
        bus.write("I_Coefficient", name, i)
        bus.write("D_Coefficient", name, d)
    bus.enable_torque(motors)


apply_limits(bus, arm_motors, 200, 10, 8, 0, 32)

ik_solve = IK_SO101()

# Target: 35cm forward from the robot base
# Base frame: +Y is forward, -X is left, +Z is up
target_base = [0.0, 0.35, 0.0]
print(f"IK target (Base frame): {target_base}")

trajectory_rad = ik_solve.generate_ik(target_base, [0, 0, 0])
print(f"IK trajectory: {len(trajectory_rad)} steps")

RAD2DEG = 180.0 / np.pi


def mjcf_to_motor(q_deg):
    out = q_deg.copy()
    out[1] = 90.0 - out[1]   # Pitch_L -> shoulder_lift
    out[2] = out[2] - 90.0   # Elbow_L -> elbow_flex
    return out


traj_rad_stack = np.stack(trajectory_rad)
trajectory = np.array([mjcf_to_motor(q * RAD2DEG) for q in traj_rad_stack])

ARM_JOINT_KEYS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def traj_to_goal(q_deg):
    return {joint: float(q_deg[i]) for i, joint in enumerate(ARM_JOINT_KEYS)}


goals = [traj_to_goal(q_deg) for q_deg in trajectory]

print("Moving to target...")
for goal in goals:
    goal["gripper"] = 50.0
    bus.sync_write("Goal_Position", goal)
    time.sleep(0.01)

final = goals[-1]
print(f"Final joint angles: {final}")
print("Holding position. Press Ctrl+C to release.")

try:
    while True:
        bus.sync_write("Goal_Position", final)
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nReleasing...")

bus.disconnect()
