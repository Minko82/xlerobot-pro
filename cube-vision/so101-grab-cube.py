from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
import numpy as np
import json
from pathlib import Path
from ik_solver import IK_SO101
import time

PORT = "/dev/ttyACM1"
CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration" / "single_bus.json"
norm_mode_body = MotorNormMode.DEGREES

# Single bus: head (IDs 1-2) + right arm (IDs 7-12)
bus = FeetechMotorsBus(
    port=PORT,
    motors={
        "head_motor_1":  Motor(1,  "sts3215", norm_mode_body),
        "head_motor_2":  Motor(2,  "sts3215", norm_mode_body),
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
if not CALIBRATION_FILE.exists():
    raise FileNotFoundError(
        f"Calibration not found: {CALIBRATION_FILE}\n"
        "Run calibration first."
    )
with open(CALIBRATION_FILE) as f:
    calib_raw = json.load(f)
bus.calibration = {
    name: MotorCalibration(**vals) for name, vals in calib_raw.items()
}

arm_motors = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def apply_limits(bus, motors, torque: int, acceleration: int, p: int, i: int, d: int):
    print("Applying motor limits:")
    print(f"    Torque_Limit = {torque} / 1000")
    print(f"    Acceleration = {acceleration} / 254")
    print(f"    P={p}, I={i}, D={d}\n")

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

dt = 0.01


def mjcf_to_motor(q_deg: np.ndarray) -> np.ndarray:
    """Convert MJCF joint angles (degrees) to motor convention (degrees).

    Joint order: Rotation_L, Pitch_L, Elbow_L, Wrist_Pitch_L, Wrist_Roll_L
    """
    out = q_deg.copy()
    out[1] = 90.0 - out[1]   # Pitch_L -> shoulder_lift
    out[2] = out[2] - 90.0   # Elbow_L -> elbow_flex
    return out


trajectory_rad = ik_solve.generate_ik([0.35, 0.0, 0.0], [-0.05, -0.01, -0.0808])
RAD2DEG = 180.0 / np.pi
traj_rad_stack = np.stack(trajectory_rad)
trajectory = np.array([mjcf_to_motor(q * RAD2DEG) for q in traj_rad_stack])

ARM_JOINT_KEYS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

for step in trajectory:
    print(step)


def traj_to_goal(q_deg: np.ndarray) -> dict:
    assert q_deg.shape[0] == len(ARM_JOINT_KEYS)
    return {joint: float(q_deg[i]) for i, joint in enumerate(ARM_JOINT_KEYS)}


goals = [traj_to_goal(q_deg) for q_deg in trajectory]

for goal in goals:
    goal["gripper"] = 100.0
    bus.sync_write("Goal_Position", goal)
    time.sleep(dt)

# Close gripper
hold_goal = {k: v for k, v in goals[-1].items() if k != "gripper"}
for grip in range(100, 5, -5):
    goal = dict(hold_goal)
    goal["gripper"] = float(grip)
    bus.sync_write("Goal_Position", goal)
    time.sleep(0.05)

# Movement: lift up
trajectory_rad = ik_solve.generate_ik([0.30, 0.0, 0.10], [-0.05, -0.01, -0.0808])
traj_rad_stack = np.stack(trajectory_rad)
trajectory = np.array([mjcf_to_motor(q * RAD2DEG) for q in traj_rad_stack])

goals = [traj_to_goal(q_deg) for q_deg in trajectory]

for goal in goals:
    bus.sync_write("Goal_Position", goal)
    time.sleep(dt)

# Movement 2: move to drop position
trajectory_rad = ik_solve.generate_ik([0.10, 0.0, 0.0], [-0.05, -0.01, -0.0808])
traj_rad_stack = np.stack(trajectory_rad)
trajectory = np.array([mjcf_to_motor(q * RAD2DEG) for q in traj_rad_stack])

goals = [traj_to_goal(q_deg) for q_deg in trajectory]

for goal in goals:
    bus.sync_write("Goal_Position", goal)
    time.sleep(dt)

# Open gripper to release
hold_goal = {k: v for k, v in goals[-1].items() if k != "gripper"}
for grip in range(5, 100, 5):
    goal = dict(hold_goal)
    goal["gripper"] = float(grip)
    bus.sync_write("Goal_Position", goal)
    time.sleep(0.05)

bus.disconnect()
