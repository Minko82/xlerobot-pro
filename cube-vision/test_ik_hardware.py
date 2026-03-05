"""Test bimanual IK on hardware with a hardcoded target (no vision)."""

from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
import numpy as np
from ik_solver import IK_SO101
from calibrate import (
    ARM_MOTOR_DEFS,
    ARM_BUS_PORT,
    DEFAULT_ARM_CALIBRATION_FILE,
    load_or_run_calibration,
)
import time

# ── Hardcoded target in Base frame (meters) ──
# Measure a point and set it here.
# Right Arm Test: [0.0, -0.30, 0.05] Left Arm Test: [-0.15, -0.20, 0.05]
# Due to crazy rotations, relative to camera: Y is backward, -Y is forward, X is right, -X is left, Z is up, negative Z is down. 
TARGET_BASE = [0.0, -0.20, 0.00]

# Set to "left", "right", or "auto" (auto picks closer arm)
ARM = "auto"

# ── IK target offset (same as control_single_bus.py) ──
IK_TARGET_OFFSET_X_M = -0.065
IK_TARGET_OFFSET_Y_M = 0.0
IK_TARGET_OFFSET_Z_M = 0.0

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

ARM_JOINT_KEYS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
]
ARM_JOINT_KEYS_2 = [
    "shoulder_pan_2", "shoulder_lift_2", "elbow_flex_2", "wrist_flex_2", "wrist_roll_2",
]
all_arm_motors = ARM_JOINT_KEYS + ["gripper"] + ARM_JOINT_KEYS_2 + ["gripper_2"]


def mjcf_to_motor(q_deg: np.ndarray) -> np.ndarray:
    out = q_deg.copy()
    out[0] = -out[0]
    out[1] = 90.0 - out[1]
    out[2] = out[2] - 90.0
    return out


def traj_to_goals(traj_rad, joint_keys):
    stack = np.stack(traj_rad)
    traj_deg = np.array([mjcf_to_motor(q * RAD2DEG) for q in stack])
    return [
        {joint: float(q_deg[i]) for i, joint in enumerate(joint_keys)}
        for q_deg in traj_deg
    ]


# ── Connect motors ──
arm_bus = FeetechMotorsBus(port=ARM_BUS_PORT, motors=ARM_MOTOR_DEFS)
arm_bus.connect()
load_or_run_calibration(arm_bus, filepath=DEFAULT_ARM_CALIBRATION_FILE)

arm_bus.disable_torque(all_arm_motors)
for name in all_arm_motors:
    arm_bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
    arm_bus.write("Torque_Limit", name, 500)
    arm_bus.write("Acceleration", name, 10)
    arm_bus.write("P_Coefficient", name, 8)
    arm_bus.write("I_Coefficient", name, 0)
    arm_bus.write("D_Coefficient", name, 32)
arm_bus.enable_torque(all_arm_motors)

# ── Solve IK ──
ik = IK_SO101()

offset = np.array([IK_TARGET_OFFSET_X_M, IK_TARGET_OFFSET_Y_M, IK_TARGET_OFFSET_Z_M])
target_left = np.array(TARGET_BASE) + offset

# For "auto" mode we need the same point in Base_2 frame.
# Approximate: convert left-arm target to world, then to Base_2 frame.
target_world = ik.base_to_world(target_left)
target_right = ik._base2_R.T @ (target_world - ik._base2_t)

if ARM == "auto":
    chosen = ik.choose_arm(target_left, target_right)
elif ARM in ("left", "right"):
    chosen = ARM
else:
    raise ValueError(f"ARM must be 'left', 'right', or 'auto', got '{ARM}'")

active_target = target_left if chosen == "left" else target_right
joint_keys = ARM_JOINT_KEYS if chosen == "left" else ARM_JOINT_KEYS_2
gripper = "gripper" if chosen == "left" else "gripper_2"

print(f"Arm: {chosen}")
print(f"Target (arm base frame): {active_target}")

traj = ik.generate_ik_bimanual(active_target.tolist(), arm=chosen)
if not traj:
    print("IK failed — target may be out of reach.")
    arm_bus.disconnect()
    raise SystemExit(1)

print(f"IK solved: {len(traj)} steps")
print(f"Final joints (deg): {np.rad2deg(traj[-1])}")

# ── Send to hardware ──
goals = traj_to_goals(traj, joint_keys)
dt = 0.01

print(f"Sending {len(goals)} waypoints...")
for goal in goals:
    goal[gripper] = 100.0
    arm_bus.sync_write("Goal_Position", goal)
    time.sleep(dt)

final_goal = goals[-1].copy()
final_goal[gripper] = 100.0
arm_bus.sync_write("Goal_Position", final_goal)
print("Holding final position for 5s...")
time.sleep(5.0)

actual = arm_bus.sync_read("Present_Position", joint_keys)
print("Actual vs goal (deg):")
for name in joint_keys:
    print(f"  {name}: {float(actual[name]):.2f}  (goal: {final_goal[name]:.2f})")

print("Done.")
arm_bus.disconnect()
