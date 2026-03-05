from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors import MotorCalibration, MotorNormMode
import numpy as np
from ik_solver import IK_SO101
from color_detect import detect_object
from frame_transform.frame_transform import camera_xyz_to_base_xyz, camera_xyz_to_base2_xyz
from realsense_capture import capture
from calibrate import (
    ARM_MOTOR_DEFS, HEAD_MOTOR_DEFS,
    ARM_BUS_PORT, HEAD_BUS_PORT,
    DEFAULT_ARM_CALIBRATION_FILE, DEFAULT_HEAD_CALIBRATION_FILE,
    load_or_run_calibration,
)
from visualize_ik import save_ik_plot
import time

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# Hardcoded IK target offsets in Base frame (meters).
# Tune these to compensate end-effector placement error without changing vision transforms.
IK_TARGET_OFFSET_X_M = -0.065
IK_TARGET_OFFSET_Y_M = 0.0
IK_TARGET_OFFSET_Z_M = 0.0

# bus0: both arms (IDs 1-6 Base_2, IDs 7-12 Base)
arm_bus = FeetechMotorsBus(port=ARM_BUS_PORT, motors=ARM_MOTOR_DEFS)
arm_bus.connect()
load_or_run_calibration(arm_bus, filepath=DEFAULT_ARM_CALIBRATION_FILE)

# bus1: head motors (pan ID 2, tilt ID 1)
head_bus = FeetechMotorsBus(port=HEAD_BUS_PORT, motors=HEAD_MOTOR_DEFS)
head_bus.connect()
load_or_run_calibration(head_bus, filepath=DEFAULT_HEAD_CALIBRATION_FILE)

# Read head positions (calibrated)
head_pos = head_bus.sync_read("Present_Position", ["head_pan", "head_tilt"])
head_pan_deg = float(head_pos["head_pan"])
head_tilt_deg = float(head_pos["head_tilt"])
print(f"Head motors (deg): pan={head_pan_deg:.2f}, tilt={head_tilt_deg:.2f}")

# Left arm (Base) motor names
ARM_JOINT_KEYS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# Right arm (Base_2) motor names
ARM_JOINT_KEYS_2 = [
    "shoulder_pan_2",
    "shoulder_lift_2",
    "elbow_flex_2",
    "wrist_flex_2",
    "wrist_roll_2",
]

# All arm motors (for applying limits to both arms)
all_arm_motors = ARM_JOINT_KEYS + ["gripper"] + ARM_JOINT_KEYS_2 + ["gripper_2"]


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


apply_limits(arm_bus, all_arm_motors, 500, 10, 8, 0, 32)

# Capture fresh RGBD frames from the RealSense
capture()

# Detect object by color (change color= to "red", "green", or "blue" as needed)
centroid = detect_object(color="red")
print(f"Camera centroid (optical frame): {centroid}")

joint_values = {
    "head_pan_joint": head_pan_deg * DEG2RAD,
    "head_tilt_joint": head_tilt_deg * DEG2RAD,
}

# Transform to both arm frames
arm_frame_x, arm_frame_y, arm_frame_z = camera_xyz_to_base_xyz(
    centroid[0], centroid[1], centroid[2], joint_values,
)
arm2_frame_x, arm2_frame_y, arm2_frame_z = camera_xyz_to_base2_xyz(
    centroid[0], centroid[1], centroid[2], joint_values,
)
print(f"Transformed to Base frame (left arm):  [{arm_frame_x:.4f}, {arm_frame_y:.4f}, {arm_frame_z:.4f}]")
print(f"Transformed to Base_2 frame (right arm): [{arm2_frame_x:.4f}, {arm2_frame_y:.4f}, {arm2_frame_z:.4f}]")

ik_solve = IK_SO101()

# Apply IK offsets
target_base = [
    arm_frame_x + IK_TARGET_OFFSET_X_M,
    arm_frame_y + IK_TARGET_OFFSET_Y_M,
    arm_frame_z + IK_TARGET_OFFSET_Z_M,
]
target_base2 = [
    arm2_frame_x + IK_TARGET_OFFSET_X_M,
    arm2_frame_y + IK_TARGET_OFFSET_Y_M,
    arm2_frame_z + IK_TARGET_OFFSET_Z_M,
]

# Choose which arm to use
chosen_arm = ik_solve.choose_arm(
    np.array(target_base), np.array(target_base2),
)

# Select the right target for the chosen arm
if chosen_arm == "left":
    active_target = target_base
    active_joint_keys = ARM_JOINT_KEYS
    active_gripper = "gripper"
else:
    active_target = target_base2
    active_joint_keys = ARM_JOINT_KEYS_2
    active_gripper = "gripper_2"

print(
    "IK target offsets (m): "
    f"[{IK_TARGET_OFFSET_X_M:.4f}, {IK_TARGET_OFFSET_Y_M:.4f}, {IK_TARGET_OFFSET_Z_M:.4f}]"
)
print(
    f"IK target ({chosen_arm} arm, Base{'_2' if chosen_arm == 'right' else ''} frame, offset): "
    f"[{active_target[0]:.4f}, {active_target[1]:.4f}, {active_target[2]:.4f}]"
)

# Visualize the IK target
save_ik_plot(
    base_pos=ik_solve._base_t if chosen_arm == "left" else ik_solve._base2_t,
    ik_target_base=np.array(active_target),
    camera_centroid_cam=np.array(centroid),
)

dt = 0.01


def mjcf_to_motor(q_deg: np.ndarray) -> np.ndarray:
    """Convert MJCF joint angles (degrees) to motor convention (degrees).

    Joint order: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll
    Same conversion applies to both arms (identical axis conventions in XML).
    """
    out = q_deg.copy()
    out[0] = -out[0]          # Rotation: MJCF positive = left, motor positive = right
    out[1] = 90.0 - out[1]   # Pitch -> shoulder_lift
    out[2] = out[2] - 90.0   # Elbow -> elbow_flex
    return out


def traj_to_goals(traj_rad: list[np.ndarray], joint_keys: list[str]) -> list[dict]:
    """Convert a list of joint configs (radians) to motor goal dicts (degrees)."""
    stack = np.stack(traj_rad)
    traj_deg = np.array([mjcf_to_motor(q * RAD2DEG) for q in stack])
    goals = []
    for q_deg in traj_deg:
        assert q_deg.shape[0] == len(joint_keys)
        goals.append({joint: float(q_deg[i]) for i, joint in enumerate(joint_keys)})
    return goals


print(f"Running bimanual IK solver (active arm: {chosen_arm})...")
trajectory_rad = ik_solve.generate_ik_bimanual(active_target, arm=chosen_arm)
if not trajectory_rad:
    print("IK failed — no trajectory returned. Target may be out of reach.")
    print(f"  Target distance from base: {np.linalg.norm(active_target):.4f} m")
    arm_bus.disconnect()
    head_bus.disconnect()
    raise SystemExit(1)

print(f"IK succeeded: {len(trajectory_rad)} steps")
print(f"  Final joint config (rad): {trajectory_rad[-1]}")
print(f"  Final joint config (deg): {np.rad2deg(trajectory_rad[-1])}")

# Convert to motor goals for the active arm
goals = traj_to_goals(trajectory_rad, active_joint_keys)
print(f"Sending {len(goals)} waypoints to {chosen_arm} arm motors...")
print(f"  First goal: {goals[0]}")
print(f"  Last goal:  {goals[-1]}")

for i, goal in enumerate(goals):
    goal[active_gripper] = 100.0
    arm_bus.sync_write("Goal_Position", goal)
    time.sleep(dt)

# Re-send final goal and hold so motors have time to physically reach it
final_goal = goals[-1].copy()
final_goal[active_gripper] = 100.0
arm_bus.sync_write("Goal_Position", final_goal)
print("Trajectory sent. Holding final position for 5 seconds...")
time.sleep(5.0)

# Read back actual positions to verify movement
actual = arm_bus.sync_read("Present_Position", active_joint_keys)
print("Actual motor positions (deg):")
for name in active_joint_keys:
    print(f"  {name}: {float(actual[name]):.2f}  (goal: {final_goal[name]:.2f})")
print("Done.")
arm_bus.disconnect()
head_bus.disconnect()
