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
from visualize_color_detect import visualize as visualize_color_detect
import time

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# Hardcoded IK target offsets in Base frame (meters).
IK_TARGET_OFFSET_X_M = -0.12
IK_TARGET_OFFSET_Y_M = 0.06
IK_TARGET_OFFSET_Z_M = 0.0

# Number of grab-lift-drop cycles
NUM_CYCLES = 3

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

joint_values = {
    "head_pan_joint": head_pan_deg * DEG2RAD,
    "head_tilt_joint": head_tilt_deg * DEG2RAD,
}

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

ik_solve = IK_SO101()
dt = 0.01


def mjcf_to_motor(q_deg: np.ndarray) -> np.ndarray:
    """Convert MJCF joint angles (degrees) to motor convention (degrees)."""
    out = q_deg.copy()
    out[0] = -out[0]
    out[1] = 90.0 - out[1]
    out[2] = out[2] - 90.0
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


# Save starting positions for both arms
left_start_pos = arm_bus.sync_read("Present_Position", ARM_JOINT_KEYS + ["gripper"])
left_start_goal = {name: float(left_start_pos[name]) for name in ARM_JOINT_KEYS + ["gripper"]}

right_start_pos = arm_bus.sync_read("Present_Position", ARM_JOINT_KEYS_2 + ["gripper_2"])
right_start_goal = {name: float(right_start_pos[name]) for name in ARM_JOINT_KEYS_2 + ["gripper_2"]}

home_goal = {}
home_goal.update(left_start_goal)
home_goal.update(right_start_goal)
print(f"Saved home positions for both arms")

for cycle in range(NUM_CYCLES):
    print(f"\n{'='*50}")
    print(f"  CYCLE {cycle + 1}/{NUM_CYCLES}")
    print(f"{'='*50}")

    # --- Vision: capture and detect ---
    print("\nCapturing image...")
    capture()
    centroid = detect_object(color="red")
    print(f"Camera centroid (optical frame): {centroid}")
    visualize_color_detect(
        color="red",
        head_pan_deg=head_pan_deg,
        head_tilt_deg=head_tilt_deg,
        out_name=f"color_detect_vis_cycle_{cycle + 1:02d}.png",
    )

    # Transform to both arm frames
    arm_frame_x, arm_frame_y, arm_frame_z = camera_xyz_to_base_xyz(
        centroid[0], centroid[1], centroid[2], joint_values,
    )
    arm2_frame_x, arm2_frame_y, arm2_frame_z = camera_xyz_to_base2_xyz(
        centroid[0], centroid[1], centroid[2], joint_values,
    )

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

    # Choose closest arm
    chosen_arm = ik_solve.choose_arm(np.array(target_base), np.array(target_base2))

    if chosen_arm == "left":
        active_target = target_base
        active_joint_keys = ARM_JOINT_KEYS
        active_gripper = "gripper"
        active_start = left_start_goal
    else:
        active_target = target_base2
        active_joint_keys = ARM_JOINT_KEYS_2
        active_gripper = "gripper_2"
        active_start = right_start_goal

    print(f"Target ({chosen_arm} arm): [{active_target[0]:.4f}, {active_target[1]:.4f}, {active_target[2]:.4f}]")

    # --- IK solve ---
    print(f"Running IK for {chosen_arm} arm...")
    trajectory_rad = ik_solve.generate_ik_bimanual(active_target, arm=chosen_arm)
    if not trajectory_rad:
        print(f"IK failed — skipping cycle {cycle + 1}")
        continue

    goals = traj_to_goals(trajectory_rad, active_joint_keys)
    print(f"IK succeeded: {len(goals)} waypoints")

    # --- Step 1: Open gripper ---
    print("Opening gripper...")
    open_goal = {name: float(active_start[name]) for name in active_joint_keys}
    open_goal[active_gripper] = 100.0
    arm_bus.sync_write("Goal_Position", open_goal)
    time.sleep(0.5)

    # --- Step 2: Move to target with gripper open ---
    print(f"Moving to cube...")
    for goal in goals:
        goal[active_gripper] = 100.0
        arm_bus.sync_write("Goal_Position", goal)
        time.sleep(dt)

    final_goal = goals[-1].copy()
    final_goal[active_gripper] = 100.0
    arm_bus.sync_write("Goal_Position", final_goal)
    print("Holding at target for 2 seconds...")
    time.sleep(2.0)

    # --- Step 3: Close gripper ---
    print("Closing gripper...")
    close_goal = dict(final_goal)
    for grip in range(100, 5, -5):
        close_goal[active_gripper] = float(grip)
        arm_bus.sync_write("Goal_Position", close_goal)
        time.sleep(0.05)
    time.sleep(0.5)

    # --- Step 4: Lift 15cm ---
    lift_target = list(active_target)
    lift_target[2] += 0.15
    print(f"Lifting to: [{lift_target[0]:.4f}, {lift_target[1]:.4f}, {lift_target[2]:.4f}]")

    lift_traj_rad = ik_solve.generate_ik_bimanual(
        lift_target, arm=chosen_arm, seed_q_rad=trajectory_rad[-1],
    )
    if lift_traj_rad:
        lift_goals = traj_to_goals(lift_traj_rad, active_joint_keys)
        for goal in lift_goals:
            goal[active_gripper] = close_goal[active_gripper]
            arm_bus.sync_write("Goal_Position", goal)
            time.sleep(dt)
        arm_bus.sync_write("Goal_Position", lift_goals[-1])
        print("Lifted. Holding for 2 seconds...")
    else:
        print("Lift IK failed, holding current position...")
    time.sleep(2.0)

    # --- Step 5: Open gripper to drop ---
    print("Dropping cube...")
    drop_goal = (lift_goals[-1].copy() if lift_traj_rad else dict(close_goal))
    for grip in range(5, 105, 5):
        drop_goal[active_gripper] = float(grip)
        arm_bus.sync_write("Goal_Position", drop_goal)
        time.sleep(0.05)
    time.sleep(0.5)

    # --- Step 6: Return to start ---
    print("Returning to start position...")
    arm_bus.sync_write("Goal_Position", active_start)
    time.sleep(3.0)

    print(f"Cycle {cycle + 1}/{NUM_CYCLES} complete.")

print("\nAll cycles done.")
arm_bus.disconnect()
head_bus.disconnect()
