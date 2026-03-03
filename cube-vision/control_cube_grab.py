from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors import MotorCalibration, MotorNormMode
import numpy as np
from ik_solver import IK_SO101
from color_detect import detect_object
from frame_transform.frame_transform import camera_xyz_to_base_xyz
from realsense_capture import capture
from calibrate import MOTOR_DEFS, BUS_PORT, load_or_run_calibration
from visualize_ik import save_ik_plot
import time

DEG2RAD = np.pi / 180.0

# Hardcoded IK target offsets in Base frame (meters).
# Tune these to compensate end-effector placement error without changing vision transforms.
IK_TARGET_OFFSET_X_M = -0.05
IK_TARGET_OFFSET_Y_M = 0.0
IK_TARGET_OFFSET_Z_M = 0.06

# Approach offset: how far in front of / above the cube to stop before closing the gripper.
# Positive Z = above the cube.
APPROACH_OFFSET_Z_M = 0.06  # 6 cm above the cube
LIFT_HEIGHT_M = 0.10         # lift 10 cm after grasping

# Single bus with head (IDs 1-2) and arm (IDs 7-12)
bus = FeetechMotorsBus(port=BUS_PORT, motors=MOTOR_DEFS)
bus.connect()

# Load or create calibration for all motors
load_or_run_calibration(bus)

# Read head positions (calibrated)
head_pos = bus.sync_read("Present_Position", ["head_motor_1", "head_motor_2"])
head_pan_deg = float(head_pos["head_motor_1"])
head_tilt_deg = float(head_pos["head_motor_2"])
print(f"Head motors (deg): pan={head_pan_deg:.2f}, tilt={head_tilt_deg:.2f}")

# Apply limits to right arm only
arm_motors = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


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

# Capture fresh RGBD frames from the RealSense
capture()

# Detect object by color (change color= to "red", "green", or "blue" as needed)
centroid = detect_object(color="red")
print(f"Camera centroid (optical frame): {centroid}")

joint_values = {
    "head_pan_joint": head_pan_deg * DEG2RAD,
    "head_tilt_joint": head_tilt_deg * DEG2RAD,
}
arm_frame_x, arm_frame_y, arm_frame_z = camera_xyz_to_base_xyz(
    centroid[0], centroid[1], centroid[2], joint_values,
)
print(f"Transformed to xlerobot Base frame: [{arm_frame_x:.4f}, {arm_frame_y:.4f}, {arm_frame_z:.4f}]")

ik_solve = IK_SO101()

# camera_xyz_to_base_xyz returns coordinates in Base frame (+Y is forward)
# generate_ik accepts Base frame coordinates directly
cube_base = [
    arm_frame_x + IK_TARGET_OFFSET_X_M,
    arm_frame_y + IK_TARGET_OFFSET_Y_M,
    arm_frame_z + IK_TARGET_OFFSET_Z_M,
]

# Three waypoints: approach (above cube), grab (at cube), lift (above cube after grab)
approach_target = [cube_base[0], cube_base[1], cube_base[2] + APPROACH_OFFSET_Z_M]
grab_target = list(cube_base)
lift_target = [cube_base[0], cube_base[1], cube_base[2] + LIFT_HEIGHT_M]

print(
    "IK target offsets (m): "
    f"[{IK_TARGET_OFFSET_X_M:.4f}, {IK_TARGET_OFFSET_Y_M:.4f}, {IK_TARGET_OFFSET_Z_M:.4f}]"
)
print(f"Cube position (Base frame):  {[f'{v:.4f}' for v in cube_base]}")
print(f"Approach target (Base frame): {[f'{v:.4f}' for v in approach_target]}")
print(f"Grab target (Base frame):     {[f'{v:.4f}' for v in grab_target]}")
print(f"Lift target (Base frame):     {[f'{v:.4f}' for v in lift_target]}")

# Visualize the IK target in Base frame
save_ik_plot(
    base_pos=ik_solve._base_t,
    ik_target_base=np.array(grab_target),
    camera_centroid_cam=np.array(centroid),
)

dt = 0.01


def solve_ik_or_exit(label, target):
    """Run IK for a target, exit on failure."""
    print(f"Running IK solver for '{label}'...")
    traj = ik_solve.generate_ik(target, [0, 0, 0])
    if not traj:
        print(f"IK failed for '{label}' — target may be out of reach.")
        print(f"  Target: {target}")
        print(f"  Distance from base: {np.linalg.norm(target):.4f} m")
        bus.disconnect()
        raise SystemExit(1)
    print(f"  '{label}' IK succeeded: {len(traj)} steps")
    return traj


approach_traj = solve_ik_or_exit("approach", approach_target)
grab_traj = solve_ik_or_exit("grab", grab_target)
lift_traj = solve_ik_or_exit("lift", lift_target)


def mjcf_to_motor(q_deg: np.ndarray) -> np.ndarray:
    """Convert MJCF joint angles (degrees) to motor convention (degrees).

    Joint order: Rotation_L, Pitch_L, Elbow_L, Wrist_Pitch_L, Wrist_Roll_L
    """
    out = q_deg.copy()
    out[0] = -out[0]          # Rotation_L: MJCF positive = left, motor positive = right
    out[1] = 90.0 - out[1]   # Pitch_L -> shoulder_lift
    out[2] = out[2] - 90.0   # Elbow_L -> elbow_flex
    return out


RAD2DEG = 180.0 / np.pi

ARM_JOINT_KEYS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


def traj_to_goals(traj_rad: list[np.ndarray]) -> list[dict]:
    """Convert a list of joint configs (radians) to motor goal dicts (degrees)."""
    stack = np.stack(traj_rad)
    traj_deg = np.array([mjcf_to_motor(q * RAD2DEG) for q in stack])
    goals = []
    for q_deg in traj_deg:
        assert q_deg.shape[0] == len(ARM_JOINT_KEYS)
        goals.append({joint: float(q_deg[i]) for i, joint in enumerate(ARM_JOINT_KEYS)})
    return goals


def send_trajectory(label, traj_rad, gripper_value):
    """Send a trajectory to the motors with a fixed gripper value."""
    goals = traj_to_goals(traj_rad)
    print(f"[{label}] Sending {len(goals)} waypoints (gripper={gripper_value:.0f})...")
    for goal in goals:
        goal["gripper"] = gripper_value
        bus.sync_write("Goal_Position", goal)
        time.sleep(dt)
    # Hold final position
    final = goals[-1].copy()
    final["gripper"] = gripper_value
    bus.sync_write("Goal_Position", final)
    return final


def close_gripper(hold_goal, grip_closed=5.0, step=5, delay=0.05):
    """Smoothly close the gripper from open (100) to grip_closed."""
    print("Closing gripper...")
    for grip in range(100, int(grip_closed) - 1, -step):
        goal = dict(hold_goal)
        goal["gripper"] = float(grip)
        bus.sync_write("Goal_Position", goal)
        time.sleep(delay)
    print(f"  Gripper closed to {grip_closed:.0f}")


# --- Phase 1: Open gripper and move to approach position (above cube) ---
print("\n=== Phase 1: Approach ===")
hold = send_trajectory("approach", approach_traj, gripper_value=100.0)
print("Holding approach position for 2s...")
time.sleep(2.0)

# --- Phase 2: Descend to grab position (gripper still open) ---
print("\n=== Phase 2: Descend to cube ===")
hold = send_trajectory("grab", grab_traj, gripper_value=100.0)
print("At cube. Holding for 1s...")
time.sleep(1.0)

# --- Phase 3: Close gripper around the cube ---
print("\n=== Phase 3: Grasp ===")
close_gripper(hold)
print("Holding grasp for 1s...")
time.sleep(1.0)

# --- Phase 4: Lift up ---
print("\n=== Phase 4: Lift ===")
hold = send_trajectory("lift", lift_traj, gripper_value=5.0)
print("Lifted. Holding for 3s...")
time.sleep(3.0)

# Read back actual positions to verify
actual = bus.sync_read("Present_Position", ARM_JOINT_KEYS)
print("\nActual motor positions (deg):")
for name in ARM_JOINT_KEYS:
    print(f"  {name}: {float(actual[name]):.2f}  (goal: {hold[name]:.2f})")
print("Done.")
bus.disconnect()
