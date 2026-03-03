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
IK_TARGET_OFFSET_X_M = -0.07
IK_TARGET_OFFSET_Y_M = 0.0
IK_TARGET_OFFSET_Z_M = 0.00

# Height above the target to hover before descending (meters)
HOVER_HEIGHT_M = 0.10

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

RAD2DEG = 180.0 / np.pi

ARM_JOINT_KEYS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


def motor_to_mjcf(q_deg: np.ndarray) -> np.ndarray:
    """Convert motor convention angles (degrees) to MJCF joint angles (degrees).

    Inverse of mjcf_to_motor.
    Joint order: Rotation_L, Pitch_L, Elbow_L, Wrist_Pitch_L, Wrist_Roll_L
    """
    out = q_deg.copy()
    out[0] = -out[0]          # motor positive = right -> MJCF positive = left
    out[1] = 90.0 - out[1]   # shoulder_lift -> Pitch_L
    out[2] = out[2] + 90.0   # elbow_flex -> Elbow_L
    return out


def mjcf_to_motor(q_deg: np.ndarray) -> np.ndarray:
    """Convert MJCF joint angles (degrees) to motor convention (degrees).

    Joint order: Rotation_L, Pitch_L, Elbow_L, Wrist_Pitch_L, Wrist_Roll_L
    """
    out = q_deg.copy()
    out[0] = -out[0]          # Rotation_L: MJCF positive = left, motor positive = right
    out[1] = 90.0 - out[1]   # Pitch_L -> shoulder_lift
    out[2] = out[2] - 90.0   # Elbow_L -> elbow_flex
    return out


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

# Compute final target (at the cube) and hover target (above the cube)
target_base = [
    arm_frame_x + IK_TARGET_OFFSET_X_M,
    arm_frame_y + IK_TARGET_OFFSET_Y_M,
    arm_frame_z + IK_TARGET_OFFSET_Z_M,
]
hover_base = [
    target_base[0],
    target_base[1],
    target_base[2] + HOVER_HEIGHT_M,
]
print(
    "IK target offsets (m): "
    f"[{IK_TARGET_OFFSET_X_M:.4f}, {IK_TARGET_OFFSET_Y_M:.4f}, {IK_TARGET_OFFSET_Z_M:.4f}]"
)
print(f"Final target (Base frame):  [{target_base[0]:.4f}, {target_base[1]:.4f}, {target_base[2]:.4f}]")
print(f"Hover target (Base frame):  [{hover_base[0]:.4f}, {hover_base[1]:.4f}, {hover_base[2]:.4f}]")

# Visualize the IK target in Base frame
save_ik_plot(
    base_pos=ik_solve._base_t,
    ik_target_base=np.array(target_base),
    camera_centroid_cam=np.array(centroid),
)

dt = 0.01

# Read current arm positions to use as IK seed (smoother trajectories from current pose)
current_arm_pos = bus.sync_read("Present_Position", ARM_JOINT_KEYS)
current_motor_deg = np.array([float(current_arm_pos[j]) for j in ARM_JOINT_KEYS])
current_mjcf_deg = motor_to_mjcf(current_motor_deg)
current_mjcf_rad = current_mjcf_deg * DEG2RAD
print(f"Current arm positions (motor deg): {current_motor_deg}")
print(f"Current arm positions (MJCF deg):  {current_mjcf_deg}")


def traj_to_goals(traj_rad: list[np.ndarray]) -> list[dict]:
    """Convert a list of joint configs (radians) to motor goal dicts (degrees)."""
    stack = np.stack(traj_rad)
    traj_deg = np.array([mjcf_to_motor(q * RAD2DEG) for q in stack])
    goals = []
    for q_deg in traj_deg:
        assert q_deg.shape[0] == len(ARM_JOINT_KEYS)
        goals.append({joint: float(q_deg[i]) for i, joint in enumerate(ARM_JOINT_KEYS)})
    return goals


def send_trajectory(goals: list[dict], gripper_deg: float, hold_time: float):
    """Send a list of motor goal dicts, then hold the final position."""
    for goal in goals:
        goal["gripper"] = gripper_deg
        bus.sync_write("Goal_Position", goal)
        time.sleep(dt)
    # Re-send final goal and hold so motors physically reach it
    final = goals[-1].copy()
    final["gripper"] = gripper_deg
    bus.sync_write("Goal_Position", final)
    time.sleep(hold_time)


# ── Stage 1: Move to hover point above the cube ──
print("\n── Stage 1: Moving to hover point above cube ──")
hover_traj = ik_solve.generate_ik(hover_base, [0, 0, 0], seed_q_rad=current_mjcf_rad)
if not hover_traj:
    print("IK failed for hover target.")
    print(f"  Hover target: {hover_base}")
    bus.disconnect()
    raise SystemExit(1)
print(f"Hover IK succeeded: {len(hover_traj)} steps")

hover_goals = traj_to_goals(hover_traj)
print(f"  First goal: {hover_goals[0]}")
print(f"  Last goal:  {hover_goals[-1]}")
send_trajectory(hover_goals, gripper_deg=100.0, hold_time=3.0)
print("Hover position reached.")

# ── Stage 2: Descend to the cube ──
print("\n── Stage 2: Descending to cube ──")
# Seed the descent from the hover configuration for a smooth vertical path
hover_q_rad = hover_traj[-1]
descend_traj = ik_solve.generate_ik(target_base, [0, 0, 0], seed_q_rad=hover_q_rad)
if not descend_traj:
    print("IK failed for descent target.")
    print(f"  Descent target: {target_base}")
    bus.disconnect()
    raise SystemExit(1)
print(f"Descent IK succeeded: {len(descend_traj)} steps")

descend_goals = traj_to_goals(descend_traj)
print(f"  First goal: {descend_goals[0]}")
print(f"  Last goal:  {descend_goals[-1]}")
send_trajectory(descend_goals, gripper_deg=100.0, hold_time=3.0)
print("Descent complete — at cube.")

# Read back actual positions to verify
actual = bus.sync_read("Present_Position", ARM_JOINT_KEYS)
final_goal = descend_goals[-1]
print("\nActual motor positions (deg):")
for name in ARM_JOINT_KEYS:
    print(f"  {name}: {float(actual[name]):.2f}  (goal: {final_goal[name]:.2f})")
print("Done.")
bus.disconnect()
