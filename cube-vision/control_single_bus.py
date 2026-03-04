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
IK_TARGET_OFFSET_X_M = -0.12
IK_TARGET_OFFSET_Y_M = 0.0
IK_TARGET_OFFSET_Z_M = 0.05

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


apply_limits(bus, [m for m in arm_motors if m != "gripper"], 500, 10, 8, 0, 32)
apply_limits(bus, ["gripper"], 500, 10, 8, 0, 32)

# Open gripper immediately
print("Opening gripper...")
bus.sync_write("Goal_Position", {"gripper": 100.0})
time.sleep(1.0)

ik_solve = IK_SO101()
dt = 0.01

ARM_JOINT_KEYS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
]


def motor_to_mjcf(q_deg: np.ndarray) -> np.ndarray:
    out = q_deg.copy()
    out[0] = -out[0]
    out[1] = 90.0 - out[1]
    out[2] = out[2] + 90.0
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


RAD2DEG = 180.0 / np.pi


def traj_to_goals(traj_rad: list[np.ndarray]) -> list[dict]:
    """Convert a list of joint configs (radians) to motor goal dicts (degrees)."""
    stack = np.stack(traj_rad)
    traj_deg = np.array([mjcf_to_motor(q * RAD2DEG) for q in stack])
    goals = []
    for q_deg in traj_deg:
        assert q_deg.shape[0] == len(ARM_JOINT_KEYS)
        goals.append({joint: float(q_deg[i]) for i, joint in enumerate(ARM_JOINT_KEYS)})
    return goals


# Save start position before looping
start_arm_pos = bus.sync_read("Present_Position", ARM_JOINT_KEYS)
start_motor_deg = np.array([float(start_arm_pos[j]) for j in ARM_JOINT_KEYS])

NUM_GRABS = 1
for grab_i in range(NUM_GRABS):
    print(f"\n{'='*40}")
    print(f"  GRAB {grab_i + 1} / {NUM_GRABS}")
    print(f"{'='*40}\n")

    # Capture fresh RGBD frames from the RealSense
    capture()

    # Detect object by color
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

    target_base = [
        arm_frame_x + IK_TARGET_OFFSET_X_M,
        arm_frame_y + IK_TARGET_OFFSET_Y_M,
        arm_frame_z + IK_TARGET_OFFSET_Z_M,
    ]
    print(f"IK target (Base frame, offset): "
          f"[{target_base[0]:.4f}, {target_base[1]:.4f}, {target_base[2]:.4f}]")

    save_ik_plot(
        base_pos=ik_solve._base_t,
        ik_target_base=np.array(target_base),
        camera_centroid_cam=np.array(centroid),
    )

    # Read current arm positions to seed IK
    current_arm_pos = bus.sync_read("Present_Position", ARM_JOINT_KEYS)
    current_motor_deg = np.array([float(current_arm_pos[j]) for j in ARM_JOINT_KEYS])
    current_mjcf_rad = motor_to_mjcf(current_motor_deg) * DEG2RAD
    current_wrist_roll = float(current_motor_deg[4])

    print("Running IK solver...")
    trajectory_rad = ik_solve.generate_ik(target_base, [0, 0, 0], seed_q_rad=current_mjcf_rad)
    if not trajectory_rad:
        print("IK failed — skipping this grab.")
        continue

    print(f"IK succeeded: {len(trajectory_rad)} steps")

    # Move to target with gripper open
    print(f"Pinning wrist_roll at {current_wrist_roll:.2f} deg")
    goals = traj_to_goals(trajectory_rad)
    print(f"Sending {len(goals)} waypoints to motors...")
    for goal in goals:
        goal["gripper"] = 100.0
        goal["wrist_roll"] = current_wrist_roll
        bus.sync_write("Goal_Position", goal)
        time.sleep(dt)

    # Hold final position
    final_goal = goals[-1].copy()
    final_goal["gripper"] = 100.0
    final_goal["wrist_roll"] = current_wrist_roll
    bus.sync_write("Goal_Position", final_goal)
    print("Holding position for 3 seconds...")
    time.sleep(3.0)

    # Close gripper
    print("Closing gripper...")
    for grip in range(100, -1, -5):
        final_goal["gripper"] = float(grip)
        bus.sync_write("Goal_Position", final_goal)
        time.sleep(0.05)
    time.sleep(1.0)
    print("Gripper closed.")

    # Lift cube 10cm
    lift_base = [target_base[0], target_base[1], target_base[2] + 0.10]
    print(f"Lifting to: [{lift_base[0]:.4f}, {lift_base[1]:.4f}, {lift_base[2]:.4f}]")
    lift_traj = ik_solve.generate_ik(lift_base, [0, 0, 0], seed_q_rad=trajectory_rad[-1])
    if not lift_traj:
        print("IK failed for lift — skipping.")
        continue
    lift_goals = traj_to_goals(lift_traj)
    for goal in lift_goals:
        goal["gripper"] = 0.0
        goal["wrist_roll"] = current_wrist_roll
        bus.sync_write("Goal_Position", goal)
        time.sleep(dt)
    time.sleep(2.0)

    # Open gripper to drop
    print("Dropping cube...")
    drop_goal = lift_goals[-1].copy()
    for grip in range(0, 101, 5):
        drop_goal["gripper"] = float(grip)
        bus.sync_write("Goal_Position", drop_goal)
        time.sleep(0.05)
    time.sleep(1.0)
    print("Cube dropped.")

    # Return to start position
    print("Returning to start position...")
    start_goal = {joint: float(start_motor_deg[i]) for i, joint in enumerate(ARM_JOINT_KEYS)}
    start_goal["gripper"] = 100.0
    bus.sync_write("Goal_Position", start_goal)
    time.sleep(5.0)

print(f"\nAll {NUM_GRABS} grabs complete.")
bus.disconnect()
