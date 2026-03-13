from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
import numpy as np
import json
from pathlib import Path
from ik_solver import IK_SO101
from point_cloud import PointCloud
from frame_transform.frame_transform import camera_xyz_to_base_xyz
from realsense_capture import capture
import time

ARM_PORT = "/dev/ttyACM1"
HEAD_PORT = "/dev/ttyACM0"
DEG2RAD = np.pi / 180.0
HEAD_CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration" / "head.json"

norm_mode_body = MotorNormMode.DEGREES

# Right arm bus (IDs 7-12) on ACM1
arm_bus = FeetechMotorsBus(
    port=ARM_PORT,
    motors={
        "shoulder_pan":  Motor(7,  "sts3215", norm_mode_body),
        "shoulder_lift": Motor(8,  "sts3215", norm_mode_body),
        "elbow_flex":    Motor(9,  "sts3215", norm_mode_body),
        "wrist_flex":    Motor(10, "sts3215", norm_mode_body),
        "wrist_roll":    Motor(11, "sts3215", norm_mode_body),
        "gripper":       Motor(12, "sts3215", MotorNormMode.RANGE_0_100),
    },
)
arm_bus.connect()

# Head bus (IDs 1-2) on ACM0
head_bus = FeetechMotorsBus(
    port=HEAD_PORT,
    motors={
        "head_motor_1": Motor(1, "sts3215", norm_mode_body),
        "head_motor_2": Motor(2, "sts3215", norm_mode_body),
    },
)
head_bus.connect()

# Load head calibration so sync_read returns calibrated degrees
if not HEAD_CALIBRATION_FILE.exists():
    raise FileNotFoundError(
        f"Head calibration not found: {HEAD_CALIBRATION_FILE}\n"
        "Run calibrate_head.py first."
    )
with open(HEAD_CALIBRATION_FILE) as f:
    head_calib_raw = json.load(f)
head_bus.calibration = {
    name: MotorCalibration(**vals) for name, vals in head_calib_raw.items()
}

# Read head positions
head_pos = head_bus.sync_read("Present_Position", ["head_motor_1", "head_motor_2"])
head_pan_deg = float(head_pos["head_motor_1"])
head_tilt_deg = float(head_pos["head_motor_2"])
print(f"Head motors (deg): pan={head_pan_deg:.2f}, tilt={head_tilt_deg:.2f}")

# Apply limits to right arm
arm_motors = list(arm_bus.motors.keys())


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


apply_limits(arm_bus, arm_motors, 200, 10, 8, 0, 32)

# Capture fresh RGBD frames from the RealSense
capture()

# Get coordinate object from the frame of the realsense
point_cloud = PointCloud()
point_cloud.create_point_cloud_from_rgbd()
point_cloud.segment_plane()
objects = point_cloud.dbscan_objects(min_points_per_object=500)
if not objects:
    raise RuntimeError("No objects detected in point cloud")
# Pick the largest cluster — the cube
objects.sort(key=lambda o: o["num_points"])
centroid = objects[-1]["centroid"]
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

# camera_xyz_to_base_xyz returns coordinates in Base frame.
# generate_ik accepts Base frame coordinates directly (it converts to world internally).
target_base = [arm_frame_x, arm_frame_y, arm_frame_z]
print(f"IK target (Base frame): [{target_base[0]:.4f}, {target_base[1]:.4f}, {target_base[2]:.4f}]")

dt = 0.01

trajectory_rad = ik_solve.generate_ik(target_base, [0, 0, 0])
# default position tolerance of 1e-3. timesteps at 500


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
traj_rad_stack = np.stack(trajectory_rad)
# Convert MJCF radians -> degrees -> motor convention
trajectory = np.array([mjcf_to_motor(q * RAD2DEG) for q in traj_rad_stack])

ARM_JOINT_KEYS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


def traj_to_goal(q_deg: np.ndarray) -> dict:
    assert q_deg.shape[0] == len(ARM_JOINT_KEYS)
    return {joint: float(q_deg[i]) for i, joint in enumerate(ARM_JOINT_KEYS)}


goals = [traj_to_goal(q_deg) for q_deg in trajectory]

for goal in goals:
    goal["gripper"] = 100.0
    arm_bus.sync_write("Goal_Position", goal)
    time.sleep(dt)

# Close gripper
hold_goal = {k: v for k, v in goals[-1].items() if k != "gripper"}
for grip in range(100, 5, -5):
    goal = dict(hold_goal)
    goal["gripper"] = float(grip)
    arm_bus.sync_write("Goal_Position", goal)
    time.sleep(0.05)

arm_bus.disconnect()
head_bus.disconnect()
