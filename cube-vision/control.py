from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
from lerobot.motors.motors_bus import MotorCalibration
from lerobot.motors.feetech.feetech import OperatingMode
import numpy as np
import json
from pathlib import Path
from ik_solver import IK_SO101
from point_cloud import PointCloud
from frame_transform.frame_transform import camera_xyz_to_base_xyz
from realsense_capture import capture
import time

SERIAL_PORT = "/dev/ttyACM0"
DEG2RAD = np.pi / 180.0
HEAD_CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration" / "head.json"


def apply_limits(robot: SO100Follower, torque: int, acceleration: int, p: int, i: int, d: int):
    bus = robot.bus
    motors = list(bus.motors.keys())

    print("Applying motor limits:")
    print(f"    Torque_Limit = {torque} / 1000")
    print(f"    Acceleration = {acceleration} / 254")
    print(f"    P={p}, I={i}, D={d}\n")

    bus.disable_torque()
    for name in motors:
        bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
        bus.write("Torque_Limit", name, torque)
        bus.write("Acceleration", name, acceleration)
        bus.write("P_Coefficient", name, p)
        bus.write("I_Coefficient", name, i)
        bus.write("D_Coefficient", name, d)
    bus.enable_torque()


# Connect to bus1 only to read head motor positions
xlerobot_config = XLerobotConfig(port1=SERIAL_PORT, use_degrees=True)
xlerobot = XLerobot(xlerobot_config)
xlerobot.bus1.connect()

# Load head calibration so sync_read returns calibrated degrees
if not HEAD_CALIBRATION_FILE.exists():
    raise FileNotFoundError(
        f"Head calibration not found: {HEAD_CALIBRATION_FILE}\n"
        "Run calibrate_head.py first."
    )
with open(HEAD_CALIBRATION_FILE) as f:
    head_calib_raw = json.load(f)
head_calibration = {
    name: MotorCalibration(**vals) for name, vals in head_calib_raw.items()
}
xlerobot.bus1.calibration = head_calibration

head_pos = xlerobot.bus1.sync_read("Present_Position", xlerobot.head_motors)
head_pan_deg = float(head_pos["head_motor_1"])
head_tilt_deg = float(head_pos["head_motor_2"])
xlerobot.bus1.disconnect()
print(f"Head motors (deg): pan={head_pan_deg:.2f}, tilt={head_tilt_deg:.2f}")

# Connect follower arm for control (SO101Follower matches the IK solver's conventions)
config = SO100FollowerConfig(port=SERIAL_PORT, use_degrees=True)
robot = SO100Follower(config)
robot.connect()

apply_limits(robot, 200, 10, 8, 0, 32)
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
print(f"Transformed to xlerobot Base_2 frame: [{arm_frame_x:.4f}, {arm_frame_y:.4f}, {arm_frame_z:.4f}]")

ik_solve = IK_SO101()

# Convert Base_2 frame coordinates to the IK model's world frame (no manual rotation needed)
target_world = ik_solve.base2_to_world(np.array([arm_frame_x, arm_frame_y, arm_frame_z]))
print(f"IK target (world frame): [{target_world[0]:.4f}, {target_world[1]:.4f}, {target_world[2]:.4f}]")

dt = 0.01

trajectory_rad = ik_solve.generate_ik(target_world.tolist(), [0, 0, 0])
# default position tolerance of 1e-3. timesteps at 500


def mjcf_to_motor(q_deg: np.ndarray) -> np.ndarray:
    """Convert MJCF joint angles (degrees) to motor convention (degrees).

    Joint order: Rotation_R, Pitch_R, Elbow_R, Wrist_Pitch_R, Wrist_Roll_R
    """
    out = q_deg.copy()
    out[1] = 90.0 - out[1]   # Pitch_R -> shoulder_lift
    out[2] = out[2] - 90.0   # Elbow_R -> elbow_flex
    return out


RAD2DEG = 180.0 / np.pi
traj_rad_stack = np.stack(trajectory_rad)
# Convert MJCF radians -> degrees -> motor convention
trajectory = np.array([mjcf_to_motor(q * RAD2DEG) for q in traj_rad_stack])

ARM_JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
]


def traj_to_action(q_deg: np.ndarray) -> dict:
    # Convert list of values to dict for lerobot usage
    assert q_deg.shape[0] == len(ARM_JOINT_KEYS)

    return {joint: float(q_deg[i]) for i, joint in enumerate(ARM_JOINT_KEYS)}


actions = [traj_to_action(q_deg) for q_deg in trajectory]

for action in actions:
    action["gripper.pos"] = 100.0
    robot.send_action(action)
    time.sleep(dt)


hold_action = {k: v for k, v in actions[-1].items() if k != "gripper.pos"}
for grip in range(100, 5, -5):
    action = dict(hold_action)
    action["gripper.pos"] = float(grip)
    robot.send_action(action)
    time.sleep(0.05)

robot.disconnect()
