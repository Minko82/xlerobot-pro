from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.xlerobot import XLerobot, XLerobotConfig
from lerobot.motors.motors_bus import MotorCalibration
import numpy as np
import json
from pathlib import Path
from ik_solver import IK_SO101
from point_cloud import PointCloud
from frame_transform import frame_transform
from realsense_capture import capture
import time

SERIAL_PORT = "/dev/ttyACM0"
DEG2RAD = np.pi / 180.0
HEAD_CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration" / "head.json"

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

# Connect follower arm for control
config = SO100FollowerConfig(port=SERIAL_PORT, use_degrees=True)
robot = SO100Follower(config)
robot.connect()

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

RS_JOINT_KEYS = {
    "head_pan_joint": head_pan_deg * DEG2RAD,
    "head_tilt_joint": head_tilt_deg * DEG2RAD,
}
arm_frame_x, arm_frame_y, arm_frame_z = frame_transform.camera_xyz_to_base_xyz(
    centroid[0], centroid[1], centroid[2], RS_JOINT_KEYS
)
print(f"Transformed to xlerobot Base_2 frame: [{arm_frame_x:.4f}, {arm_frame_y:.4f}, {arm_frame_z:.4f}]")

# Rotate from xlerobot Base_2 frame to SO101 IK base_link frame (Rz +90°).
# The xlerobot arm extends along -Y in Base_2, but the SO101 IK URDF
# expects the arm to extend along +X.
ik_x = -arm_frame_y
ik_y = arm_frame_x
ik_z = arm_frame_z
print(f"Rotated to SO101 IK frame: [{ik_x:.4f}, {ik_y:.4f}, {ik_z:.4f}]")

ik_solve = IK_SO101()

dt = 0.01
test_dt = 0.1

trajectory_rad = ik_solve.generate_ik([ik_x, ik_y, ik_z], [0, 0, 0])
# default position tolerance of 1e-3. timesteps at 500
# Move individual joints (degrees)
RAD2DEG = 180.0 / np.pi
traj_rad_stack = np.stack(trajectory_rad)
# Reduced model has gripper locked out; q shape is (5,): shoulder_pan through wrist_roll
trajectory = traj_rad_stack * RAD2DEG

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
