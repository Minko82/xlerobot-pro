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
IK_TARGET_OFFSET_X_M = 0.0
IK_TARGET_OFFSET_Y_M = 0.0
IK_TARGET_OFFSET_Z_M = 0.0

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
target_base = [
    arm_frame_x + IK_TARGET_OFFSET_X_M,
    arm_frame_y + IK_TARGET_OFFSET_Y_M,
    arm_frame_z + IK_TARGET_OFFSET_Z_M,
]
print(
    "IK target offsets (m): "
    f"[{IK_TARGET_OFFSET_X_M:.4f}, {IK_TARGET_OFFSET_Y_M:.4f}, {IK_TARGET_OFFSET_Z_M:.4f}]"
)
print(
    f"IK target (Base frame, offset): "
    f"[{target_base[0]:.4f}, {target_base[1]:.4f}, {target_base[2]:.4f}]"
)

# Visualize the IK target in Base frame
save_ik_plot(
    base_pos=ik_solve._base2_t,
    ik_target_base=np.array(target_base),
    camera_centroid_cam=np.array(centroid),
)

dt = 0.01

trajectory_rad = ik_solve.generate_ik(target_base, [0, 0, 0])
if not trajectory_rad:
    print("IK failed — aborting.")
    bus.disconnect()
    raise SystemExit(1)


def mjcf_to_motor(q_deg: np.ndarray) -> np.ndarray:
    """Convert MJCF joint angles (degrees) to motor convention (degrees).

    Joint order: Rotation_L, Pitch_L, Elbow_L, Wrist_Pitch_L, Wrist_Roll_L
    """
    out = q_deg.copy()
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


# Move to 10cm above target (gripper open)
goals = traj_to_goals(trajectory_rad)
for goal in goals:
    goal["gripper"] = 100.0
    bus.sync_write("Goal_Position", goal)
    time.sleep(dt)

bus.disconnect()
