from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
import numpy as np
import json
from pathlib import Path
from ik_solver import IK_SO101
from color_detect import detect_object
from frame_transform.frame_transform import camera_xyz_to_base_xyz
from realsense_capture import capture
import time

# All motors (head IDs 1-2, arm IDs 7-12) on the same bus
BUS_PORT = "/dev/ttyACM0"
DEG2RAD = np.pi / 180.0
CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"

norm_mode_body = MotorNormMode.DEGREES

# Single bus with head (IDs 1-2) and arm (IDs 7-12)
bus = FeetechMotorsBus(
    port=BUS_PORT,
    motors={
        # Head motors
        "head_motor_1": Motor(1, "sts3215", norm_mode_body),
        "head_motor_2": Motor(2, "sts3215", norm_mode_body),
        # Right arm motors
        "shoulder_pan":  Motor(7,  "sts3215", norm_mode_body),
        "shoulder_lift": Motor(8,  "sts3215", norm_mode_body),
        "elbow_flex":    Motor(9,  "sts3215", norm_mode_body),
        "wrist_flex":    Motor(10, "sts3215", norm_mode_body),
        "wrist_roll":    Motor(11, "sts3215", norm_mode_body),
        "gripper":       Motor(12, "sts3215", MotorNormMode.RANGE_0_100),
    },
)
bus.connect()

# Load or create calibration for all motors
CALIBRATION_FILE = CALIBRATION_DIR / "single_bus.json"
if CALIBRATION_FILE.exists():
    with open(CALIBRATION_FILE) as f:
        calib_raw = json.load(f)
    bus.calibration = {
        name: MotorCalibration(**vals) for name, vals in calib_raw.items()
    }
    print(f"Loaded calibration from {CALIBRATION_FILE}")
else:
    print("No calibration file found. Running calibration...")
    all_motors = list(bus.motors.keys())
    bus.disable_torque(all_motors)

    input("\n>>> Move ALL motors to the MIDDLE of their range of motion, then press ENTER...")
    homing_offsets = bus.set_half_turn_homings(all_motors)
    print(f"Homing offsets set: {homing_offsets}")

    print("\n>>> Move ALL motors through their FULL range of motion.")
    input("    Move each joint to both extremes. Press ENTER when done...")
    range_mins, range_maxes = bus.record_ranges_of_motion(all_motors)
    print(f"Range mins: {range_mins}")
    print(f"Range maxes: {range_maxes}")

    calib_raw = {}
    for name in all_motors:
        motor = bus.motors[name]
        calib_raw[name] = {
            "id": motor.id,
            "drive_mode": 0,
            "homing_offset": homing_offsets[name],
            "range_min": range_mins[name],
            "range_max": range_maxes[name],
        }

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(calib_raw, f, indent=4)
    print(f"Calibration saved to {CALIBRATION_FILE}")

    bus.calibration = {
        name: MotorCalibration(**vals) for name, vals in calib_raw.items()
    }

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
    bus.sync_write("Goal_Position", goal)
    time.sleep(dt)

# Close gripper
hold_goal = {k: v for k, v in goals[-1].items() if k != "gripper"}
for grip in range(100, 5, -5):
    goal = dict(hold_goal)
    goal["gripper"] = float(grip)
    bus.sync_write("Goal_Position", goal)
    time.sleep(0.05)

bus.disconnect()
