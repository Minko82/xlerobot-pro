#!/usr/bin/env python3
"""Diagnostic: place the gripper on the cube, then compare FK position vs vision position.

Usage:
    1. Disable torque so you can move the arm freely
    2. Place the gripper tip directly on the cube
    3. Run:  python diagnose.py
    4. The script reads arm joints (FK) and camera (vision) and compares them
"""

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import MotorNormMode
import numpy as np
import pinocchio as pin
from pathlib import Path

from calibrate import MOTOR_DEFS, BUS_PORT, load_or_run_calibration
from color_detect import detect_object
from frame_transform.frame_transform import camera_xyz_to_base_xyz
from realsense_capture import capture

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# ── Connect and calibrate ──────────────────────────────────────────────
bus = FeetechMotorsBus(port=BUS_PORT, motors=MOTOR_DEFS)
bus.connect()
load_or_run_calibration(bus)

# Disable torque so the user can position the arm by hand
all_motors = list(bus.motors.keys())
bus.disable_torque(all_motors)
input("\n>>> Place the gripper tip ON the cube, then press ENTER...")

# ── Step 1: Record arm + head positions ─────────────────────────────────
positions = bus.sync_read("Present_Position", all_motors)
print("\n=== Motor Positions (calibrated degrees) ===")
for name in all_motors:
    print(f"  {name:20s} = {float(positions[name]):8.2f}°")

head_pan_deg = float(positions["head_motor_1"])
head_tilt_deg = float(positions["head_motor_2"])

arm_motor_deg = np.array([
    float(positions["shoulder_pan"]),
    float(positions["shoulder_lift"]),
    float(positions["elbow_flex"]),
    float(positions["wrist_flex"]),
    float(positions["wrist_roll"]),
])

print(f"\nHead: pan={head_pan_deg:.2f}°, tilt={head_tilt_deg:.2f}°")
print(f"Arm (motor deg): {arm_motor_deg}")
print("\nArm position recorded!")

# ── Step 2: Move gripper out of the way before capturing ────────────────
input("\n>>> Now move the gripper OUT OF THE WAY of the cube, then press ENTER...")

# Re-read head in case it shifted (arm positions already saved above)
head_pos2 = bus.sync_read("Present_Position", ["head_motor_1", "head_motor_2"])
head_pan_deg = float(head_pos2["head_motor_1"])
head_tilt_deg = float(head_pos2["head_motor_2"])
print(f"Head (re-read): pan={head_pan_deg:.2f}°, tilt={head_tilt_deg:.2f}°")

# ── Arm FK: compute EE position in Base frame ──────────────────────────
MJCF_PATH = Path(__file__).resolve().parent / "assets" / "xlerobot.xml"
ARM_JOINTS = {"Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L", "Wrist_Roll_L"}

full_model = pin.buildModelFromMJCF(str(MJCF_PATH))
q_neutral = pin.neutral(full_model)

# Build reduced arm model (same as IK solver)
lock_ids = [
    i for i in range(1, full_model.njoints)
    if full_model.names[i] not in ARM_JOINTS
]
arm_model = pin.buildReducedModel(full_model, lock_ids, q_neutral)
arm_data = arm_model.createData()

# Convert motor degrees -> MJCF degrees -> radians
def motor_to_mjcf(q_deg):
    out = q_deg.copy()
    out[1] = 90.0 - out[1]   # shoulder_lift -> Pitch_L
    out[2] = out[2] + 90.0   # elbow_flex -> Elbow_L
    return out

mjcf_deg = motor_to_mjcf(arm_motor_deg)
mjcf_rad = mjcf_deg * DEG2RAD
print(f"Arm (MJCF deg):  {mjcf_deg}")
print(f"Arm (MJCF rad):  {mjcf_rad}")

# Set joint configuration
q = pin.neutral(arm_model)
arm_joint_names = ["Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L", "Wrist_Roll_L"]
for i, jname in enumerate(arm_joint_names):
    jid = arm_model.getJointId(jname)
    q[arm_model.joints[jid].idx_q] = mjcf_rad[i]

pin.forwardKinematics(arm_model, arm_data, q)
pin.updateFramePlacements(arm_model, arm_data)

# Get EE and Base in world frame
ee_frame_id = arm_model.getFrameId("Fixed_Jaw")
base_frame_id = arm_model.getFrameId("Base")

oMee = arm_data.oMf[ee_frame_id]
oMbase = arm_data.oMf[base_frame_id]

# EE in Base frame
ee_in_base = oMbase.rotation.T @ (oMee.translation - oMbase.translation)

print(f"\n=== ARM FK (End-Effector in Base frame) ===")
print(f"  EE world:  [{oMee.translation[0]:.4f}, {oMee.translation[1]:.4f}, {oMee.translation[2]:.4f}]")
print(f"  Base world: [{oMbase.translation[0]:.4f}, {oMbase.translation[1]:.4f}, {oMbase.translation[2]:.4f}]")
print(f"  EE in Base: [{ee_in_base[0]:.4f}, {ee_in_base[1]:.4f}, {ee_in_base[2]:.4f}]")

# ── Vision: detect cube and transform to Base ──────────────────────────
print("\n=== Capturing from RealSense... ===")
capture()

try:
    centroid_cam = detect_object(color="red")
    print(f"  Camera centroid (optical frame): [{centroid_cam[0]:.4f}, {centroid_cam[1]:.4f}, {centroid_cam[2]:.4f}]")

    joint_values = {
        "head_pan_joint": head_pan_deg * DEG2RAD,
        "head_tilt_joint": head_tilt_deg * DEG2RAD,
    }

    # Show what the head motor-to-MJCF conversion produces
    from frame_transform.frame_transform import _head_motor_to_mjcf
    head_mjcf = _head_motor_to_mjcf(np.array([head_pan_deg, head_tilt_deg]))
    print(f"\n  Head motor deg:  pan={head_pan_deg:.2f}, tilt={head_tilt_deg:.2f}")
    print(f"  Head MJCF deg:   pan={head_mjcf[0]:.2f}, tilt={head_mjcf[1]:.2f}")

    vx, vy, vz = camera_xyz_to_base_xyz(
        centroid_cam[0], centroid_cam[1], centroid_cam[2], joint_values,
    )
    print(f"\n=== VISION (Cube in Base frame) ===")
    print(f"  Vision Base: [{vx:.4f}, {vy:.4f}, {vz:.4f}]")
except Exception as e:
    print(f"  Vision detection failed: {e}")
    vx, vy, vz = None, None, None

# ── Compare ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"  FK  (EE in Base): [{ee_in_base[0]:.4f}, {ee_in_base[1]:.4f}, {ee_in_base[2]:.4f}]")
if vx is not None:
    print(f"  VIS (cube Base):  [{vx:.4f}, {vy:.4f}, {vz:.4f}]")
    err = np.array([vx - ee_in_base[0], vy - ee_in_base[1], vz - ee_in_base[2]])
    print(f"  Error (VIS - FK):   [{err[0]:.4f}, {err[1]:.4f}, {err[2]:.4f}]")
    print(f"  Error magnitude:    {np.linalg.norm(err)*100:.1f} cm")
    print(f"\n  Breakdown:")
    print(f"    X error: {err[0]*100:+.1f} cm  (Base left/right)")
    print(f"    Y error: {err[1]*100:+.1f} cm  (Base forward/back)")
    print(f"    Z error: {err[2]*100:+.1f} cm  (Base up/down)")
else:
    print(f"  VIS: failed — no comparison possible")

print()
bus.disconnect()
