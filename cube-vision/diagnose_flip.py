#!/usr/bin/env python3
"""Diagnostic: dump every intermediate matrix in the camera-to-base transform.

Run with head at neutral (facing forward) to verify frame orientations.
No hardware needed — uses pinocchio FK only.
"""
import numpy as np
import pinocchio as pin
from pathlib import Path

np.set_printoptions(precision=4, suppress=True)

MJCF_PATH = Path(__file__).resolve().parent / "frame_transform" / "xlerobot" / "xlerobot.xml"
model = pin.buildModelFromMJCF(str(MJCF_PATH))
data = model.createData()

BASE_ID = model.getFrameId("Base")
CAM_LINK_ID = model.getFrameId("head_camera_link")

R_LINK_TO_OPTICAL = np.array([
    [0,  0, 1],
    [1,  0, 0],
    [0, -1, 0],
])

# Neutral configuration (head straight forward)
q = pin.neutral(model)
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

oMbase = data.oMf[BASE_ID]
oMcam = data.oMf[CAM_LINK_ID]

print("=" * 60)
print("FRAME POSES AT NEUTRAL (pinocchio world frame)")
print("=" * 60)

print(f"\nBase translation (world): {oMbase.translation}")
print(f"Base rotation (world):\n{oMbase.rotation}")
print(f"  Base X axis in world: {oMbase.rotation[:, 0]}")
print(f"  Base Y axis in world: {oMbase.rotation[:, 1]}")
print(f"  Base Z axis in world: {oMbase.rotation[:, 2]}")

print(f"\nCamera link translation (world): {oMcam.translation}")
print(f"Camera link rotation (world):\n{oMcam.rotation}")
print(f"  Cam link X axis in world: {oMcam.rotation[:, 0]}")
print(f"  Cam link Y axis in world: {oMcam.rotation[:, 1]}")
print(f"  Cam link Z axis in world: {oMcam.rotation[:, 2]}")

R_cam_optical = oMcam.rotation @ R_LINK_TO_OPTICAL
print(f"\nR_cam_optical (optical axes in world):\n{R_cam_optical}")
print(f"  Optical X (right) in world:   {R_cam_optical[:, 0]}")
print(f"  Optical Y (down) in world:    {R_cam_optical[:, 1]}")
print(f"  Optical Z (forward) in world: {R_cam_optical[:, 2]}")
print(f"  det(R_cam_optical) = {np.linalg.det(R_cam_optical):.4f}")

print(f"\ndet(R_LINK_TO_OPTICAL) = {np.linalg.det(R_LINK_TO_OPTICAL):.4f}")

# Transform: camera optical → Base frame
R = oMbase.rotation.T @ R_cam_optical
t = oMbase.rotation.T @ (oMcam.translation - oMbase.translation)
print(f"\nR_base_from_optical:\n{R}")
print(f"t (camera origin in Base frame): {t}")

print("\n" + "=" * 60)
print("TEST POINTS (camera optical frame → Base frame)")
print("=" * 60)

test_points = {
    "center, 50cm forward":   [0.0,  0.0,  0.5],
    "10cm RIGHT, 50cm fwd":   [0.1,  0.0,  0.5],
    "10cm LEFT, 50cm fwd":    [-0.1, 0.0,  0.5],
    "10cm DOWN, 50cm fwd":    [0.0,  0.1,  0.5],
    "10cm UP, 50cm fwd":      [0.0, -0.1,  0.5],
}

for label, p_opt in test_points.items():
    p_opt = np.array(p_opt)
    p_base = R @ p_opt + t
    print(f"\n  {label}")
    print(f"    optical: {p_opt}")
    print(f"    base:    {p_base}")

print("\n" + "=" * 60)
print("INTERPRETATION GUIDE")
print("=" * 60)
print("""
Robot faces -X world. Standing BEHIND the robot, looking forward:
  Robot RIGHT = -Y world
  Robot LEFT  = +Y world

Check: when optical point moves RIGHT (+X optical),
which Base axis changes and in which direction?
That tells you if left/right mapping is correct.
""")
