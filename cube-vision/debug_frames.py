"""Diagnostic: compare arm FK at neutral in both URDFs to find the frame relationship."""

import numpy as np
from frame_transform import frame_transform

# --- xlerobot arm FK at neutral (all joints = 0) using the chain from frame_transform ---
chain = frame_transform._ARM_CHAIN
T = np.eye(4)
for xyz, rpy, axis in chain:
    if axis is None:
        T_local = frame_transform._tf(xyz, rpy)
    else:
        T_local = frame_transform._revolute(xyz, rpy, axis, 0.0)
    T = T @ T_local

xlerobot_tip = T[:3, 3]
print("=== xlerobot_front.urdf: arm tip in Base_2 frame at neutral ===")
print(f"  Position: x={xlerobot_tip[0]:.4f}, y={xlerobot_tip[1]:.4f}, z={xlerobot_tip[2]:.4f}")
print(f"  Rotation:\n{T[:3,:3]}")

# --- SO101 arm FK at neutral using pinocchio ---
try:
    import pinocchio as pin
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent / "ik_solver"
    URDF_PATH = str(BASE_DIR / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf")
    MESH_DIR = str(BASE_DIR / "SO-ARM100" / "Simulation" / "SO101")

    full_model = pin.buildModelFromUrdf(URDF_PATH)
    data = full_model.createData()
    q = pin.neutral(full_model)

    pin.forwardKinematics(full_model, data, q)
    pin.updateFramePlacements(full_model, data)

    # Find gripper frame
    frame_id = full_model.getFrameId("gripper_frame_link")
    oMf = data.oMf[frame_id]
    so101_tip = oMf.translation
    print("\n=== so101_new_calib.urdf: gripper_frame_link at neutral ===")
    print(f"  Position: x={so101_tip[0]:.4f}, y={so101_tip[1]:.4f}, z={so101_tip[2]:.4f}")
    print(f"  Rotation:\n{oMf.rotation}")
except Exception as e:
    print(f"\nCould not load SO101 URDF with pinocchio: {e}")

# --- Also show what the camera transform looks like ---
print("\n=== Camera transform test (head at 0,0) ===")
joint_values = {"head_pan_joint": 0.0, "head_tilt_joint": 0.0}
# Test point: camera optical frame (0, 0, 0.5) = 50cm directly ahead
bx, by, bz = frame_transform.camera_xyz_to_base_xyz(0.0, 0.0, 0.5, joint_values)
print(f"  Camera (0, 0, 0.5) -> Base_2 ({bx:.4f}, {by:.4f}, {bz:.4f})")

bx2, by2, bz2 = frame_transform.camera_xyz_to_base_xyz(0.0, 0.05, 0.5, joint_values)
print(f"  Camera (0, 0.05, 0.5) -> Base_2 ({bx2:.4f}, {by2:.4f}, {bz2:.4f})")

print("\n=== Proposed Rz(90) correction ===")
print(f"  Camera (0, 0, 0.5) -> SO101 IK frame ({-by:.4f}, {bx:.4f}, {bz:.4f})")
print(f"  Camera (0, 0.05, 0.5) -> SO101 IK frame ({-by2:.4f}, {bx2:.4f}, {bz2:.4f})")
