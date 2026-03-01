"""Diagnostic: dump the full camera-to-base transform chain."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pinocchio as pin

from pincer_transform.constants import (
    ARM_JOINTS, BASE_FRAME, CAMERA_FRAME, HEAD_JOINTS, URDF_PATH,
)
from pincer_transform.conventions import arm_motor_to_urdf, head_motor_to_urdf


def main():
    # ── Inputs (paste your measured values here) ──
    head_pan_motor = 80.57   # degrees, from control.py output
    head_tilt_motor = 30.73

    camera_centroid = np.array([0.18457174, -0.25867821, 0.67791008])  # optical frame

    ee_base = np.array([0.11664671, -0.07417904, 0.47315754])  # from measure_ee.py

    # ── Build model ──
    model = pin.buildModelFromUrdf(str(URDF_PATH))
    data = model.createData()

    q_full = pin.neutral(model)

    # Set arm joints to zero (they don't affect Base or camera placement)
    q_arm = np.zeros(5)
    for name, q_deg in zip(ARM_JOINTS, arm_motor_to_urdf(q_arm)):
        jid = model.getJointId(name)
        q_full[model.joints[jid].idx_q] = np.deg2rad(float(q_deg))

    # Set head joints
    q_head_motor = np.array([head_pan_motor, head_tilt_motor])
    q_head_urdf = head_motor_to_urdf(q_head_motor)
    print(f"Head motor (deg): pan={head_pan_motor}, tilt={head_tilt_motor}")
    print(f"Head URDF  (deg): pan={q_head_urdf[0]:.2f}, tilt={q_head_urdf[1]:.2f}")
    print()

    for name, q_deg in zip(HEAD_JOINTS, q_head_urdf):
        jid = model.getJointId(name)
        idx = model.joints[jid].idx_q
        q_full[idx] = np.deg2rad(float(q_deg))

    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)

    # ── Dump key frames in world (root) frame ──
    frames_to_check = [
        "base_link",
        "Base",
        "top_base_link",
        "head_pan_link",
        "head_tilt_link",
        "head_camera_link",
        "head_camera_depth_frame",
        "head_camera_depth_optical_frame",
    ]
    print("=== Frame positions (world/root frame) ===")
    for fname in frames_to_check:
        fid = model.getFrameId(fname)
        oMf = data.oMf[fid]
        print(f"  {fname:40s}  pos={oMf.translation}  ")

    # ── Camera optical frame axes in world ──
    cam_fid = model.getFrameId(CAMERA_FRAME)
    oMcam = data.oMf[cam_fid]
    print(f"\nCamera optical frame rotation (world):\n{oMcam.rotation}")
    print(f"Camera Z-axis (forward) in world: {oMcam.rotation[:, 2]}")
    print(f"Camera X-axis (right)   in world: {oMcam.rotation[:, 0]}")
    print(f"Camera Y-axis (down)    in world: {oMcam.rotation[:, 1]}")

    # ── Base frame ──
    base_fid = model.getFrameId(BASE_FRAME)
    oMbase = data.oMf[base_fid]
    print(f"\nBase position (world): {oMbase.translation}")
    print(f"Base Z-axis (up) in world: {oMbase.rotation[:, 2]}")
    print(f"Base X-axis in world:      {oMbase.rotation[:, 0]}")
    print(f"Base Y-axis in world:      {oMbase.rotation[:, 1]}")

    # ── Build T_base_camera ──
    R = oMbase.rotation.T @ oMcam.rotation
    t = oMbase.rotation.T @ (oMcam.translation - oMbase.translation)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    print(f"\n=== T_base_camera ===")
    print(f"Camera position in Base frame: {t}")
    print(f"Camera forward (Z) in Base frame: {R[:, 2]}")

    # ── Transform centroid ──
    p_h = np.array([*camera_centroid, 1.0])
    p_base = (T @ p_h)[:3]
    print(f"\n=== Results ===")
    print(f"Camera centroid (optical): {camera_centroid}")
    print(f"Transformed to Base:     {p_base}")
    print(f"Expected (from EE):        {ee_base}")
    print(f"Error:                     {p_base - ee_base}")
    print(f"Error magnitude:           {np.linalg.norm(p_base - ee_base):.4f} m")

    # ── Sanity: where does the camera think the point is in world? ──
    p_world = oMcam.rotation @ camera_centroid + oMcam.translation
    print(f"\nObject in world frame (from camera): {p_world}")
    p_world_ee = oMbase.rotation @ ee_base + oMbase.translation
    print(f"EE in world frame (from measure_ee): {p_world_ee}")
    print(f"World frame error:                   {p_world - p_world_ee}")


if __name__ == "__main__":
    main()
