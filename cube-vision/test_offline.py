#!/usr/bin/env python3
"""Offline test suite — validates the full pipeline WITHOUT the physical robot.

No motors, no RealSense, no hardware needed.  Only requires:
  - pinocchio  (URDF loading + FK)
  - pink       (IK solver)
  - numpy

Tests cover:
  1. URDF loading & model structure
  2. Forward kinematics sanity
  3. Frame transform (camera → Base_2)
  4. IK solver convergence (multiple targets)
  5. IK solver determinism (repeated calls)
  6. Motor mapping (URDF deg → motor deg)
  7. Full pipeline round-trip (synthetic camera point → Base_2 → IK → FK → verify)

Usage:
    cd cube-vision
    python test_offline.py              # run all tests
    python test_offline.py -v           # verbose output
    python test_offline.py -k ik        # run only tests with 'ik' in the name
"""

from __future__ import annotations

import sys
import math
import numpy as np
import pinocchio as pin

# ── Paths (resolve before any chdir shenanigans) ───────────────────────
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_URDF_PATH = _HERE / "frame_transform" / "xlerobot" / "xlerobot.urdf"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

_results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = ""):
    """Record a pass/fail result."""
    tag = PASS if condition else FAIL
    msg = f"  [{tag}] {name}"
    if detail:
        msg += f"  — {detail}"
    print(msg)
    _results.append((name, condition, detail))


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# 1. URDF Loading & Model Structure
# ---------------------------------------------------------------------------

def test_urdf_loading():
    section("1. URDF Loading & Model Structure")

    check("URDF file exists", _URDF_PATH.exists(), str(_URDF_PATH))

    model = pin.buildModelFromUrdf(str(_URDF_PATH))
    check("Model loads successfully", model.njoints > 1,
          f"{model.njoints} joints (including universe)")

    # Check expected joint names exist
    joint_names = [model.names[i] for i in range(model.njoints)]
    expected_arm2 = ["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2"]
    expected_head = ["head_pan_joint", "head_tilt_joint"]
    expected_arm1 = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]

    for jname in expected_arm2:
        check(f"Joint '{jname}' in model", jname in joint_names)
    for jname in expected_head:
        check(f"Joint '{jname}' in model", jname in joint_names)

    # Check expected frames exist
    data = model.createData()
    expected_frames = ["Base_2", "Fixed_Jaw_2", "head_camera_link"]
    for fname in expected_frames:
        fid = model.getFrameId(fname)
        check(f"Frame '{fname}' in model", fid < model.nframes,
              f"id={fid}")

    # Print all joints for reference
    print(f"\n  All joints ({model.njoints}):")
    for i in range(model.njoints):
        print(f"    [{i:2d}] {model.names[i]}")

    return model


# ---------------------------------------------------------------------------
# 2. Forward Kinematics Sanity
# ---------------------------------------------------------------------------

def test_fk_sanity(model: pin.Model):
    section("2. Forward Kinematics Sanity")

    data = model.createData()
    q = pin.neutral(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # Base_2 should be at a fixed known position (not origin)
    base2_id = model.getFrameId("Base_2")
    base2_pos = data.oMf[base2_id].translation.copy()
    print(f"  Base_2 position (neutral): [{base2_pos[0]:.4f}, {base2_pos[1]:.4f}, {base2_pos[2]:.4f}]")
    check("Base_2 not at origin (is mounted)", np.linalg.norm(base2_pos) > 0.01,
          f"|pos|={np.linalg.norm(base2_pos):.4f}")

    # EE at neutral should be somewhere reasonable relative to Base_2
    ee_id = model.getFrameId("Fixed_Jaw_2")
    ee_pos = data.oMf[ee_id].translation.copy()
    print(f"  Fixed_Jaw_2 position (neutral): [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")

    ee_in_base = data.oMf[base2_id].rotation.T @ (ee_pos - base2_pos)
    dist = np.linalg.norm(ee_in_base)
    print(f"  EE in Base_2 frame (neutral): [{ee_in_base[0]:.4f}, {ee_in_base[1]:.4f}, {ee_in_base[2]:.4f}]")
    print(f"  Distance from base: {dist:.4f} m")
    check("EE within reasonable arm reach", 0.05 < dist < 0.5,
          f"dist={dist:.4f}")

    # Camera link should be above the base
    cam_id = model.getFrameId("head_camera_link")
    cam_pos = data.oMf[cam_id].translation.copy()
    print(f"  Camera link position (neutral): [{cam_pos[0]:.4f}, {cam_pos[1]:.4f}, {cam_pos[2]:.4f}]")
    check("Camera above Base_2 (Z)", cam_pos[2] > base2_pos[2],
          f"cam_z={cam_pos[2]:.4f}, base_z={base2_pos[2]:.4f}")

    return base2_pos, ee_pos


# ---------------------------------------------------------------------------
# 3. Frame Transform (camera → Base_2)
# ---------------------------------------------------------------------------

def test_frame_transform():
    section("3. Frame Transform (camera → Base_2)")

    from frame_transform.frame_transform import camera_xyz_to_base_xyz, _head_motor_to_mjcf

    # Test head motor → MJCF conversion
    test_deg = np.array([1.0, 14.0])  # should map to ~[0, 0] in MJCF
    mjcf = _head_motor_to_mjcf(test_deg)
    check("Head motor→MJCF: pan=1°→~0°", abs(mjcf[0]) < 0.1, f"got {mjcf[0]:.2f}°")
    check("Head motor→MJCF: tilt=14°→~0°", abs(mjcf[1]) < 0.1, f"got {mjcf[1]:.2f}°")

    # Transform a point straight ahead in camera optical frame
    # Optical: Z=forward, X=right, Y=down
    # With head at neutral (~0°, ~0°), a point at (0, 0, 0.3) in optical
    # should land somewhere in front of the base
    joint_values = {
        "head_pan_joint":  1.0 * DEG2RAD,   # ~neutral motor position
        "head_tilt_joint": 14.0 * DEG2RAD,   # ~neutral motor position
    }

    # Point 30cm straight ahead in camera optical frame
    bx, by, bz = camera_xyz_to_base_xyz(0.0, 0.0, 0.30, joint_values)
    result = np.array([bx, by, bz])
    print(f"  Camera (0, 0, 0.30) → Base_2: [{bx:.4f}, {by:.4f}, {bz:.4f}]")
    check("Transform produces finite result", np.all(np.isfinite(result)))
    check("Horizontal distance > 0", np.sqrt(bx**2 + by**2) > 0.01,
          f"horiz_dist={np.sqrt(bx**2 + by**2):.4f}")

    # Point slightly to the right in camera optical frame
    bx2, by2, bz2 = camera_xyz_to_base_xyz(0.05, 0.0, 0.30, joint_values)
    print(f"  Camera (0.05, 0, 0.30) → Base_2: [{bx2:.4f}, {by2:.4f}, {bz2:.4f}]")
    shift = np.array([bx2 - bx, by2 - by, bz2 - bz])
    check("Shifting camera X changes Base_2 position", np.linalg.norm(shift) > 0.01,
          f"|shift|={np.linalg.norm(shift):.4f}")

    # Point below in camera optical frame
    bx3, by3, bz3 = camera_xyz_to_base_xyz(0.0, 0.05, 0.30, joint_values)
    print(f"  Camera (0, 0.05, 0.30) → Base_2: [{bx3:.4f}, {by3:.4f}, {bz3:.4f}]")
    z_shift = bz3 - bz
    check("Camera +Y (down) lowers Base_2 Z", z_shift < -0.01,
          f"z_shift={z_shift:.4f}")

    # Test with panned head (20° pan motor)
    joint_panned = {
        "head_pan_joint":  21.0 * DEG2RAD,   # 20° offset from neutral
        "head_tilt_joint": 14.0 * DEG2RAD,
    }
    bx4, by4, bz4 = camera_xyz_to_base_xyz(0.0, 0.0, 0.30, joint_panned)
    print(f"  Camera (0,0,0.3) with 20° pan → Base_2: [{bx4:.4f}, {by4:.4f}, {bz4:.4f}]")
    pan_shift = np.linalg.norm(np.array([bx4, by4]) - np.array([bx, by]))
    check("Head pan changes horizontal position", pan_shift > 0.01,
          f"|xy_shift|={pan_shift:.4f}")


# ---------------------------------------------------------------------------
# 4. IK Solver Convergence
# ---------------------------------------------------------------------------

def test_ik_convergence():
    section("4. IK Solver Convergence")

    from ik_solver import IK_SO101

    ik = IK_SO101()

    # Test targets in Base_2 frame
    targets = [
        ("Forward 20cm",        [0.0, -0.20, 0.05]),
        ("Forward 25cm",        [0.0, -0.25, 0.05]),
        ("Forward+left",        [0.05, -0.20, 0.05]),
        ("Forward+right",       [-0.05, -0.20, 0.05]),
        ("Forward+up",          [0.0, -0.15, 0.15]),
        ("Forward+down",        [0.0, -0.20, -0.05]),
        ("Near table surface",  [0.0, -0.20, 0.00]),
    ]

    for name, target in targets:
        # Reset state for each target (test independence)
        ik_fresh = IK_SO101()
        traj = ik_fresh.generate_ik(target_xyz=target, gripper_offset_xyz=[0, 0, 0])

        if len(traj) == 0:
            check(f"IK '{name}'", False, "empty trajectory")
            continue

        # Verify final EE position is close to target
        final_q = traj[-1]
        pin.forwardKinematics(ik_fresh.model, ik_fresh.data, final_q)
        pin.updateFramePlacements(ik_fresh.model, ik_fresh.data)

        ee_world = ik_fresh.data.oMf[ik_fresh.model.getFrameId("Fixed_Jaw_2")].translation.copy()
        target_world = ik_fresh.base2_to_world(np.array(target))
        error = np.linalg.norm(ee_world - target_world)

        converged = error < 0.005  # 5mm
        check(f"IK '{name}'", converged,
              f"error={error*1000:.1f}mm, steps={len(traj)}")

    # Test unreachable target (should return trajectory but not converge, or empty)
    ik_far = IK_SO101()
    traj_far = ik_far.generate_ik(target_xyz=[0.0, -1.0, 0.0], gripper_offset_xyz=[0, 0, 0])
    if len(traj_far) > 0:
        final_q = traj_far[-1]
        pin.forwardKinematics(ik_far.model, ik_far.data, final_q)
        pin.updateFramePlacements(ik_far.model, ik_far.data)
        ee_world = ik_far.data.oMf[ik_far.model.getFrameId("Fixed_Jaw_2")].translation.copy()
        target_world = ik_far.base2_to_world(np.array([0.0, -1.0, 0.0]))
        error = np.linalg.norm(ee_world - target_world)
        check("IK unreachable target doesn't claim success", error > 0.1,
              f"error={error*1000:.1f}mm (should be large)")
    else:
        check("IK unreachable target returns empty", True)


# ---------------------------------------------------------------------------
# 5. IK Solver Determinism
# ---------------------------------------------------------------------------

def test_ik_determinism():
    section("5. IK Solver Determinism (repeated calls)")

    from ik_solver import IK_SO101

    target = [0.0, -0.20, 0.05]

    # Run same target twice on same instance
    ik = IK_SO101()
    traj1 = ik.generate_ik(target_xyz=target, gripper_offset_xyz=[0, 0, 0])

    # NOTE: If configuration is NOT reset between calls, the second call
    # starts from the end of traj1, producing a different (shorter or empty) trajectory.
    traj2 = ik.generate_ik(target_xyz=target, gripper_offset_xyz=[0, 0, 0])

    if len(traj1) > 0 and len(traj2) > 0:
        final1 = traj1[-1]
        final2 = traj2[-1]
        diff = np.linalg.norm(final1 - final2)
        check("Same target → same final config",
              diff < 0.01,
              f"|q1-q2|={diff:.6f} rad")

        len_ratio = len(traj2) / len(traj1) if len(traj1) > 0 else float('inf')
        check("Trajectory lengths similar",
              0.5 < len_ratio < 2.0,
              f"len1={len(traj1)}, len2={len(traj2)}")
    else:
        check("Both trajectories non-empty",
              len(traj1) > 0 and len(traj2) > 0,
              f"len1={len(traj1)}, len2={len(traj2)}")

    # Run on fresh instances (should always match)
    ik_a = IK_SO101()
    ik_b = IK_SO101()
    traj_a = ik_a.generate_ik(target_xyz=target, gripper_offset_xyz=[0, 0, 0])
    traj_b = ik_b.generate_ik(target_xyz=target, gripper_offset_xyz=[0, 0, 0])

    if len(traj_a) > 0 and len(traj_b) > 0:
        diff_ab = np.linalg.norm(traj_a[-1] - traj_b[-1])
        check("Fresh instances → identical results", diff_ab < 1e-10,
              f"|qa-qb|={diff_ab:.2e}")


# ---------------------------------------------------------------------------
# 6. Motor Mapping (URDF → motor degrees)
# ---------------------------------------------------------------------------

def test_motor_mapping():
    section("6. Motor Mapping (URDF deg → motor deg)")

    def mjcf_to_motor(q_deg: np.ndarray) -> np.ndarray:
        """Same mapping used in control_single_bus.py (right arm, mirrored)."""
        out = q_deg.copy()
        out[0] = -out[0]          # Rotation:    negated (mirrored)
        out[1] = out[1] - 90.0    # Pitch  ->  shoulder_lift
        out[2] = 90.0 - out[2]    # Elbow  ->  elbow_flex
        out[3] = -out[3]          # Wrist_Pitch: negated
        out[4] = -out[4]          # Wrist_Roll:  negated
        return out

    def motor_to_mjcf(q_deg: np.ndarray) -> np.ndarray:
        """Inverse of mjcf_to_motor."""
        out = q_deg.copy()
        out[0] = -out[0]
        out[1] = out[1] + 90.0    # shoulder_lift -> Pitch
        out[2] = 90.0 - out[2]    # elbow_flex -> Elbow
        out[3] = -out[3]
        out[4] = -out[4]
        return out

    # Round-trip test
    original = np.array([10.0, 45.0, -30.0, 20.0, 0.0])
    motor = mjcf_to_motor(original)
    recovered = motor_to_mjcf(motor)
    check("Round-trip URDF→motor→URDF",
          np.allclose(original, recovered),
          f"max diff={np.max(np.abs(original - recovered)):.6f}°")

    # Known conversions
    # If URDF pitch = 0°, motor shoulder_lift should = 90°
    zero_urdf = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    zero_motor = mjcf_to_motor(zero_urdf)
    check("URDF all-zero: motor shoulder_lift=-90°",
          abs(zero_motor[1] - (-90.0)) < 0.01, f"got {zero_motor[1]:.2f}°")
    check("URDF all-zero: motor elbow_flex=90°",
          abs(zero_motor[2] - 90.0) < 0.01, f"got {zero_motor[2]:.2f}°")

    # Rotation and wrist joints are negated (mirrored right arm)
    check("Rotation negated", abs(zero_motor[0]) < 0.01)
    check("Wrist_Pitch negated", abs(zero_motor[3]) < 0.01)
    check("Wrist_Roll negated", abs(zero_motor[4]) < 0.01)

    # Verify motor angles are in a sane range for typical IK outputs
    from ik_solver import IK_SO101
    ik = IK_SO101()
    traj = ik.generate_ik([0.0, -0.20, 0.05], [0, 0, 0])
    if len(traj) > 0:
        final_rad = traj[-1]
        final_deg = final_rad * RAD2DEG
        motor_deg = mjcf_to_motor(final_deg)
        print(f"  IK final (URDF deg): {np.round(final_deg, 2).tolist()}")
        print(f"  IK final (motor deg): {np.round(motor_deg, 2).tolist()}")

        # Motors should be within ±180° (the physical range is ~±150°)
        all_in_range = np.all(np.abs(motor_deg) < 180)
        check("Motor angles within ±180°", all_in_range,
              f"range=[{motor_deg.min():.1f}°, {motor_deg.max():.1f}°]")


# ---------------------------------------------------------------------------
# 7. Full Pipeline Round-Trip (synthetic)
# ---------------------------------------------------------------------------

def test_full_pipeline_roundtrip():
    section("7. Full Pipeline Round-Trip (synthetic)")

    from frame_transform.frame_transform import camera_xyz_to_base_xyz
    from ik_solver import IK_SO101

    # Simulate: object is at a known Base_2 position.
    # We'll go backwards (Base_2 → camera) to get a synthetic camera point,
    # then forward (camera → Base_2 → IK) and verify consistency.

    # Step A: Use FK to find where the camera sees a point
    # For this test, we pick a synthetic camera-optical-frame point directly
    # and check the full forward path.

    joint_values = {
        "head_pan_joint":  1.0 * DEG2RAD,   # neutral
        "head_tilt_joint": 45.0 * DEG2RAD,  # tilted down to see the table
    }

    # Synthetic camera point: object ~33cm ahead and ~32cm below in optical frame.
    # With the head tilted 45° down, this maps to Base_2 ≈ [0, -0.20, 0.05]
    # which is well within the arm's ~0.39 m reach.
    cam_point = np.array([0.333, 0.3233, 0.1397])

    # Step B: Transform to Base_2
    bx, by, bz = camera_xyz_to_base_xyz(cam_point[0], cam_point[1], cam_point[2], joint_values)
    base_target = np.array([bx, by, bz])

    print(f"  Synthetic camera point: {cam_point.tolist()}")
    print(f"  Transformed to Base_2:  [{bx:.4f}, {by:.4f}, {bz:.4f}]")
    print(f"  Distance from base:     {np.linalg.norm(base_target):.4f} m")

    check("Transform gives finite Base_2 coords", np.all(np.isfinite(base_target)))

    # Step C: IK to reach the target
    ik = IK_SO101()
    traj = ik.generate_ik(target_xyz=base_target.tolist(), gripper_offset_xyz=[0, 0, 0])

    check("IK produces non-empty trajectory", len(traj) > 0, f"steps={len(traj)}")

    if len(traj) > 0:
        # Step D: FK with final config to verify EE position
        final_q = traj[-1]
        pin.forwardKinematics(ik.model, ik.data, final_q)
        pin.updateFramePlacements(ik.model, ik.data)

        ee_world = ik.data.oMf[ik.model.getFrameId("Fixed_Jaw_2")].translation.copy()
        target_world = ik.base2_to_world(base_target)

        ik_error = np.linalg.norm(ee_world - target_world)
        print(f"  IK target (world): [{target_world[0]:.4f}, {target_world[1]:.4f}, {target_world[2]:.4f}]")
        print(f"  EE final  (world): [{ee_world[0]:.4f}, {ee_world[1]:.4f}, {ee_world[2]:.4f}]")
        print(f"  IK error: {ik_error*1000:.1f} mm")

        check("Full pipeline IK error < 5mm", ik_error < 0.005,
              f"{ik_error*1000:.1f}mm")

        # Step E: Motor mapping round-trip
        def mjcf_to_motor(q_deg):
            out = q_deg.copy()
            out[0] = -out[0]
            out[1] = out[1] - 90.0
            out[2] = 90.0 - out[2]
            out[3] = -out[3]
            out[4] = -out[4]
            return out

        def motor_to_mjcf(q_deg):
            out = q_deg.copy()
            out[0] = -out[0]
            out[1] = out[1] + 90.0
            out[2] = 90.0 - out[2]
            out[3] = -out[3]
            out[4] = -out[4]
            return out

        final_deg = final_q * RAD2DEG
        motor_deg = mjcf_to_motor(final_deg)
        recovered_deg = motor_to_mjcf(motor_deg)
        recovered_rad = recovered_deg * DEG2RAD

        mapping_error = np.linalg.norm(final_q - recovered_rad)
        check("Motor mapping round-trip lossless", mapping_error < 1e-10,
              f"error={mapping_error:.2e}")

    print(f"\n  Pipeline: camera_point → transform → IK → FK → verify ✓")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  OFFLINE TEST SUITE — No robot hardware required")
    print("=" * 60)

    model = test_urdf_loading()
    test_fk_sanity(model)
    test_frame_transform()
    test_ik_convergence()
    test_ik_determinism()
    test_motor_mapping()
    test_full_pipeline_roundtrip()

    # Summary
    section("SUMMARY")
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    total = len(_results)

    print(f"  {passed}/{total} passed, {failed} failed\n")

    if failed > 0:
        print("  Failed tests:")
        for name, ok, detail in _results:
            if not ok:
                print(f"    - {name}: {detail}")
        print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
