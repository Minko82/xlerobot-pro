#!/usr/bin/env python3
"""MuJoCo simulation for testing the IK solver and motor mapping pipeline.

Builds a MuJoCo model kinematically equivalent to the URDF arm 2 (right arm),
runs the IK solver, applies the control_single_bus.py motor mapping, and
visualizes the results.

Tests:
  1. FK comparison — Pinocchio vs MuJoCo (validates kinematic equivalence)
  2. IK solver — solve for targets, verify EE reaches them in MuJoCo
  3. Motor mapping — full mjcf_to_motor round-trip pipeline

Usage:
    conda activate lerobot
    cd cube-vision
    python mujoco_ik_test.py                          # run all tests
    python mujoco_ik_test.py --target 0 -0.30 0.10    # custom target (Base_2 frame)
    python mujoco_ik_test.py --viewer                  # interactive MuJoCo viewer
    python mujoco_ik_test.py --viewer --target 0 -0.3 0.1
"""

import sys
import argparse
import time
from pathlib import Path

import numpy as np
import mujoco

# Ensure cube-vision imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ik_solver.ik_so101 import IK_SO101

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# ═══════════════════════════════════════════════════════════════════════════
# MJCF model — kinematically equivalent to xlerobot.urdf arm 2 (right arm)
# Uses primitive geoms (capsules, spheres, boxes) instead of STL meshes.
# Joint definitions taken directly from the URDF:
#   Rotation_2, Pitch_2, Elbow_2, Wrist_Pitch_2, Wrist_Roll_2
# ═══════════════════════════════════════════════════════════════════════════

MJCF_XML = """
<mujoco model="so101_arm2">
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global offwidth="1280" offheight="960"/>
  </visual>

  <default>
    <geom contype="0" conaffinity="0"/>
    <joint damping="0.5"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker"
             rgb1="0.85 0.85 0.85" rgb2="0.65 0.65 0.65"
             width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8"/>
  </asset>

  <worldbody>
    <!-- Ground plane -->
    <geom type="plane" size="0.6 0.6 0.01" material="grid_mat"/>

    <!-- Table (visual only) -->
    <geom type="box" size="0.18 0.18 0.38" pos="-0.135 0.133 0.38"
          rgba="0.55 0.45 0.35 0.4"/>

    <!-- ── Arm 2 (right arm) ──────────────────────────────────── -->
    <!-- arm_base_joint_2 (fixed): xyz="-0.135 0.133 0.760" rpy="0 0 90°" -->
    <body name="Base_2" pos="-0.135 0.133 0.760" euler="0 0 90">
      <geom type="box" size="0.025 0.025 0.016" rgba="0.25 0.25 0.25 1"/>
      <site name="base_site" size="0.008" rgba="1 1 0 1"/>

      <!-- Rotation_2: xyz="0 -0.0452 0.0165" rpy="90 0 0" axis="0 -1 0" -->
      <body name="Rotation_Pitch_2" pos="0 -0.0452 0.0165" euler="90 0 0">
        <joint name="Rotation_2" type="hinge" axis="0 -1 0"
               range="-120.3 120.3"/>
        <geom type="capsule" fromto="0 0 0  0 0.1025 0.0306"
              size="0.014" rgba="0.92 0.92 0.92 1"/>
        <geom type="sphere" size="0.018" rgba="1 0.5 0 1"/>

        <!-- Pitch_2: xyz="0 0.1025 0.0306" rpy="90 0 0" axis="-1 0 0" -->
        <body name="Upper_Arm_2" pos="0 0.1025 0.0306" euler="90 0 0">
          <joint name="Pitch_2" type="hinge" axis="-1 0 0"
                 range="-5.7 197.7"/>
          <geom type="capsule" fromto="0 0 0  0 0.11257 0.028"
                size="0.013" rgba="0.88 0.88 0.88 1"/>
          <geom type="sphere" size="0.016" rgba="0 0.5 1 1"/>

          <!-- Elbow_2: xyz="0 0.11257 0.028" rpy="-90 0 0" axis="1 0 0" -->
          <body name="Lower_Arm_2" pos="0 0.11257 0.028" euler="-90 0 0">
            <joint name="Elbow_2" type="hinge" axis="1 0 0"
                   range="-11.5 180"/>
            <geom type="capsule" fromto="0 0 0  0 0.0052 0.1349"
                  size="0.012" rgba="0.84 0.84 0.84 1"/>
            <geom type="sphere" size="0.015" rgba="0 1 0.5 1"/>

            <!-- Wrist_Pitch_2: xyz="0 0.0052 0.1349" rpy="-90 0 0" axis="1 0 0" -->
            <body name="Wrist_Pitch_Roll_2" pos="0 0.0052 0.1349" euler="-90 0 0">
              <joint name="Wrist_Pitch_2" type="hinge" axis="1 0 0"
                     range="-103.1 103.1"/>
              <geom type="capsule" fromto="0 0 0  0 -0.0601 0"
                    size="0.010" rgba="0.80 0.80 0.80 1"/>
              <geom type="sphere" size="0.013" rgba="1 0 0.5 1"/>

              <!-- Wrist_Roll_2: xyz="0 -0.0601 0" rpy="0 90 0" axis="0 -1 0" -->
              <body name="Fixed_Jaw_2" pos="0 -0.0601 0" euler="0 90 0">
                <joint name="Wrist_Roll_2" type="hinge" axis="0 -1 0"
                       range="-180 180"/>
                <!-- Simplified gripper shape -->
                <geom type="box" size="0.016 0.022 0.005" pos="0 -0.012 0"
                      rgba="0.55 0.55 0.85 1"/>
                <geom type="box" size="0.003 0.018 0.005" pos="0.013 -0.038 0"
                      rgba="0.55 0.55 0.85 1"/>
                <geom type="box" size="0.003 0.018 0.005" pos="-0.013 -0.038 0"
                      rgba="0.55 0.55 0.85 1"/>
                <!-- EE site at body-frame origin (= Wrist_Roll joint frame) -->
                <site name="ee_site" pos="0 0 0" size="0.009" rgba="1 0 0 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Target sphere (mocap — user-positionable, no dynamics) -->
    <body name="target" mocap="true">
      <geom type="sphere" size="0.012" rgba="0 1 0 0.8"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

# Joint names in qpos order (must match XML definition order)
JOINT_NAMES = ["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2"]


# ═══════════════════════════════════════════════════════════════════════════
# Motor mapping (copied from control_single_bus.py)
# ═══════════════════════════════════════════════════════════════════════════

def mjcf_to_motor(q_deg: np.ndarray) -> np.ndarray:
    """MJCF/URDF joint angles (deg) → motor command (deg). Right-arm convention."""
    out = q_deg.copy()
    out[0] = -out[0]            # Rotation:    negated (mirrored)
    out[1] = out[1] - 90.0      # Pitch_2  →  shoulder_lift
    out[2] = 90.0 - out[2]      # Elbow_2  →  elbow_flex
    out[3] = -out[3]            # Wrist_Pitch: negated
    out[4] = -out[4]            # Wrist_Roll:  negated
    return out


def motor_to_mjcf(m_deg: np.ndarray) -> np.ndarray:
    """Motor command (deg) → MJCF/URDF joint angles (deg). Inverse of above."""
    q = m_deg.copy()
    q[0] = -m_deg[0]
    q[1] = m_deg[1] + 90.0
    q[2] = 90.0 - m_deg[2]
    q[3] = -m_deg[3]
    q[4] = -m_deg[4]
    return q


# ═══════════════════════════════════════════════════════════════════════════
# MuJoCo helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_model():
    """Compile MJCF and create simulation data."""
    model = mujoco.MjModel.from_xml_string(MJCF_XML)
    data = mujoco.MjData(model)
    return model, data


def get_ee_pos(model, data):
    """World position of the Fixed_Jaw_2 frame origin."""
    return data.site_xpos[model.site("ee_site").id].copy()


def get_base_pos(model, data):
    """World position of the Base_2 frame origin."""
    return data.site_xpos[model.site("base_site").id].copy()


def set_arm_joints(data, q_rad):
    """Set arm joint angles from a 5-element radians array."""
    data.qpos[:5] = q_rad


def set_target(model, data, world_pos):
    """Position the green target sphere (mocap body)."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    mocap_id = model.body_mocapid[body_id]
    data.mocap_pos[mocap_id] = world_pos


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: FK comparison — Pinocchio vs MuJoCo
# ═══════════════════════════════════════════════════════════════════════════

def test_fk_comparison(model, data, ik):
    """Verify kinematic equivalence between Pinocchio and MuJoCo."""
    import pinocchio as pin

    print("\n" + "=" * 65)
    print(" TEST 1: FK Comparison — Pinocchio vs MuJoCo")
    print("=" * 65)

    configs_deg = [
        [0, 0, 0, 0, 0],
        [30, 0, 0, 0, 0],
        [0, 45, 0, 0, 0],
        [0, 0, 90, 0, 0],
        [0, 0, 0, 45, 0],
        [0, 0, 0, 0, 90],
        [0, 90, 90, 0, 0],
        [-30, 45, 60, -20, 45],
        [0, 135, 135, 45, 0],
    ]

    ee_fid = ik.model.getFrameId("Fixed_Jaw_2")
    all_pass = True

    for cfg_deg in configs_deg:
        q_rad = np.deg2rad(cfg_deg).astype(float)
        q_rad = np.clip(q_rad, ik.model.lowerPositionLimit,
                        ik.model.upperPositionLimit)

        # Pinocchio FK
        pin.forwardKinematics(ik.model, ik.data, q_rad)
        pin.updateFramePlacements(ik.model, ik.data)
        pin_ee = ik.data.oMf[ee_fid].translation.copy()

        # MuJoCo FK
        set_arm_joints(data, q_rad)
        mujoco.mj_forward(model, data)
        mj_ee = get_ee_pos(model, data)

        err_mm = np.linalg.norm(pin_ee - mj_ee) * 1000
        ok = err_mm < 2.0
        if not ok:
            all_pass = False

        q_deg_str = ", ".join(f"{d:6.1f}" for d in np.rad2deg(q_rad))
        print(f"  q=[{q_deg_str}]°  err={err_mm:5.1f}mm  "
              f"{'PASS' if ok else 'FAIL'}")

    return all_pass


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: IK solver
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_TARGETS = [
    ("Forward",      [0.0, -0.30,  0.01]),
    ("Forward+up",   [0.0, -0.20,  0.20]),
    ("Forward+down", [0.0, -0.20, -0.05]),
    ("Left",         [0.15, -0.20,  0.05]),
    ("Right",        [-0.15, -0.20, 0.05]),
    ("Close",        [0.0, -0.15,  0.10]),
]


def test_ik_targets(model, data, ik, targets=None):
    """Solve IK for each target, set MuJoCo joints, verify EE position."""
    print("\n" + "=" * 65)
    print(" TEST 2: IK Solver → MuJoCo FK Verification")
    print("=" * 65)

    targets = targets or DEFAULT_TARGETS
    all_pass = True
    results = []

    for name, tgt_b2 in targets:
        traj = ik.generate_ik(tgt_b2, [0, 0, 0])

        if not traj:
            print(f"  {name:15s}  IK FAILED (no trajectory)")
            all_pass = False
            continue

        q_rad = traj[-1]
        tgt_w = ik.base2_to_world(tgt_b2)

        # MuJoCo FK
        set_arm_joints(data, q_rad)
        mujoco.mj_forward(model, data)
        mj_ee = get_ee_pos(model, data)

        err_mm = np.linalg.norm(mj_ee - tgt_w) * 1000
        ok = err_mm < 5.0
        if not ok:
            all_pass = False

        results.append((name, tgt_b2, q_rad, tgt_w, mj_ee, err_mm))
        q_str = ", ".join(f"{d:6.1f}" for d in np.rad2deg(q_rad))
        print(f"  {name:15s}  q=[{q_str}]°  "
              f"ee_err={err_mm:5.1f}mm  {'PASS' if ok else 'FAIL'}")

    return all_pass, results


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Motor mapping round-trip
# ═══════════════════════════════════════════════════════════════════════════

def test_motor_mapping(model, data, ik, targets=None):
    """IK → mjcf_to_motor → motor_to_mjcf → MuJoCo FK. Verifies full pipeline."""
    print("\n" + "=" * 65)
    print(" TEST 3: Motor Mapping Pipeline (mjcf_to_motor round-trip)")
    print("=" * 65)

    targets = targets or DEFAULT_TARGETS[:4]
    all_pass = True

    for name, tgt_b2 in targets:
        traj = ik.generate_ik(tgt_b2, [0, 0, 0])
        if not traj:
            print(f"  {name}: IK failed")
            all_pass = False
            continue

        q_ik_rad = traj[-1]
        q_ik_deg = np.rad2deg(q_ik_rad)

        # Full pipeline from control_single_bus.py
        motor_deg = mjcf_to_motor(q_ik_deg)
        q_rt_deg = motor_to_mjcf(motor_deg)
        q_rt_rad = np.deg2rad(q_rt_deg)

        # Joint-level round-trip error
        joint_err_max = np.max(np.abs(q_rt_deg - q_ik_deg))

        # MuJoCo FK with round-tripped joints
        set_arm_joints(data, q_rt_rad)
        mujoco.mj_forward(model, data)
        mj_ee = get_ee_pos(model, data)
        tgt_w = ik.base2_to_world(tgt_b2)
        ee_err_mm = np.linalg.norm(mj_ee - tgt_w) * 1000

        ok = ee_err_mm < 5.0 and joint_err_max < 0.1
        if not ok:
            all_pass = False

        print(f"  {name:15s}  "
              f"joint_roundtrip_maxerr={joint_err_max:.3f}°  "
              f"ee_err={ee_err_mm:.1f}mm  {'PASS' if ok else 'FAIL'}")
        print(f"    IK  (deg): [{', '.join(f'{d:7.2f}' for d in q_ik_deg)}]")
        print(f"    Motor cmd: [{', '.join(f'{d:7.2f}' for d in motor_deg)}]")
        print(f"    Round-trip: [{', '.join(f'{d:7.2f}' for d in q_rt_deg)}]")

    return all_pass


# ═══════════════════════════════════════════════════════════════════════════
# Rendering (matplotlib fallback — works on WSL / headless)
# ═══════════════════════════════════════════════════════════════════════════

def render_results(model, data, ik, results):
    """Render a matplotlib figure showing the arm at each IK solution."""
    import pinocchio as pin
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    if n == 0:
        return
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(6 * cols, 5.5 * rows))
    fig.suptitle("MuJoCo IK Test — Arm 2 (Right Arm)", fontsize=14, y=0.98)

    # Collect joint positions for arm skeleton using Pinocchio FK
    frame_names = [
        "Base_2", "Rotation_Pitch_2", "Upper_Arm_2", "Lower_Arm_2",
        "Wrist_Pitch_Roll_2", "Fixed_Jaw_2",
    ]
    frame_ids = [ik.model.getFrameId(n) for n in frame_names]

    for idx, (name, tgt_b2, q_rad, tgt_w, mj_ee, err_mm) in enumerate(results):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")

        # Pinocchio FK for arm skeleton
        pin.forwardKinematics(ik.model, ik.data, q_rad)
        pin.updateFramePlacements(ik.model, ik.data)
        joint_pos = np.array([ik.data.oMf[fid].translation for fid in frame_ids])

        # Draw arm skeleton
        ax.plot(joint_pos[:, 0], joint_pos[:, 1], joint_pos[:, 2],
                "o-", color="steelblue", linewidth=2.5, markersize=5,
                label="Arm skeleton")

        # Draw target
        ax.scatter(*tgt_w, c="lime", s=120, marker="*", edgecolors="green",
                   linewidths=1, zorder=5, label="Target")

        # Draw MuJoCo EE
        ax.scatter(*mj_ee, c="red", s=60, marker="^", zorder=5,
                   label=f"MuJoCo EE ({err_mm:.1f}mm)")

        # Draw base site
        ax.scatter(*joint_pos[0], c="gold", s=80, marker="s", zorder=5,
                   label="Base_2")

        # Line from EE to target
        ax.plot([mj_ee[0], tgt_w[0]], [mj_ee[1], tgt_w[1]],
                [mj_ee[2], tgt_w[2]], "r--", alpha=0.5, linewidth=1)

        ax.set_title(f"{name}\nerr={err_mm:.1f}mm", fontsize=10)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend(fontsize=7, loc="upper left")

        # Equal axes
        all_pts = np.vstack([joint_pos, [tgt_w], [mj_ee]])
        mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
        span = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max(), 0.15) / 2 * 1.3
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "mujoco_ik_test.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"\nSaved visualization to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Interactive viewer
# ═══════════════════════════════════════════════════════════════════════════

def run_viewer(model, data, ik, target_b2=None):
    """Launch the MuJoCo interactive viewer with IK solution animated."""
    if target_b2 is None:
        target_b2 = [0.0, -0.30, 0.01]

    print(f"\nSolving IK for target (Base_2 frame): {target_b2}")
    traj = ik.generate_ik(target_b2, [0, 0, 0])

    if not traj:
        print("IK failed — cannot launch viewer.")
        return

    q_final = traj[-1]
    tgt_w = ik.base2_to_world(target_b2)

    # Set target sphere position
    set_target(model, data, tgt_w)

    # Start at neutral
    set_arm_joints(data, np.zeros(5))
    mujoco.mj_forward(model, data)

    q_deg = np.rad2deg(q_final)
    mj_ee = get_ee_pos(model, data)
    print(f"  IK joints (deg): [{', '.join(f'{d:.1f}' for d in q_deg)}]")
    print(f"  Motor commands:  [{', '.join(f'{d:.1f}' for d in mjcf_to_motor(q_deg))}]")
    print(f"  Trajectory: {len(traj)} steps")
    print(f"\nLaunching viewer... (close window to exit)")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Animate trajectory from neutral to IK solution
            for q_step in traj:
                if not viewer.is_running():
                    break
                set_arm_joints(data, q_step)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.02)

            # Report final EE error
            mj_ee = get_ee_pos(model, data)
            err_mm = np.linalg.norm(mj_ee - tgt_w) * 1000
            print(f"  Final EE error: {err_mm:.1f}mm")

            # Keep viewer open
            while viewer.is_running():
                time.sleep(0.1)

    except Exception as e:
        print(f"\nViewer failed: {e}")
        print("  (This may happen in headless / WSL environments without display)")
        print("  Run without --viewer to get test results + PNG output instead.")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo test for SO-101 IK solver & motor mapping")
    parser.add_argument(
        "--target", "-t", nargs=3, type=float, metavar=("X", "Y", "Z"),
        help="Target in Base_2 frame [x, y, z] meters (default: 0 -0.30 0.01)")
    parser.add_argument(
        "--viewer", action="store_true",
        help="Launch interactive MuJoCo viewer (requires display)")
    args = parser.parse_args()

    # ── Load models ───────────────────────────────────────────────────
    print("Loading MuJoCo model...")
    model, data = load_model()

    print(f"MuJoCo model: {model.njnt} joints, {model.nq} DOF, "
          f"{model.nmocap} mocap bodies")
    for i in range(model.njnt):
        jnt = model.joint(i)
        lo, hi = np.rad2deg(jnt.range)
        print(f"  [{i}] {jnt.name:20s}  range=[{lo:7.1f}, {hi:7.1f}]°")

    print("\nLoading Pinocchio IK solver...")
    ik = IK_SO101()
    print(f"  Pinocchio reduced model: {ik.model.nq} DOF")
    print(f"  Base_2 world pos: [{ik._base2_t[0]:.4f}, "
          f"{ik._base2_t[1]:.4f}, {ik._base2_t[2]:.4f}]")

    # ── Viewer mode ───────────────────────────────────────────────────
    if args.viewer:
        target = args.target if args.target else None
        run_viewer(model, data, ik, target)
        return

    # ── Test mode ─────────────────────────────────────────────────────
    custom_target = None
    if args.target:
        custom_target = [("Custom", args.target)]

    pass1 = test_fk_comparison(model, data, ik)

    targets2 = custom_target or None
    pass2, results = test_ik_targets(model, data, ik, targets2)

    targets3 = custom_target or None
    pass3 = test_motor_mapping(model, data, ik, targets3)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    overall = pass1 and pass2 and pass3
    print(f" OVERALL: {'ALL PASS' if overall else 'SOME FAILED'}")
    print(f"   FK comparison:  {'PASS' if pass1 else 'FAIL'}")
    print(f"   IK targets:     {'PASS' if pass2 else 'FAIL'}")
    print(f"   Motor mapping:  {'PASS' if pass3 else 'FAIL'}")
    print("=" * 65)

    # ── Render visualization ──────────────────────────────────────────
    if results:
        render_results(model, data, ik, results)

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
