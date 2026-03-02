#!/usr/bin/env python3
"""Compare OLD vs NEW mjcf_to_motor mapping using MuJoCo FK.

Simulates the full control_single_bus.py pipeline:
  1. IK solver produces URDF joint angles (radians)
  2. mjcf_to_motor converts to motor commands (degrees)
  3. The physical motor receives those commands
  4. We model "what the physical right arm actually does" by inverting
     the motor command back to URDF angles with the NEW (correct) inverse,
     since the real motor IS wired with the new convention.

For the OLD mapping, the motor commands are wrong for the right arm, so
when the real motor interprets them, the arm goes to the wrong pose.

Produces a side-by-side visualization saved to outputs/old_vs_new_mapping.png.

Usage:
    conda activate lerobot
    cd cube-vision
    python mujoco_mapping_compare.py
"""

import sys
from pathlib import Path

import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ik_solver.ik_so101 import IK_SO101

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# ═══════════════════════════════════════════════════════════════════════════
# MJCF Model (arm 2 / right arm — identical to mujoco_ik_test.py)
# ═══════════════════════════════════════════════════════════════════════════

MJCF_XML = """
<mujoco model="so101_arm2">
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>
  <default>
    <geom contype="0" conaffinity="0"/>
    <joint damping="0.5"/>
  </default>
  <worldbody>
    <geom type="plane" size="0.6 0.6 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="Base_2" pos="-0.135 0.133 0.760" euler="0 0 90">
      <geom type="box" size="0.025 0.025 0.016" rgba="0.25 0.25 0.25 1"/>
      <site name="base_site" size="0.008" rgba="1 1 0 1"/>
      <body name="Rotation_Pitch_2" pos="0 -0.0452 0.0165" euler="90 0 0">
        <joint name="Rotation_2" type="hinge" axis="0 -1 0" range="-120.3 120.3"/>
        <geom type="capsule" fromto="0 0 0 0 0.1025 0.0306" size="0.014" rgba="0.92 0.92 0.92 1"/>
        <body name="Upper_Arm_2" pos="0 0.1025 0.0306" euler="90 0 0">
          <joint name="Pitch_2" type="hinge" axis="-1 0 0" range="-5.7 197.7"/>
          <geom type="capsule" fromto="0 0 0 0 0.11257 0.028" size="0.013" rgba="0.88 0.88 0.88 1"/>
          <body name="Lower_Arm_2" pos="0 0.11257 0.028" euler="-90 0 0">
            <joint name="Elbow_2" type="hinge" axis="1 0 0" range="-11.5 180"/>
            <geom type="capsule" fromto="0 0 0 0 0.0052 0.1349" size="0.012" rgba="0.84 0.84 0.84 1"/>
            <body name="Wrist_Pitch_Roll_2" pos="0 0.0052 0.1349" euler="-90 0 0">
              <joint name="Wrist_Pitch_2" type="hinge" axis="1 0 0" range="-103.1 103.1"/>
              <geom type="capsule" fromto="0 0 0 0 -0.0601 0" size="0.010" rgba="0.80 0.80 0.80 1"/>
              <body name="Fixed_Jaw_2" pos="0 -0.0601 0" euler="0 90 0">
                <joint name="Wrist_Roll_2" type="hinge" axis="0 -1 0" range="-180 180"/>
                <geom type="box" size="0.016 0.022 0.005" pos="0 -0.012 0" rgba="0.55 0.55 0.85 1"/>
                <geom type="box" size="0.003 0.018 0.005" pos="0.013 -0.038 0" rgba="0.55 0.55 0.85 1"/>
                <geom type="box" size="0.003 0.018 0.005" pos="-0.013 -0.038 0" rgba="0.55 0.55 0.85 1"/>
                <site name="ee_site" pos="0 0 0" size="0.009" rgba="1 0 0 1"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

JOINT_NAMES = ["Rotation_2", "Pitch_2", "Elbow_2", "Wrist_Pitch_2", "Wrist_Roll_2"]


# ═══════════════════════════════════════════════════════════════════════════
# Motor Mappings — OLD (broken) vs NEW (fixed)
# ═══════════════════════════════════════════════════════════════════════════

def mjcf_to_motor_OLD(q_deg):
    """OLD mapping (left-arm convention, WRONG for right arm).
    This was the original code that caused the flip."""
    out = q_deg.copy()
    out[1] = 90.0 - out[1]   # Pitch_R -> shoulder_lift
    out[2] = out[2] - 90.0   # Elbow_R -> elbow_flex          # Wrist_Roll:  NOT negated
    return out


def mjcf_to_motor_NEW(q_deg):
    """NEW mapping (right-arm convention, CORRECT for right arm).
    This is the current code in control_single_bus.py."""
    out = q_deg.copy()
    out[0] = -out[0]            # Rotation:    negated (mirrored)
    out[1] = out[1] - 90.0      # Pitch_2  →  shoulder_lift
    out[2] = 90.0 - out[2]      # Elbow_2  →  elbow_flex
    out[3] = -out[3]            # Wrist_Pitch: negated
    out[4] = -out[4]            # Wrist_Roll:  negated
    return out


def motor_to_mjcf_PHYSICAL(m_deg):
    """Inverse of the NEW mapping — models what the real right arm does
    when it receives a motor command. The physical motor is wired with
    the right-arm convention, so this is always the correct inverse."""
    q = m_deg.copy()
    q[0] = -m_deg[0]
    q[1] = m_deg[1] + 90.0
    q[2] = 90.0 - m_deg[2]
    q[3] = -m_deg[3]
    q[4] = -m_deg[4]
    return q


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_model():
    model = mujoco.MjModel.from_xml_string(MJCF_XML)
    data = mujoco.MjData(model)
    return model, data


def get_ee_pos(model, data):
    return data.site_xpos[model.site("ee_site").id].copy()


def set_joints(data, q_rad):
    data.qpos[:5] = q_rad


def get_body_positions(model, data):
    """Get world positions of all arm bodies for skeleton drawing."""
    names = ["Base_2", "Rotation_Pitch_2", "Upper_Arm_2", "Lower_Arm_2",
             "Wrist_Pitch_Roll_2", "Fixed_Jaw_2"]
    positions = []
    for name in names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        positions.append(data.xpos[body_id].copy())
    return np.array(positions)


# ═══════════════════════════════════════════════════════════════════════════
# Test targets
# ═══════════════════════════════════════════════════════════════════════════

TARGETS = [
    ("Forward",      [0.0, -0.30,  0.01]),
    ("Forward+up",   [0.0, -0.20,  0.20]),
    ("Forward+down", [0.0, -0.20, -0.05]),
    ("Left",         [0.15, -0.20,  0.05]),
    ("Right",        [-0.15, -0.20, 0.05]),
    ("Close",        [0.0, -0.15,  0.10]),
]


# ═══════════════════════════════════════════════════════════════════════════
# Main comparison
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    print("Loading models...")
    model, data = load_model()
    ik = IK_SO101()

    print(f"MuJoCo: {model.njnt} joints | Pinocchio: {ik.model.nq} DOF\n")

    # ── Run comparison for each target ────────────────────────────────
    results = []  # (name, tgt_w, q_ik_deg, old_ee, new_ee, old_skel, new_skel, old_err, new_err)

    header = f"{'Target':15s} │ {'OLD err (mm)':>12s} │ {'NEW err (mm)':>12s} │ {'Result':8s}"
    print("=" * len(header))
    print(" OLD vs NEW mjcf_to_motor — MuJoCo FK Comparison")
    print("=" * len(header))
    print(header)
    print("─" * len(header))

    for name, tgt_b2 in TARGETS:
        traj = ik.generate_ik(tgt_b2, [0, 0, 0])
        if not traj:
            print(f"  {name}: IK failed — skipping")
            continue

        q_ik_rad = traj[-1]
        q_ik_deg = np.rad2deg(q_ik_rad)
        tgt_w = ik.base2_to_world(tgt_b2)

        # ── OLD mapping pipeline ──────────────────────────────────────
        # IK → old mjcf_to_motor → physical motor (right arm) interprets it
        motor_cmd_old = mjcf_to_motor_OLD(q_ik_deg)
        q_physical_old_deg = motor_to_mjcf_PHYSICAL(motor_cmd_old)
        q_physical_old_rad = np.deg2rad(q_physical_old_deg)

        # Clamp to joint limits so MuJoCo doesn't clip silently
        for i in range(5):
            jnt = model.joint(i)
            q_physical_old_rad[i] = np.clip(q_physical_old_rad[i], jnt.range[0], jnt.range[1])

        set_joints(data, q_physical_old_rad)
        mujoco.mj_forward(model, data)
        old_ee = get_ee_pos(model, data)
        old_skel = get_body_positions(model, data)
        old_err_mm = np.linalg.norm(old_ee - tgt_w) * 1000

        # ── NEW mapping pipeline ──────────────────────────────────────
        motor_cmd_new = mjcf_to_motor_NEW(q_ik_deg)
        q_physical_new_deg = motor_to_mjcf_PHYSICAL(motor_cmd_new)
        q_physical_new_rad = np.deg2rad(q_physical_new_deg)

        set_joints(data, q_physical_new_rad)
        mujoco.mj_forward(model, data)
        new_ee = get_ee_pos(model, data)
        new_skel = get_body_positions(model, data)
        new_err_mm = np.linalg.norm(new_ee - tgt_w) * 1000

        results.append((name, tgt_w, q_ik_deg, old_ee, new_ee,
                         old_skel, new_skel, old_err_mm, new_err_mm))

        status = "PASS" if new_err_mm < 5.0 else "FAIL"
        old_status = "FAIL" if old_err_mm > 5.0 else "ok"
        print(f"{name:15s} │ {old_err_mm:9.1f} {old_status:4s} │ "
              f"{new_err_mm:9.1f} {'PASS':4s} │ {status:8s}")

        # Joint-level detail
        print(f"  {'':15s}   IK joints:       [{', '.join(f'{d:7.1f}' for d in q_ik_deg)}]")
        print(f"  {'':15s}   OLD motor cmd:   [{', '.join(f'{d:7.1f}' for d in motor_cmd_old)}]")
        print(f"  {'':15s}   OLD physical q:  [{', '.join(f'{d:7.1f}' for d in q_physical_old_deg)}]")
        print(f"  {'':15s}   NEW motor cmd:   [{', '.join(f'{d:7.1f}' for d in motor_cmd_new)}]")
        print(f"  {'':15s}   NEW physical q:  [{', '.join(f'{d:7.1f}' for d in q_physical_new_deg)}]")
        # Show per-joint differences
        diff = q_physical_old_deg - q_ik_deg
        flipped = [i for i, d in enumerate(diff) if abs(d) > 1.0]
        if flipped:
            flip_names = [JOINT_NAMES[i] for i in flipped]
            print(f"  {'':15s}   FLIPPED joints:  {', '.join(flip_names)}")
        print()

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 65)
    old_errs = [r[7] for r in results]
    new_errs = [r[8] for r in results]
    print(f"  OLD mapping:  avg={np.mean(old_errs):.1f}mm  "
          f"max={np.max(old_errs):.1f}mm  → {'ALL FAIL' if all(e > 5 for e in old_errs) else 'MIXED'}")
    print(f"  NEW mapping:  avg={np.mean(new_errs):.1f}mm  "
          f"max={np.max(new_errs):.1f}mm  → {'ALL PASS' if all(e < 5 for e in new_errs) else 'SOME FAIL'}")
    print("=" * 65)

    # ═════════════════════════════════════════════════════════════════
    # Visualization — side-by-side OLD vs NEW for each target
    # ═════════════════════════════════════════════════════════════════
    n = len(results)
    fig = plt.figure(figsize=(12, 5 * n))
    fig.suptitle("OLD (left-arm) vs NEW (right-arm) Motor Mapping\n"
                 "MuJoCo FK — What the real arm would do",
                 fontsize=15, fontweight="bold", y=0.995)

    for idx, (name, tgt_w, q_ik_deg, old_ee, new_ee,
              old_skel, new_skel, old_err, new_err) in enumerate(results):

        # ── OLD mapping subplot ───────────────────────────────────
        ax_old = fig.add_subplot(n, 2, idx * 2 + 1, projection="3d")
        _draw_arm(ax_old, old_skel, old_ee, tgt_w, old_err,
                  f"OLD mapping — {name}", color_arm="orangered", color_ee="red")

        # ── NEW mapping subplot ───────────────────────────────────
        ax_new = fig.add_subplot(n, 2, idx * 2 + 2, projection="3d")
        _draw_arm(ax_new, new_skel, new_ee, tgt_w, new_err,
                  f"NEW mapping — {name}", color_arm="steelblue", color_ee="blue")

        # Match axis limits between left/right
        all_pts = np.vstack([old_skel, new_skel, [tgt_w], [old_ee], [new_ee]])
        mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
        span = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max(), 0.20) / 2 * 1.3
        for ax in [ax_old, ax_new]:
            ax.set_xlim(mid[0] - span, mid[0] + span)
            ax.set_ylim(mid[1] - span, mid[1] + span)
            ax.set_zlim(mid[2] - span, mid[2] + span)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "old_vs_new_mapping.png"
    fig.savefig(str(out_path), dpi=140)
    plt.close(fig)
    print(f"\nSaved comparison visualization to {out_path}")


def _draw_arm(ax, skeleton, ee_pos, target_pos, err_mm, title, color_arm, color_ee):
    """Draw an arm skeleton + target + EE on a 3D axes."""
    # Arm skeleton
    ax.plot(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2],
            "o-", color=color_arm, linewidth=2.5, markersize=6, label="Arm")

    # Base
    ax.scatter(*skeleton[0], c="gold", s=90, marker="s", zorder=5, label="Base_2")

    # EE
    ax.scatter(*ee_pos, c=color_ee, s=80, marker="^", zorder=5,
               label=f"EE (err={err_mm:.0f}mm)")

    # Target
    ax.scatter(*target_pos, c="lime", s=130, marker="*", edgecolors="green",
               linewidths=1.0, zorder=5, label="Target")

    # Line from EE to target
    ax.plot([ee_pos[0], target_pos[0]],
            [ee_pos[1], target_pos[1]],
            [ee_pos[2], target_pos[2]],
            "--", color=color_ee, alpha=0.4, linewidth=1)

    # Pass/fail label
    status = "PASS" if err_mm < 5.0 else f"FAIL ({err_mm:.0f}mm)"
    status_color = "green" if err_mm < 5.0 else "red"
    ax.set_title(f"{title}\n{status}", fontsize=10, color=status_color,
                 fontweight="bold")

    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    ax.set_zlabel("Z (m)", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7)


if __name__ == "__main__":
    main()
