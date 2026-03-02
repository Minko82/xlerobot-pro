#!/usr/bin/env python3
"""Simulate the motor-mapping fix for the right arm (arm 2).

No hardware needed — uses Pinocchio FK to show where the EE ends up under
both the OLD (left-arm) and NEW (right-arm, mirrored) motor mappings.

Physical model:
  The right arm's motors are mounted mirrored vs. the left arm.
  motor_to_mjcf_NEW correctly describes how motor degrees map to
  physical URDF angles on the RIGHT arm.

  OLD mapping path (broken):
      IK → mjcf_to_motor_OLD → send to right arm → physical URDF angles
      = motor_to_mjcf_NEW(mjcf_to_motor_OLD(q_ik))  ← WRONG pose

  NEW mapping path (fixed):
      IK → mjcf_to_motor_NEW → send to right arm → physical URDF angles
      = motor_to_mjcf_NEW(mjcf_to_motor_NEW(q_ik)) = q_ik  ← CORRECT

Usage:
    cd cube-vision
    python test_motor_mapping_sim.py
"""

from __future__ import annotations
import numpy as np
import pinocchio as pin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

from ik_solver import IK_SO101

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0

# ── Motor mappings ────────────────────────────────────────────────────

def mjcf_to_motor_OLD(q_deg: np.ndarray) -> np.ndarray:
    """LEFT-arm mapping (the old, incorrect one for the right arm)."""
    out = q_deg.copy()
    out[1] = 90.0 - out[1]   # Pitch -> shoulder_lift
    out[2] = out[2] - 90.0   # Elbow -> elbow_flex
    return out

def mjcf_to_motor_NEW(q_deg: np.ndarray) -> np.ndarray:
    """RIGHT-arm mapping (mirrored — the fix)."""
    out = q_deg.copy()
    out[0] = -out[0]          # Rotation: negated
    out[1] = out[1] - 90.0    # Pitch:    negated offset
    out[2] = 90.0 - out[2]    # Elbow:    negated offset
    out[3] = -out[3]          # Wrist_Pitch: negated
    out[4] = -out[4]          # Wrist_Roll:  negated
    return out

def motor_to_mjcf_NEW(q_deg: np.ndarray) -> np.ndarray:
    """Physical right-arm: motor degrees → actual URDF angles the arm achieves."""
    out = q_deg.copy()
    out[0] = -out[0]
    out[1] = out[1] + 90.0
    out[2] = 90.0 - out[2]
    out[3] = -out[3]
    out[4] = -out[4]
    return out

# ── FK / plotting helpers ─────────────────────────────────────────────

_SKELETON_FRAMES = [
    "Base_2", "Rotation_Pitch_2", "Upper_Arm_2",
    "Lower_Arm_2", "Wrist_Pitch_Roll_2", "Fixed_Jaw_2",
]

def fk_skeleton(ik: IK_SO101, q_rad: np.ndarray) -> list[np.ndarray]:
    """FK → list of world-frame positions along the arm skeleton."""
    q_clamped = np.clip(q_rad,
                        ik.model.lowerPositionLimit,
                        ik.model.upperPositionLimit)
    pin.forwardKinematics(ik.model, ik.data, q_clamped)
    pin.updateFramePlacements(ik.model, ik.data)
    return [ik.data.oMf[ik.model.getFrameId(n)].translation.copy()
            for n in _SKELETON_FRAMES]

def to_base2(ik: IK_SO101, pts: list[np.ndarray]) -> np.ndarray:
    """Convert world positions to Base_2 frame."""
    return np.array([ik._base2_R.T @ (p - ik._base2_t) for p in pts])

def plot_arm(ax, pts_b2: np.ndarray, color, label, lw=2.5, alpha=1.0):
    ax.plot(pts_b2[:,0], pts_b2[:,1], pts_b2[:,2],
            "-o", color=color, linewidth=lw, markersize=5, alpha=alpha, label=label)
    ax.scatter(*pts_b2[-1], c=color, s=100, marker="^", zorder=5)

# ── Main ──────────────────────────────────────────────────────────────

TARGETS = [
    ("Forward 20cm",   [0.0,  -0.20, 0.05]),
    ("Forward+left",   [0.10, -0.20, 0.05]),
    ("Forward+right",  [-0.10, -0.20, 0.05]),
    ("Forward+up",     [0.0,  -0.15, 0.15]),
    ("Forward+down",   [0.0,  -0.20, -0.05]),
    ("Near surface",   [0.0,  -0.20, 0.00]),
]

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ik = IK_SO101()

    print("=" * 70)
    print("  Motor-Mapping Simulation: OLD (left-arm) vs NEW (right-arm)")
    print("  Physical model: right arm motors are mirrored.")
    print("=" * 70)
    print(f"  {'Target':<18s} {'IK err':>8s}   {'OLD phys err':>13s}   {'NEW phys err':>13s}")
    print("  " + "-" * 60)

    n = len(TARGETS)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n),
                             subplot_kw={"projection": "3d"})
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (name, target) in enumerate(TARGETS):
        traj = ik.generate_ik(target_xyz=target, gripper_offset_xyz=[0, 0, 0])
        if not traj:
            print(f"  {name:<18s} IK failed")
            continue

        q_ik_rad = traj[-1]
        q_ik_deg = q_ik_rad * RAD2DEG
        target_b2 = np.array(target)
        target_world = ik.base2_to_world(target_b2)

        # ── OLD mapping: what the right arm PHYSICALLY does ──
        motor_old = mjcf_to_motor_OLD(q_ik_deg)
        phys_old_deg = motor_to_mjcf_NEW(motor_old)
        phys_old_rad = phys_old_deg * DEG2RAD

        skel_old = fk_skeleton(ik, phys_old_rad)
        ee_old_world = skel_old[-1]
        err_old = np.linalg.norm(ee_old_world - target_world) * 1000

        # ── NEW mapping: what the right arm PHYSICALLY does ──
        motor_new = mjcf_to_motor_NEW(q_ik_deg)
        phys_new_deg = motor_to_mjcf_NEW(motor_new)  # = q_ik_deg (identity)
        phys_new_rad = phys_new_deg * DEG2RAD

        skel_new = fk_skeleton(ik, phys_new_rad)
        ee_new_world = skel_new[-1]
        err_new = np.linalg.norm(ee_new_world - target_world) * 1000

        # IK reference skeleton
        skel_ik = fk_skeleton(ik, q_ik_rad)
        err_ik = np.linalg.norm(skel_ik[-1] - target_world) * 1000

        old_tag = FAIL if err_old > 5 else PASS
        new_tag = PASS if err_new < 5 else FAIL
        print(f"  {name:<18s} {err_ik:5.1f}mm   "
              f"{err_old:8.1f}mm {old_tag}   "
              f"{err_new:8.1f}mm {new_tag}")

        # ── Plot ──
        for col, (phys_skel, err, mapping_label) in enumerate([
            (skel_old, err_old, "OLD mapping (left-arm) on RIGHT arm"),
            (skel_new, err_new, "NEW mapping (right-arm, fixed)"),
        ]):
            ax = axes[row, col]

            pts_ik_b2 = to_base2(ik, skel_ik)
            pts_phys_b2 = to_base2(ik, phys_skel)

            # Base origin + axes
            axis_len = 0.07
            ax.scatter(0, 0, 0, c="black", s=60, marker="s")
            for v, c in [([1,0,0],"r"), ([0,1,0],"g"), ([0,0,1],"b")]:
                ax.quiver(0,0,0, *[vi*axis_len for vi in v],
                          color=c, linewidth=1.2, arrow_length_ratio=0.12)

            # Target
            ax.scatter(*target_b2, c="magenta", s=140, marker="*", zorder=10, label="Target")

            # IK ground truth (gray, transparent)
            plot_arm(ax, pts_ik_b2, "gray", "IK (correct)", lw=1.5, alpha=0.3)

            # Physical outcome
            clr = "green" if err < 5 else "red"
            plot_arm(ax, pts_phys_b2, clr, f"Physical (err={err:.0f}mm)")

            ax.set_title(f"{name}\n{mapping_label}", fontsize=9,
                         color="green" if err < 5 else "red", fontweight="bold")
            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.legend(fontsize=7, loc="upper left")

            # Equal scale
            all_pts = np.vstack([pts_ik_b2, pts_phys_b2, [target_b2], [np.zeros(3)]])
            mid = (all_pts.max(0) + all_pts.min(0)) / 2
            span = max((all_pts.max(0) - all_pts.min(0)).max(), 0.15) / 2 * 1.3
            ax.set_xlim(mid[0]-span, mid[0]+span)
            ax.set_ylim(mid[1]-span, mid[1]+span)
            ax.set_zlim(mid[2]-span, mid[2]+span)

    print("  " + "-" * 60)

    # ── Joint-level breakdown ──
    print("\n  Joint-level comparison (Forward 20cm):")
    traj = ik.generate_ik(target_xyz=[0.0, -0.20, 0.05], gripper_offset_xyz=[0,0,0])
    q_deg = traj[-1] * RAD2DEG
    joints = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
    motor_old = mjcf_to_motor_OLD(q_deg)
    phys_old = motor_to_mjcf_NEW(motor_old)
    motor_new = mjcf_to_motor_NEW(q_deg)
    phys_new = motor_to_mjcf_NEW(motor_new)

    print(f"  {'Joint':<14s} {'IK(URDF)':>10s} {'OLD motor':>10s} {'OLD phys':>10s} "
          f"{'NEW motor':>10s} {'NEW phys':>10s}")
    for i, jn in enumerate(joints):
        match = "  OK" if abs(phys_new[i] - q_deg[i]) < 0.01 else " BAD"
        flip = " FLIP!" if abs(phys_old[i] - q_deg[i]) > 1.0 else ""
        print(f"  {jn:<14s} {q_deg[i]:>9.1f}° {motor_old[i]:>9.1f}° "
              f"{phys_old[i]:>9.1f}°{flip} {motor_new[i]:>9.1f}° "
              f"{phys_new[i]:>9.1f}°{match}")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle("Motor Mapping: OLD (left-arm) vs NEW (right-arm) on physical right arm",
                 fontsize=13, fontweight="bold")

    out_path = OUTPUT_DIR / "motor_mapping_sim.png"
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"\n  Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
